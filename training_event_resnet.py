import gc
import os
import time
import argparse
import json
import traceback
import multiprocessing
import collections

import pandas as pd
import numpy as np
import tqdm
import torch
import h5py

import config
import manager_folds
import manager_models
import metrics
import logging_memory_utils
import convert_to_h5py_naive
import convert_to_h5py_splitted
import model_unet
import kaggle_ap_detection
import prediction_event_generator

def focal_loss(preds: torch.Tensor, ground_truth: torch.Tensor):
    assert preds.shape == ground_truth.shape, "preds.shape = {}, ground_truth.shape = {}".format(preds.shape, ground_truth.shape)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(preds, ground_truth, reduction="none")
    with torch.no_grad():
        weight = ground_truth * (positive_weight - 1) + 1
    assert weight.shape == bce.shape, "weight.shape = {}, bce.shape = {}".format(weight.shape, bce.shape)
    return torch.sum(
        ((torch.sigmoid(preds) - ground_truth) ** 2) * bce * weight
    ) / 461900.0 # mean length as shown in check_series_properties.py


def single_training_step(model_: torch.nn.Module, optimizer_: torch.optim.Optimizer,
                            accel_data_batch: torch.Tensor, labels_batch: torch.Tensor):
    optimizer_.zero_grad()
    pred_logits = model_(accel_data_batch) # shape (batch_size, 2, T), where batch_size = 1
    loss = focal_loss(pred_logits, labels_batch)
    loss.backward()
    optimizer_.step()

    with torch.no_grad():
        preds = pred_logits > 0.0

    return loss.item(), preds.to(torch.long)

def sample_cutmix(batch_entry, num_mix, extra_shuffle_indices, k):
    accel_data_batch_list = []
    events_list = []
    current_length = 0

    is_event = np.random.rand() < 0.5
    while current_length < length:
        type_event = "event" if is_event else "non_event"

        appended = False
        while not appended:
            choice = np.random.randint(0, num_mix)
            if choice == 0:
                intervals = all_data_events[batch_entry][type_event]
            else:
                intervals = all_data_events[extra_shuffle_indices[choice - 1][k]][type_event]
            num_intervals = len(intervals)
            if len(intervals) == 0:
                continue

            interval = intervals[np.random.randint(0, num_intervals)]

            low = interval[0]
            high = interval[1]
            max_diff = min(int(high - low) // 4, 720)
            low += np.random.randint(0, max_diff)
            high -= np.random.randint(0, max_diff)

            accel_data_batch_list.append(all_data[batch_entry]["accel"][:, low:high])
            events_list.append(is_event)
            appended = True

        is_event = not is_event
        current_length += accel_data_batch_list[-1].shape[-1]

    accel_data_batch = np.concatenate(accel_data_batch_list, axis=-1)
    labels_batch = np.zeros(shape=(2, accel_data_batch.shape[-1]), dtype=np.int32) # first dimension is onset, wakeup
    current_length = accel_data_batch.shape[-1]

    cum_length = 0
    for k in range(len(events_list)):
        if events_list[k]:
            onset = cum_length
            wakeup = onset + accel_data_batch_list[k].shape[-1]

            for tolerance in tolerances:
                onset_low = max(onset - tolerance + 1, 0)
                onset_high = min(onset + tolerance, current_length)

                wakeup_low = max(wakeup - tolerance + 1, 0)
                wakeup_high = min(wakeup + tolerance, current_length)

                if onset_low < onset_high:
                    labels_batch[0, onset_low:onset_high] = labels_batch[0, onset_low:onset_high] + 1

                if wakeup_low < wakeup_high:
                    labels_batch[1, wakeup_low:wakeup_high] = labels_batch[1, wakeup_low:wakeup_high] + 1

        cum_length += accel_data_batch_list[k].shape[-1]
    labels_batch = labels_batch.astype(np.float32) / len(tolerances)

    # randomly pick a segment to match length
    left_erosion = np.random.randint(0, accel_data_batch.shape[-1] - length + 1)
    accel_data_batch = accel_data_batch[:, left_erosion:(left_erosion + length)]
    labels_batch = labels_batch[:, left_erosion:(left_erosion + length)]

    return accel_data_batch, labels_batch

def training_step(record: bool):
    if record:
        for key in train_metrics:
            train_metrics[key].reset()

    # shuffle
    shuffle_indices = np.random.permutation(len(training_entries))
    num_mix = 11
    extra_shuffle_indices = [np.random.permutation(len(training_entries)) for _ in range(num_mix - 1)]

    # training
    trained = 0
    with tqdm.tqdm(total=len(shuffle_indices)) as pbar:
        for k in range(len(shuffle_indices)):
            batch_entry = training_entries[shuffle_indices[k]] # series ids
            accel_data_batch, labels_batch = sample_cutmix(batch_entry, num_mix, extra_shuffle_indices, k)
            accel_data_batch2, labels_batch2 = sample_cutmix(batch_entry, num_mix, extra_shuffle_indices, k)

            lam = np.random.beta(0.2, 0.2)
            accel_data_batch = lam * accel_data_batch + (1 - lam) * accel_data_batch2
            labels_batch = lam * labels_batch + (1 - lam) * labels_batch2

            accel_data_batch = torch.tensor(accel_data_batch, dtype=torch.float32, device=config.device)
            labels_batch = torch.tensor(labels_batch, dtype=torch.float32, device=config.device)

            accel_data_batch = accel_data_batch.unsqueeze(0)
            labels_batch = labels_batch.unsqueeze(0)

            # train model now
            loss, preds = single_training_step(model, optimizer,
                                               accel_data_batch,
                                               labels_batch)
            time.sleep(0.15)

            # record
            if record:
                with torch.no_grad():
                    train_metrics["loss"].add(loss, 1)
                    train_metrics["metric"].add(preds, (labels_batch > 0.5).to(torch.long))

            trained += 1
            pbar.update(1)

    if record:
        current_metrics = {}
        for key in train_metrics:
            train_metrics[key].write_to_dict(current_metrics)

        for key in current_metrics:
            train_history[key].append(current_metrics[key])

def single_validation_step(model_: torch.nn.Module, accel_data_batch: torch.Tensor, labels_batch: torch.Tensor,
                           pad_left, pad_right):
    accel_data_batch = torch.nn.functional.pad(accel_data_batch, (pad_left, pad_right), mode="replicate")
    pred_logits = model_(accel_data_batch)  # shape (batch_size, 1, T), where batch_size = 1
    pred_logits = pred_logits[..., pad_left:(pred_logits.shape[-1] - pad_right)]
    loss = focal_loss(pred_logits, labels_batch)
    preds = pred_logits > 0.0
    pred_probas = torch.sigmoid(pred_logits)

    return loss.item(), preds.to(torch.long), pred_probas

def validation_step():
    for key in val_metrics:
        val_metrics[key].reset()

    # validation
    with tqdm.tqdm(total=len(validation_entries)) as pbar:
        for k in range(len(validation_entries)):
            batch_entry = validation_entries[k]  # series ids

            # load the batch
            accel_data_batch = all_data[batch_entry]["accel"]
            labels_batch = np.zeros(shape=(2, accel_data_batch.shape[-1]), dtype=np.int32)  # first dimension is onset, wakeup

            # load the events
            for event in all_data[batch_entry]["event"]:
                onset = event[0]
                wakeup = event[1]

                for tolerance in tolerances:
                    onset_low = max(onset - tolerance + 1, 0)
                    onset_high = min(onset + tolerance, accel_data_batch.shape[-1])

                    wakeup_low = max(wakeup - tolerance + 1, 0)
                    wakeup_high = min(wakeup + tolerance, accel_data_batch.shape[-1])

                    if onset_low < onset_high:
                        labels_batch[0, onset_low:onset_high] = labels_batch[0, onset_low:onset_high] + 1

                    if wakeup_low < wakeup_high:
                        labels_batch[1, wakeup_low:wakeup_high] = labels_batch[1, wakeup_low:wakeup_high] + 1
            labels_batch = labels_batch.astype(np.float32) / len(tolerances)

            accel_data_batch = torch.tensor(accel_data_batch, dtype=torch.float32, device=config.device).unsqueeze(0)
            labels_batch = torch.tensor(labels_batch, dtype=torch.float32, device=config.device).unsqueeze(0)

            # pad such that lengths is a multiple of 16
            pad_length = 16 - (accel_data_batch.shape[-1] % 16)
            if pad_length == 16:
                pad_length = 0
            pad_left = pad_length // 2
            pad_right = pad_length - pad_left

            # train model now
            loss, preds, pred_probas = single_validation_step(model, accel_data_batch, labels_batch, pad_left, pad_right)

            # record
            with torch.no_grad():
                val_metrics["loss"].add(loss, 1)
                val_metrics["metric"].add(preds, (labels_batch > 0.5).to(torch.long))
                val_events_gen.record(batch_entry, pred_probas, tolerances)
            pbar.update(1)

    current_metrics = {}
    for key in val_metrics:
        val_metrics[key].write_to_dict(current_metrics)

    for key in current_metrics:
        val_history[key].append(current_metrics[key])

    ctime = time.time()
    val_events_preds = val_events_gen.convert_to_df()
    ap_score = kaggle_ap_detection.score(solution=val_events_ground_truth,
                              submission=val_events_preds,
                              tolerances={"onset": tolerances, "wakeup": tolerances},
                              series_id_column_name="series_id",
                              time_column_name="step",
                              event_column_name="event",
                              score_column_name="score")
    val_history["val_ap"].append(ap_score)
    print("AP computation time elapsed: {}".format(time.time() - ctime))

def print_history(metrics_history):
    for key in metrics_history:
        print("{}      {}".format(key, metrics_history[key][-1]))

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    all_data = convert_to_h5py_naive.load_all_data_into_dict()
    all_data_events = convert_to_h5py_splitted.load_all_data_into_dict()
    tolerances = [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]

    parser = argparse.ArgumentParser(description="Train a injury prediction model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for. Default 100.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate to use. Default 1e-3.")
    parser.add_argument("--momentum", type=float, default=0.99, help="Momentum to use. Default 0.9. This would be the momentum for SGD, and beta1 for Adam.")
    parser.add_argument("--second_momentum", type=float, default=0.999, help="Second momentum to use. Default 0.999. This would be beta2 for Adam. Ignored if SGD.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Which optimizer to use. Available options: adam, sgd. Default adam.")
    parser.add_argument("--epochs_per_save", type=int, default=2, help="Number of epochs between saves. Default 2.")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[1, 6, 8, 23, 8],
                        help="Number of hidden 2d blocks for ResNet backbone.")
    parser.add_argument("--hidden_channels", type=int, default=32, help="Number of hidden channels. Default None.")
    parser.add_argument("--bottleneck_factor", type=int, default=4, help="The bottleneck factor of the ResNet backbone. Default 4.")
    parser.add_argument("--squeeze_excitation", action="store_false", help="Whether to use squeeze and excitation. Default True.")
    parser.add_argument("--disable_odd_random_shift", action="store_true", help="Whether to disable odd random shift. Default False.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate. Default 0.0.")
    parser.add_argument("--length", default=700000, help="The fixed length of the training. Default 700000.")
    parser.add_argument("--num_extra_steps", type=int, default=0, help="Extra steps of gradient descent before the usual step in an epoch. Default 0.")
    parser.add_argument("--positive_weight", type=float, default=1.0, help="Positive weight")
    manager_folds.add_argparse_arguments(parser)
    manager_models.add_argparse_arguments(parser)
    config.add_argparse_arguments(parser)
    args = parser.parse_args()

    # check entries
    training_entries, validation_entries, train_dset_name, val_dset_name = manager_folds.parse_args(args)
    assert type(training_entries) == list
    assert type(validation_entries) == list
    print("Training dataset: {}".format(train_dset_name))
    print("Validation dataset: {}".format(val_dset_name))
    val_events_ground_truth = pd.read_csv(os.path.join("data", "train_events.csv")).dropna()
    val_events_ground_truth = val_events_ground_truth.loc[val_events_ground_truth["series_id"].isin(validation_entries)]
    assert set(val_events_ground_truth["series_id"].unique()).issubset(set(validation_entries))

    # initialize gpu
    config.parse_args(args)

    # get model directories
    model_dir, prev_model_dir = manager_models.parse_args(args)

    # obtain model and training parameters
    epochs = args.epochs
    learning_rate = args.learning_rate
    momentum = args.momentum
    second_momentum = args.second_momentum
    optimizer_type = args.optimizer
    epochs_per_save = args.epochs_per_save
    hidden_blocks = args.hidden_blocks
    hidden_channels = args.hidden_channels
    bottleneck_factor = args.bottleneck_factor
    squeeze_excitation = args.squeeze_excitation
    disable_odd_random_shift = args.disable_odd_random_shift
    dropout = args.dropout
    length = args.length
    num_extra_steps = args.num_extra_steps
    positive_weight = args.positive_weight

    print("Epochs: " + str(epochs))
    print("Learning rate: " + str(learning_rate))
    print("Momentum: " + str(momentum))
    print("Second momentum: " + str(second_momentum))
    print("Optimizer: " + optimizer_type)
    print("Dropout: " + str(dropout))
    #model_resnet_old.BATCH_NORM_MOMENTUM = 1 - momentum

    # initialize model
    model = model_unet.Unet(2, hidden_channels, kernel_size=11, blocks=hidden_blocks,
                            bottleneck_factor=bottleneck_factor, squeeze_excitation=squeeze_excitation,
                            squeeze_excitation_bottleneck_factor=4, odd_random_shift_training=(not disable_odd_random_shift),
                            dropout=dropout, out_channels=2)
    model = model.to(config.device)

    # initialize optimizer
    print("Learning rate: " + str(learning_rate))
    print("Momentum: " + str(momentum))
    print("Second momentum: " + str(second_momentum))
    print("Optimizer: " + optimizer_type)
    print("Positive weight: " + str(positive_weight))
    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(momentum, second_momentum))
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        print("Invalid optimizer. The available options are: adam, sgd.")
        exit(1)

    # Load previous model checkpoint if available
    if prev_model_dir is None:
        warmup_steps = 2
        for g in optimizer.param_groups:
            g["lr"] = 0.0
    else:
        warmup_steps = 0
        model_checkpoint_path = os.path.join(prev_model_dir, "model.pt")
        optimizer_checkpoint_path = os.path.join(prev_model_dir, "optimizer.pt")

        model.load_state_dict(torch.load(model_checkpoint_path))
        optimizer.load_state_dict(torch.load(optimizer_checkpoint_path))

        for g in optimizer.param_groups:
            g["lr"] = learning_rate
            if optimizer_type == "adam":
                g["betas"] = (momentum, second_momentum)
            elif optimizer_type == "sgd":
                g["momentum"] = momentum

    model_config = {
        "model": "Event deep learning segmentation approach",
        "epochs": epochs,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "second_momentum": second_momentum,
        "optimizer": optimizer_type,
        "epochs_per_save": epochs_per_save,
        "hidden_blocks": hidden_blocks,
        "hidden_channels": hidden_channels,
        "bottleneck_factor": bottleneck_factor,
        "squeeze_excitation": squeeze_excitation,
        "disable_odd_random_shift": disable_odd_random_shift,
        "dropout": dropout,
        "length": length,
        "num_extra_steps": num_extra_steps,
        "positive_weight": positive_weight
    }

    # Save the model config
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    # Create the metrics
    train_history = collections.defaultdict(list)
    train_metrics = {}
    val_history = collections.defaultdict(list)
    val_metrics = {}
    train_metrics["loss"] = metrics.NumericalMetric("train_loss")
    train_metrics["metric"] = metrics.BinaryMetrics("train_metric")
    val_metrics["loss"] = metrics.NumericalMetric("val_loss")
    val_metrics["metric"] = metrics.BinaryMetrics("val_metric")
    val_events_gen = prediction_event_generator.EventsRecorder()

    # Compile
    #single_training_step_compile = torch.compile(single_training_step)

    # Start training loop
    print("Training for {} epochs......".format(epochs))
    memory_logger = logging_memory_utils.obtain_memory_logger(model_dir)

    try:
        for epoch in range(epochs):
            memory_logger.log("Epoch {}".format(epoch))
            print("------------------------------------ Epoch {} ------------------------------------".format(epoch))
            model.train()
            print("Running {} extra steps of gradient descent.".format(num_extra_steps))
            for k in range(num_extra_steps):
                training_step(record=False)

            print("Running the usual step of gradient descent.")
            training_step(record=True)

            if warmup_steps > 0:
                warmup_steps -= 1
                if warmup_steps == 0:
                    for g in optimizer.param_groups:
                        g["lr"] = learning_rate

            # switch model to eval mode, and reset all running stats for batchnorm layers
            model.eval()
            with torch.no_grad():
                validation_step()

            print()
            print_history(train_history)
            print_history(val_history)
            # save metrics
            train_df = pd.DataFrame(train_history)
            val_df = pd.DataFrame(val_history)
            train_df.to_csv(os.path.join(model_dir, "train_metrics.csv"), index=True)
            val_df.to_csv(os.path.join(model_dir, "val_metrics.csv"), index=True)

            if epoch % epochs_per_save == 0:
                torch.save(model.state_dict(), os.path.join(model_dir, "model_{}.pt".format(epoch)))
                torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer_{}.pt".format(epoch)))
            gc.collect()

        print("Training complete! Saving and finalizing...")
    except KeyboardInterrupt:
        print("Training interrupted! Saving and finalizing...")
    except Exception as e:
        print("Training interrupted due to exception! Saving and finalizing...")
        traceback.print_exc()
    # save model
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer.pt"))

    # save metrics
    if len(train_history) > 0:
        train_df = pd.DataFrame(train_history)
        val_df = pd.DataFrame(val_history)
        train_df.to_csv(os.path.join(model_dir, "train_metrics.csv"), index=True)
        val_df.to_csv(os.path.join(model_dir, "val_metrics.csv"), index=True)

    memory_logger.close()

    """if use_async_sampler:
        image_organ_sampler_async.clean_and_destroy_sampler()"""