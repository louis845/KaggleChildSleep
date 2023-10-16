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
import convert_to_good_events
import model_unet
import model_attention_unet

# same as training_clean_data.py, but with 3 factor downsampling at the first layer

def ce_loss(preds: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor = None):
    assert preds.shape == ground_truth.shape, "preds.shape = {}, ground_truth.shape = {}".format(preds.shape,
                                                                                                 ground_truth.shape)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(preds, ground_truth, reduction="none")
    if mask is None:
        return torch.mean(bce)
    else:
        return torch.mean(bce * mask)


def single_training_step(model_: torch.nn.Module, optimizer_: torch.optim.Optimizer,
                            accel_data_batch: torch.Tensor, labels_batch: torch.Tensor):
    optimizer_.zero_grad()
    if use_deep_supervision:
        pred_logits_small, pred_logits_mid, pred_logits = model_(accel_data_batch, True) # shape (batch_size, 1, T), where batch_size = 1
        loss = (ce_loss(pred_logits, labels_batch) + ce_loss(pred_logits_small, labels_batch) + ce_loss(pred_logits_mid, labels_batch)) / 3.0
    else:
        pred_logits = model_(accel_data_batch)  # shape (batch_size, 1, T), where batch_size = 1
        loss = ce_loss(pred_logits, labels_batch)
    loss.backward()
    optimizer_.step()

    with torch.no_grad():
        preds = pred_logits > 0.0

    return loss.item(), preds.to(torch.long)

def training_step(record: bool):
    if record:
        for key in train_metrics:
            train_metrics[key].reset()

    # shuffle
    training_sampler.shuffle()
    if use_mixup_training:
        training_sampler2.shuffle()

    # training
    with tqdm.tqdm(total=training_sampler.series_remaining()) as pbar:
        while training_sampler.series_remaining() > 0:
            accel_data_batch, labels_batch, increment = training_sampler.get_next(length, all_data)
            if use_mixup_training:
                accel_data_batch2, labels_batch2, _ = training_sampler2.get_next(length, all_data)
                lam = np.random.beta(0.2, 0.2)
                accel_data_batch = lam * accel_data_batch + (1.0 - lam) * accel_data_batch2
                labels_batch = lam * labels_batch.astype(dtype=np.float32) + (1.0 - lam) * labels_batch2.astype(dtype=np.float32)

            accel_data_batch_torch = torch.tensor(accel_data_batch, dtype=torch.float32, device=config.device)
            labels_batch_torch = torch.tensor(labels_batch, dtype=torch.float32, device=config.device)
            accel_data_batch_torch = accel_data_batch_torch.unsqueeze(0)
            labels_batch_torch = labels_batch_torch.unsqueeze(0).unsqueeze(0)

            # train model now
            loss, preds = single_training_step(model, optimizer,
                                               accel_data_batch_torch,
                                               labels_batch_torch)
            time.sleep(0.6)

            # record
            if record:
                with torch.no_grad():
                    train_metrics["loss"].add(loss, 1)
                    train_metrics["metric"].add(preds, (labels_batch_torch > 0.5).to(torch.long))

            pbar.update(increment)

    if record:
        current_metrics = {}
        for key in train_metrics:
            train_metrics[key].write_to_dict(current_metrics)

        for key in current_metrics:
            train_history[key].append(current_metrics[key])

def single_validation_step(model_: torch.nn.Module, accel_data_batch: torch.Tensor,
                           labels_batch: torch.Tensor, mask: torch.Tensor, more_mask: torch.Tensor,
                           pad_left, pad_right):
    with torch.no_grad():
        accel_data_batch = torch.nn.functional.pad(accel_data_batch, (pad_left, pad_right), mode="replicate")
        pred_logits = model_(accel_data_batch)  # shape (batch_size, 1, T), where batch_size = 1
        pred_logits = pred_logits[..., pad_left:(pred_logits.shape[-1] - pad_right)]
        loss = ce_loss(pred_logits, labels_batch)
        masked_loss = ce_loss(pred_logits, labels_batch, mask)
        more_masked_loss = ce_loss(pred_logits, labels_batch, more_mask)
        preds = pred_logits > 0.0

        return loss.item(), preds.to(torch.long), masked_loss.item(), more_masked_loss.item()

def single_validation_step_deep_supervision(model_: torch.nn.Module, accel_data_batch: torch.Tensor,
                           labels_batch: torch.Tensor, mask: torch.Tensor, more_mask: torch.Tensor,
                           pad_left, pad_right):
    with torch.no_grad():
        accel_data_batch = torch.nn.functional.pad(accel_data_batch, (pad_left, pad_right), mode="replicate")

        pred_logits_small, pred_logits_mid, pred_logits = model_(accel_data_batch, True)  # shape (batch_size, 1, T), where batch_size = 1
        pred_logits = pred_logits[..., pad_left:(pred_logits.shape[-1] - pad_right)]
        pred_logits_small = pred_logits_small[..., pad_left:(pred_logits_small.shape[-1] - pad_right)]
        pred_logits_mid = pred_logits_mid[..., pad_left:(pred_logits_mid.shape[-1] - pad_right)]

        loss = (ce_loss(pred_logits, labels_batch) + ce_loss(pred_logits_small, labels_batch) + ce_loss(pred_logits_mid, labels_batch)) / 3.0
        masked_loss = ce_loss(pred_logits, labels_batch, mask)
        more_masked_loss = ce_loss(pred_logits, labels_batch, more_mask)

        preds = pred_logits > 0.0
        preds_small = pred_logits_small > 0.0
        preds_mid = pred_logits_mid > 0.0

        return loss.item(), preds.to(torch.long), preds_mid.to(torch.long),\
            preds_small.to(torch.long), masked_loss.item(), more_masked_loss.item()

def validation_step():
    for key in val_metrics:
        val_metrics[key].reset()

    # validation
    with (tqdm.tqdm(total=len(validation_entries)) as pbar):
        for k in range(len(validation_entries)):
            batch_entry = validation_entries[k]  # series ids

            # load the batch
            accel_data_batch = all_data[batch_entry]["accel"]
            labels_batch = all_data[batch_entry]["sleeping_timesteps"]
            accel_data_batch = torch.tensor(accel_data_batch, dtype=torch.float32, device=config.device).unsqueeze(0)
            labels_batch = torch.tensor(labels_batch, dtype=torch.float32, device=config.device).unsqueeze(0).unsqueeze(0)
            mask = val_sampler.get_series_mask(batch_entry, all_data)
            mask = torch.tensor(mask, dtype=torch.float32, device=config.device).unsqueeze(0).unsqueeze(0)
            more_mask = val_sampler_more.get_series_mask(batch_entry, all_data)
            more_mask = torch.tensor(more_mask, dtype=torch.float32, device=config.device).unsqueeze(0).unsqueeze(0)

            # pad such that lengths is a multiple of 48
            pad_length = 48 - (accel_data_batch.shape[-1] % 48)
            if pad_length == 48:
                pad_length = 0
            pad_left = pad_length // 2
            pad_right = pad_length - pad_left

            # val model now
            if use_deep_supervision:
                loss, preds, preds_mid, preds_small, masked_loss, more_masked_loss =\
                    single_validation_step_deep_supervision(model, accel_data_batch, labels_batch, mask, more_mask, pad_left, pad_right)
            else:
                loss, preds, masked_loss, more_masked_loss = single_validation_step(model, accel_data_batch, labels_batch,
                                                                                    mask, more_mask,
                                                                                    pad_left, pad_right)

            # record
            with torch.no_grad():
                val_metrics["loss"].add(loss, 1)
                val_metrics["metric"].add(preds, labels_batch.to(torch.long))
                if use_deep_supervision:
                    val_metrics["metric_mid"].add(preds_mid, labels_batch.to(torch.long))
                    val_metrics["metric_small"].add(preds_small, labels_batch.to(torch.long))
                val_metrics["loss_masked"].add(masked_loss, 1)
                val_metrics["metric_masked"].add(preds, labels_batch.to(torch.long), mask.to(torch.long))
                val_metrics["loss_more_masked"].add(more_masked_loss, 1)
                val_metrics["metric_more_masked"].add(preds, labels_batch.to(torch.long), more_mask.to(torch.long))

            pbar.update(1)

    current_metrics = {}
    for key in val_metrics:
        val_metrics[key].write_to_dict(current_metrics)

    for key in current_metrics:
        val_history[key].append(current_metrics[key])

def print_history(metrics_history):
    for key in metrics_history:
        print("{}      {}".format(key, metrics_history[key][-1]))

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    all_data = convert_to_h5py_naive.load_all_data_into_dict()
    all_good_events = convert_to_good_events.load_all_data_into_dict()
    all_good_events_no_exclusion = convert_to_good_events.load_all_data_into_dict(all_events=True)

    parser = argparse.ArgumentParser(description="Train a sleeping prediction model with only clean data.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train for. Default 30.")
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="Learning rate to use. Default 5e-3.")
    parser.add_argument("--use_decay_schedule", action="store_true", help="Whether to use a decay schedule. Default False.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum to use. Default 0.9. This would be the momentum for SGD, and beta1 for Adam.")
    parser.add_argument("--second_momentum", type=float, default=0.999, help="Second momentum to use. Default 0.999. This would be beta2 for Adam. Ignored if SGD.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use. Default 0.0.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Which optimizer to use. Available options: adam, sgd. Default adam.")
    parser.add_argument("--epochs_per_save", type=int, default=2, help="Number of epochs between saves. Default 2.")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[1, 6, 8, 23, 8],
                        help="Number of hidden 2d blocks for ResNet backbone.")
    parser.add_argument("--hidden_channels", type=int, default=32, help="Number of hidden channels. Default None.")
    parser.add_argument("--bottleneck_factor", type=int, default=4, help="The bottleneck factor of the ResNet backbone. Default 4.")
    parser.add_argument("--squeeze_excitation", action="store_false", help="Whether to use squeeze and excitation. Default True.")
    parser.add_argument("--kernel_size", type=int, default=11, help="Kernel size to use. Default 11.")
    parser.add_argument("--disable_odd_random_shift", action="store_true", help="Whether to disable odd random shift. Default False.")
    parser.add_argument("--use_batch_norm", action="store_true", help="Whether to use batch norm. Default False.")
    parser.add_argument("--use_mixup_training", action="store_true", help="Whether to use mixup training. Default False.")
    parser.add_argument("--use_deep_supervision", action="store_true", help="Whether to use deep supervision. Default False.")
    parser.add_argument("--record_best_model", action="store_true", help="Whether to record the best model. Default False.")
    parser.add_argument("--do_not_exclude", action="store_true", help="Whether to not exclude any events where the watch isn't being worn. Default False.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate. Default 0.0.")
    parser.add_argument("--length", default=9600000, help="The fixed length of the training. Default 9600000.")
    parser.add_argument("--num_extra_steps", type=int, default=0, help="Extra steps of gradient descent before the usual step in an epoch. Default 0.")
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

    # initialize gpu
    config.parse_args(args)

    # get model directories
    model_dir, prev_model_dir = manager_models.parse_args(args)

    # obtain model and training parameters
    epochs = args.epochs
    learning_rate = args.learning_rate
    use_decay_schedule = args.use_decay_schedule
    momentum = args.momentum
    second_momentum = args.second_momentum
    weight_decay = args.weight_decay
    optimizer_type = args.optimizer
    epochs_per_save = args.epochs_per_save
    hidden_blocks = args.hidden_blocks
    hidden_channels = args.hidden_channels
    bottleneck_factor = args.bottleneck_factor
    squeeze_excitation = args.squeeze_excitation
    kernel_size = args.kernel_size
    disable_odd_random_shift = args.disable_odd_random_shift
    use_batch_norm = args.use_batch_norm
    use_mixup_training = args.use_mixup_training
    use_deep_supervision = args.use_deep_supervision
    record_best_model = args.record_best_model
    do_not_exclude = args.do_not_exclude
    dropout = args.dropout
    length = args.length
    num_extra_steps = args.num_extra_steps

    print("Epochs: " + str(epochs))
    print("Dropout: " + str(dropout))
    print("Batch norm: " + str(use_batch_norm))
    print("Squeeze excitation: " + str(squeeze_excitation))
    print("Kernel size: " + str(kernel_size))
    model_unet.BATCH_NORM_MOMENTUM = 1 - momentum

    # initialize model
    if use_deep_supervision:
        model = model_attention_unet.Unet3fDeepSupervision(2, hidden_channels, kernel_size=kernel_size, blocks=hidden_blocks,
                                bottleneck_factor=bottleneck_factor, squeeze_excitation=squeeze_excitation,
                                squeeze_excitation_bottleneck_factor=4,
                                dropout=dropout, use_batch_norm=use_batch_norm, out_channels=1, attn_out_channels=2)
    else:
        model = model_unet.Unet(2, hidden_channels, kernel_size=kernel_size, blocks=hidden_blocks,
                                bottleneck_factor=bottleneck_factor, squeeze_excitation=squeeze_excitation,
                                squeeze_excitation_bottleneck_factor=4, odd_random_shift_training=(not disable_odd_random_shift),
                                dropout=dropout, use_batch_norm=use_batch_norm, initial_3f_downsampling=True)
    model = model.to(config.device)

    # initialize optimizer
    print("Learning rate: " + str(learning_rate))
    print("Momentum: " + str(momentum))
    print("Second momentum: " + str(second_momentum))
    print("Weight decay: " + str(weight_decay))
    print("Optimizer: " + optimizer_type)
    if optimizer_type.lower() == "adam":
        if weight_decay > 0.0:
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(momentum, second_momentum), weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(momentum, second_momentum))
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
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
        "model": "Naive deep learning segmentation approach",
        "training_dataset": train_dset_name,
        "validation_dataset": val_dset_name,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "use_decay_schedule": use_decay_schedule,
        "momentum": momentum,
        "second_momentum": second_momentum,
        "weight_decay": weight_decay,
        "optimizer": optimizer_type,
        "epochs_per_save": epochs_per_save,
        "hidden_blocks": hidden_blocks,
        "hidden_channels": hidden_channels,
        "bottleneck_factor": bottleneck_factor,
        "squeeze_excitation": squeeze_excitation,
        "kernel_size": kernel_size,
        "disable_odd_random_shift": disable_odd_random_shift,
        "use_batch_norm": use_batch_norm,
        "use_mixup_training": use_mixup_training,
        "use_deep_supervision": use_deep_supervision,
        "record_best_model": record_best_model,
        "do_not_exclude": do_not_exclude,
        "dropout": dropout,
        "length": length,
        "num_extra_steps": num_extra_steps,
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
    if use_deep_supervision:
        val_metrics["metric_mid"] = metrics.BinaryMetrics("val_metric_deep_mid")
        val_metrics["metric_small"] = metrics.BinaryMetrics("val_metric_deep_small")
    val_metrics["loss_masked"] = metrics.NumericalMetric("val_loss_masked")
    val_metrics["metric_masked"] = metrics.BinaryMetrics("val_metric_masked")
    val_metrics["loss_more_masked"] = metrics.NumericalMetric("val_loss_more_masked")
    val_metrics["metric_more_masked"] = metrics.BinaryMetrics("val_metric_more_masked")

    # Compile
    #single_training_step_compile = torch.compile(single_training_step)

    # Initialize the sampler
    print("Using mixup training: {}".format(use_mixup_training))
    print("Length: {}".format(length))
    training_good_events = all_good_events_no_exclusion if do_not_exclude else all_good_events
    training_sampler = convert_to_good_events.GoodEvents(training_good_events,
                                                         training_entries)
    if use_mixup_training:
        training_sampler2 = convert_to_good_events.GoodEvents(training_good_events,
                                                            training_entries)
    val_sampler = convert_to_good_events.GoodEvents(all_good_events,
                                                    validation_entries,
                                                    is_train=False,
                                                    load_head_entries=True,
                                                    load_tail_entries=True)
    val_sampler_more = convert_to_good_events.GoodEvents(all_good_events,
                                                    validation_entries,
                                                    is_train=False,
                                                    load_head_entries=False,
                                                    load_tail_entries=False)


    # Start training loop
    print("Training for {} epochs......".format(epochs))
    memory_logger = logging_memory_utils.obtain_memory_logger(model_dir)

    best_epoch = -1
    best_val_masked_precision = -1
    best_val_masked_recall = -1

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
            if np.isnan(train_history["train_loss"][-1]):
                print("Is nan. Reverting...")
                model.load_state_dict(torch.load(os.path.join(model_dir, "latest_model.pt")))
                optimizer.load_state_dict(torch.load(os.path.join(model_dir, "latest_optimizer.pt")))
            else:
                torch.save(model.state_dict(), os.path.join(model_dir, "latest_model.pt"))
                torch.save(optimizer.state_dict(), os.path.join(model_dir, "latest_optimizer.pt"))

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

            if record_best_model:
                if (val_history["val_metric_more_masked_precision"][-1] > best_val_masked_precision) and\
                        (val_history["val_metric_more_masked_recall"][-1] > 0.96) and \
                        (val_history["val_metric_precision"][-1] > 0.74):
                    if (val_history["val_metric_more_masked_recall"][-1] > best_val_masked_recall) or \
                            (best_val_masked_recall > 0.97 and val_history["val_metric_more_masked_recall"][-1] > 0.97):
                        best_val_masked_precision = val_history["val_metric_more_masked_precision"][-1]
                        best_val_masked_recall = val_history["val_metric_more_masked_recall"][-1]
                        best_epoch = epoch
                        torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
                        torch.save(optimizer.state_dict(), os.path.join(model_dir, "best_optimizer.pt"))
                        # update best information text file
                        with open(os.path.join(model_dir, "best_info.txt"), "w") as f:
                            f.write("Best epoch: {}\n".format(best_epoch))
                            f.write("Best val masked precision: {}\n".format(best_val_masked_precision))
                            f.write("Best val masked recall: {}\n".format(best_val_masked_recall))
                            f.write("Best val masked accuracy: {}\n".format(val_history["val_metric_more_masked_accuracy"][-1]))
                            f.write("Best val precision: {}\n".format(val_history["val_metric_precision"][-1]))

            gc.collect()

            # adjust learning rate
            if warmup_steps > 0:
                warmup_steps -= 1
                if warmup_steps == 0:
                    for g in optimizer.param_groups:
                        g["lr"] = learning_rate
            else:
                if use_decay_schedule:
                    if epoch > 25:
                        new_rate = learning_rate * 0.98 * np.power(0.9, (epoch - 25)) + learning_rate * 0.02
                        print("Setting learning rate to {:.5f}".format(new_rate))
                        for g in optimizer.param_groups:
                            g["lr"] = new_rate

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