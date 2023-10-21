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
import metrics_iou
import logging_memory_utils
import convert_to_h5py_naive
import convert_to_interval_events
import convert_to_good_events
import model_unet
import model_attention_unet

# same as training_clean_data.py, but with 3 factor downsampling at the first layer

def ce_loss(preds: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor = None):
    assert preds.shape == ground_truth.shape, "preds.shape = {}, ground_truth.shape = {}".format(preds.shape,
                                                                                                 ground_truth.shape)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(preds, ground_truth, reduction="none")
    if mask is None:
        return torch.sum(torch.mean(bce, dim=(1, 2)))
    else:
        return torch.sum(torch.mean(bce * mask, dim=(1, 2)))

def focal_loss(preds: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor = None):
    assert preds.shape == ground_truth.shape, "preds.shape = {}, ground_truth.shape = {}".format(preds.shape, ground_truth.shape)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(preds, ground_truth, reduction="none")
    if mask is None:
        return torch.sum(torch.mean(((torch.sigmoid(preds) - ground_truth) ** 2) * bce, dim=(1, 2)))
    else:
        return torch.sum(torch.mean(((torch.sigmoid(preds) - ground_truth) ** 2) * bce * mask, dim=(1, 2)))

def dice_loss(preds: torch.Tensor, ground_truth: torch.Tensor, eps=1e-5):
    assert preds.shape == ground_truth.shape, "preds.shape = {}, ground_truth.shape = {}".format(preds.shape,
                                                                                                 ground_truth.shape)
    pred_probas = torch.sigmoid(preds)
    intersection = torch.sum(pred_probas * ground_truth, dim=(1, 2))
    union = torch.sum(pred_probas + ground_truth, dim=(1, 2))
    return torch.sum(1 - (2 * intersection + eps) / (union + eps))



def single_training_step(model_: torch.nn.Module, optimizer_: torch.optim.Optimizer,
                            accel_data_batch: torch.Tensor, labels_batch: torch.Tensor):
    optimizer_.zero_grad()
    pred_logits = model_(accel_data_batch, deep_supervision=False)  # shape (batch_size, 2, T // 12)
    if use_ce_loss:
        loss = ce_loss(pred_logits, labels_batch)
    elif use_iou_loss:
        loss = dice_loss(pred_logits, labels_batch) + 0.001 * focal_loss(pred_logits, labels_batch)
    else:
        loss = focal_loss(pred_logits, labels_batch)
    loss.backward()
    optimizer_.step()

    with torch.no_grad():
        preds = pred_logits > 0.0

    return loss.item(), preds.to(torch.long)

def single_training_step_deep(model_: torch.nn.Module, optimizer_: torch.optim.Optimizer,
                            accel_data_batch: torch.Tensor, labels_batch: torch.Tensor):
    optimizer_.zero_grad()
    pred_logits_small, pred_logits_mid, pred_logits = model_(accel_data_batch, True) # shape (batch_size, 1, T), where batch_size = 1
    loss = batch_size * (ce_loss(pred_logits, labels_batch) + ce_loss(pred_logits_small, labels_batch) + ce_loss(pred_logits_mid, labels_batch)) / 3.0
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

    # training
    cur_deep_supervision = False
    with tqdm.tqdm(total=len(training_sampler)) as pbar:
        while training_sampler.entries_remaining() > 0:
            if use_deep_supervision is not None and \
                    (deep_supervision_limit is None or epoch < deep_supervision_limit):
                cur_deep_supervision = not cur_deep_supervision

            if cur_deep_supervision:
                accel_data_batch, labels_batch = training_sampler_deep.get_next_streaming(target_length=use_deep_supervision,
                                                                                                     all_series_data=all_data)

                accel_data_batch_torch = torch.tensor(accel_data_batch, dtype=torch.float32, device=config.device)
                labels_batch_torch = torch.tensor(labels_batch, dtype=torch.float32, device=config.device)
                if use_anglez_only:
                    accel_data_batch_torch = accel_data_batch_torch[:, 0:1, :]
                if random_flip:
                    if np.random.rand() > 0.5:
                        accel_data_batch_torch = torch.flip(accel_data_batch_torch, dims=(2,))
                        labels_batch_torch = torch.flip(labels_batch_torch, dims=(2,))

                # train model now
                loss, preds = single_training_step_deep(model, optimizer,
                                                   accel_data_batch_torch,
                                                   labels_batch_torch)

                # record
                if record:
                    with torch.no_grad():
                        train_metrics["deep_loss"].add(loss, 1)
                        train_metrics["deep_metric"].add(preds, (labels_batch_torch > 0.5).to(torch.long))

            else:
                accel_data_batch, labels_batch, increment = training_sampler.sample(batch_size, random_shift=random_shift,
                                                                                    random_flip=random_flip, always_flip=always_flip,
                                                                                    expand=expand)

                accel_data_batch_torch = torch.tensor(accel_data_batch, dtype=torch.float32, device=config.device)
                labels_batch_torch = torch.tensor(labels_batch, dtype=torch.float32, device=config.device)
                if use_anglez_only:
                    accel_data_batch_torch = accel_data_batch_torch[:, 0:1, :]

                # train model now
                loss, preds = single_training_step(model, optimizer,
                                                   accel_data_batch_torch,
                                                   labels_batch_torch)
                #time.sleep(0.2)

                # record
                if record:
                    with torch.no_grad():
                        labels_long = (labels_batch_torch > 0.5).to(torch.long)
                        train_metrics["loss"].add(loss, increment)
                        train_metrics["metric"].add(preds, labels_long)
                        true_positives, false_positives, true_negatives, false_negatives =\
                            metrics_iou.compute_iou_metrics(labels_long[:, 0:1, :], preds[:, 0:1, :])
                        train_metrics["onset_iou_metric"].add_direct(true_positives, true_negatives, false_positives, false_negatives)
                        true_positives, false_positives, true_negatives, false_negatives = \
                            metrics_iou.compute_iou_metrics(labels_long[:, 1:2, :], preds[:, 1:2, :])
                        train_metrics["wakeup_iou_metric"].add_direct(true_positives, true_negatives, false_positives,
                                                                     false_negatives)

                pbar.update(increment)

    if record:
        current_metrics = {}
        for key in train_metrics:
            train_metrics[key].write_to_dict(current_metrics)

        for key in current_metrics:
            train_history[key].append(current_metrics[key])

def single_validation_step(model_: torch.nn.Module, accel_data_batch: torch.Tensor,
                           labels_batch: torch.Tensor):
    with torch.no_grad():
        pred_logits = model_(accel_data_batch)  # shape (batch_size, 2, T)
        if use_ce_loss:
            loss = ce_loss(pred_logits, labels_batch)
        elif use_iou_loss:
            loss = dice_loss(pred_logits, labels_batch) + 0.001 * focal_loss(pred_logits, labels_batch)
        else:
            loss = focal_loss(pred_logits, labels_batch)
        preds = pred_logits > 0.0

        return loss.item(), preds.to(torch.long)

def single_validation_step_deep(model_: torch.nn.Module, accel_data_batch: torch.Tensor,
                           labels_batch: torch.Tensor, pad_left, pad_right):
    with torch.no_grad():
        accel_data_batch = torch.nn.functional.pad(accel_data_batch, (pad_left, pad_right), mode="replicate")

        pred_logits_small, pred_logits_mid, pred_logits = model_(accel_data_batch, True)  # shape (batch_size, 1, T), where batch_size = 1
        pred_logits = pred_logits[..., pad_left:(pred_logits.shape[-1] - pad_right)]
        pred_logits_small = pred_logits_small[..., pad_left:(pred_logits_small.shape[-1] - pad_right)]
        pred_logits_mid = pred_logits_mid[..., pad_left:(pred_logits_mid.shape[-1] - pad_right)]

        loss = (ce_loss(pred_logits, labels_batch) + ce_loss(pred_logits_small, labels_batch) + ce_loss(pred_logits_mid, labels_batch)) / 3.0

        preds = pred_logits > 0.0

        return loss.item(), preds.to(torch.long)

def validation_step():
    for key in val_metrics:
        val_metrics[key].reset()

    # validation
    val_sampler.shuffle()
    with (tqdm.tqdm(total=len(val_sampler)) as pbar):
        while val_sampler.entries_remaining() > 0:
            # load the batch
            accel_data_batch, labels_batch, increment = val_sampler.sample(batch_size, expand=expand)
            accel_data_batch = torch.tensor(accel_data_batch, dtype=torch.float32, device=config.device)
            labels_batch = torch.tensor(labels_batch, dtype=torch.float32, device=config.device)
            if use_anglez_only:
                accel_data_batch = accel_data_batch[:, 0:1, :]

            # val model now
            loss, preds = single_validation_step(model, accel_data_batch, labels_batch)

            # record
            with torch.no_grad():
                labels_long = labels_batch.to(torch.long)
                val_metrics["loss"].add(loss, increment)
                val_metrics["metric"].add(preds, labels_long)
                val_metrics["onset_metric"].add(preds[:, 0, :], labels_long[:, 0, :])
                val_metrics["wakeup_metric"].add(preds[:, 1, :], labels_long[:, 1, :])

                true_positives, false_positives, true_negatives, false_negatives = \
                    metrics_iou.compute_iou_metrics(labels_long[:, 0:1, :], preds[:, 0:1, :], iou_threshold=0.1)
                val_metrics["onset_iou_metric1"].add_direct(true_positives, true_negatives, false_positives,
                                                            false_negatives)
                true_positives, false_positives, true_negatives, false_negatives = \
                    metrics_iou.compute_iou_metrics(labels_long[:, 1:2, :], preds[:, 1:2, :], iou_threshold=0.1)
                val_metrics["wakeup_iou_metric1"].add_direct(true_positives, true_negatives, false_positives,
                                                             false_negatives)

                true_positives, false_positives, true_negatives, false_negatives = \
                    metrics_iou.compute_iou_metrics(labels_long[:, 0:1, :], preds[:, 0:1, :], iou_threshold=0.3)
                val_metrics["onset_iou_metric3"].add_direct(true_positives, true_negatives, false_positives,
                                                            false_negatives)
                true_positives, false_positives, true_negatives, false_negatives = \
                    metrics_iou.compute_iou_metrics(labels_long[:, 1:2, :], preds[:, 1:2, :], iou_threshold=0.3)
                val_metrics["wakeup_iou_metric3"].add_direct(true_positives, true_negatives, false_positives,
                                                             false_negatives)

                true_positives, false_positives, true_negatives, false_negatives = \
                    metrics_iou.compute_iou_metrics(labels_long[:, 0:1, :], preds[:, 0:1, :])
                val_metrics["onset_iou_metric5"].add_direct(true_positives, true_negatives, false_positives,
                                                             false_negatives)
                true_positives, false_positives, true_negatives, false_negatives = \
                    metrics_iou.compute_iou_metrics(labels_long[:, 1:2, :], preds[:, 1:2, :])
                val_metrics["wakeup_iou_metric5"].add_direct(true_positives, true_negatives, false_positives,
                                                              false_negatives)

            pbar.update(increment)

    # validation on deep supervision
    if use_deep_supervision is not None:
        with (tqdm.tqdm(total=len(validation_entries)) as pbar):
            for k in range(len(validation_entries)):
                batch_entry = validation_entries[k]  # series ids

                # load the batch
                accel_data_batch = all_data[batch_entry]["accel"]
                labels_batch = all_data[batch_entry]["sleeping_timesteps"]
                accel_data_batch = torch.tensor(accel_data_batch, dtype=torch.float32, device=config.device)\
                    .unsqueeze(0)
                labels_batch = torch.tensor(labels_batch, dtype=torch.float32, device=config.device)\
                    .unsqueeze(0).unsqueeze(0)
                if use_anglez_only:
                    accel_data_batch = accel_data_batch[:, 0:1, :]

                # pad such that lengths is a multiple of 48
                pad_length = 48 - (accel_data_batch.shape[-1] % 48)
                if pad_length == 48:
                    pad_length = 0
                pad_left = pad_length // 2
                pad_right = pad_length - pad_left

                loss, preds = single_validation_step_deep(model, accel_data_batch, labels_batch, pad_left, pad_right)

                # record
                with torch.no_grad():
                    val_metrics["deep_loss"].add(loss, 1)
                    val_metrics["deep_metric"].add(preds, labels_batch.to(torch.long))

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
    all_interval_segmentations = convert_to_interval_events.load_all_segmentations()
    all_good_events_no_exclusion = convert_to_good_events.load_all_data_into_dict(all_events=True)

    parser = argparse.ArgumentParser(description="Train a sleeping prediction model with only clean data.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for. Default 50.")
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="Learning rate to use. Default 5e-3.")
    parser.add_argument("--use_decay_schedule", action="store_true", help="Whether to use a decay schedule. Default False.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum to use. Default 0.9. This would be the momentum for SGD, and beta1 for Adam.")
    parser.add_argument("--second_momentum", type=float, default=0.999, help="Second momentum to use. Default 0.999. This would be beta2 for Adam. Ignored if SGD.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use. Default 0.0.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Which optimizer to use. Available options: adam, sgd. Default adam.")
    parser.add_argument("--epochs_per_save", type=int, default=2, help="Number of epochs between saves. Default 2.")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[1, 6, 8, 23, 8],
                        help="Number of hidden 2d blocks for ResNet backbone.")
    parser.add_argument("--hidden_channels", type=int, nargs="+", default=[2], help="Number of hidden channels. Default 2. Can be a list to specify num channels in each downsampled layer.")
    parser.add_argument("--bottleneck_factor", type=int, default=4, help="The bottleneck factor of the ResNet backbone. Default 4.")
    parser.add_argument("--squeeze_excitation", action="store_false", help="Whether to use squeeze and excitation. Default True.")
    parser.add_argument("--kernel_size", type=int, default=11, help="Kernel size for the first layer. Default 11.")
    parser.add_argument("--attention_bottleneck", type=int, default=None, help="The bottleneck factor of the attention module. Default None.")
    parser.add_argument("--random_shift", type=int, default=0, help="Randomly shift the intervals by at most this amount. Default 0.")
    parser.add_argument("--random_flip", action="store_true", help="Randomly flip the intervals. Default False.")
    parser.add_argument("--always_flip", action="store_true", help="Always flip the intervals. Default False.")
    parser.add_argument("--expand", type=int, default=0, help="Expand the intervals by this amount. Default 0.")
    parser.add_argument("--use_batch_norm", action="store_true", help="Whether to use batch norm. Default False.")
    parser.add_argument("--do_not_exclude", action="store_true", help="Whether to not exclude any events where the watch isn't being worn. Default False.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate. Default 0.0.")
    parser.add_argument("--dropout_pos_embeddings", action="store_true", help="Whether to dropout the positional embeddings. Default False.")
    parser.add_argument("--use_ce_loss", action="store_true", help="Whether to use cross entropy loss. Default False.")
    parser.add_argument("--use_iou_loss", action="store_true", help="Whether to use IOU loss. Default False.")
    parser.add_argument("--use_deep_supervision", type=int, default=None, help="Whether to use deep supervision. Default None. If specified, must be an integer indicating the length of deep supervision training.")
    parser.add_argument("--deep_supervision_limit", type=int, default=None, help="The maximum number of deep supervision training steps. Default None. If specified, must be an integer. Ignored if not using deep supervision.")
    parser.add_argument("--use_cutmix", action="store_true", help="Whether to use cutmix. Default False.")
    parser.add_argument("--cutmix_skip", type=int, default=540, help="Maximum of random skip intervals for cutmix. Default 540. Ignored if not using cutmix.")
    parser.add_argument("--cutmix_length", type=int, default=540, help="Length of intervals for cutmix. Default 540. Ignored if not using cutmix.")
    parser.add_argument("--use_anglez_only", action="store_true", help="Whether to use only anglez. Default False.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size. Default 512.")
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
    attention_bottleneck = args.attention_bottleneck
    random_shift = args.random_shift
    random_flip = args.random_flip
    always_flip = args.always_flip
    expand = args.expand
    use_batch_norm = args.use_batch_norm
    do_not_exclude = args.do_not_exclude
    dropout = args.dropout
    dropout_pos_embeddings = args.dropout_pos_embeddings
    use_ce_loss = args.use_ce_loss
    use_iou_loss = args.use_iou_loss
    use_deep_supervision = args.use_deep_supervision
    deep_supervision_limit = args.deep_supervision_limit
    use_cutmix = args.use_cutmix
    cutmix_skip = args.cutmix_skip
    cutmix_length = args.cutmix_length
    use_anglez_only = args.use_anglez_only
    batch_size = args.batch_size
    num_extra_steps = args.num_extra_steps

    assert not (use_iou_loss and use_ce_loss), "Cannot use both IOU loss and cross entropy loss."

    if isinstance(hidden_channels, int):
        hidden_channels = [hidden_channels]
    if len(hidden_channels) == 1:
        chnls = hidden_channels[0]
        hidden_channels = []
        for k in range(len(hidden_blocks)):
            hidden_channels.append(chnls * (2 ** k))

    print("Epochs: " + str(epochs))
    print("Dropout: " + str(dropout))
    print("Dropout pos embeddings: " + str(dropout_pos_embeddings))
    print("Batch norm: " + str(use_batch_norm))
    print("Squeeze excitation: " + str(squeeze_excitation))
    print("Bottleneck factor: " + str(bottleneck_factor))
    print("Hidden channels: " + str(hidden_channels))
    print("Kernel size: " + str(kernel_size))
    print("Attention bottleneck: " + str(attention_bottleneck))
    model_unet.BATCH_NORM_MOMENTUM = 1 - momentum

    # initialize model
    in_channels = 1 if use_anglez_only else 2
    model = model_attention_unet.Unet3fDeepSupervision(in_channels, hidden_channels, kernel_size=kernel_size, blocks=hidden_blocks,
                            bottleneck_factor=bottleneck_factor, squeeze_excitation=squeeze_excitation,
                            squeeze_excitation_bottleneck_factor=4,
                            dropout=dropout, dropout_pos_embeddings=dropout_pos_embeddings,
                            use_batch_norm=use_batch_norm, out_channels=1, attn_out_channels=2, attention_bottleneck=attention_bottleneck,
                            expected_attn_input_length=17280 + (2 * expand))
    model = model.to(config.device)

    # initialize optimizer
    loss_print = "Cross entropy" if use_ce_loss else ("IOU" if use_iou_loss else "Focal")
    print("Loss: " + loss_print)
    print("Learning rate: " + str(learning_rate))
    print("Momentum: " + str(momentum))
    print("Second momentum: " + str(second_momentum))
    print("Weight decay: " + str(weight_decay))
    print("Optimizer: " + optimizer_type)
    print("Use deep supervision: " + str(use_deep_supervision))
    print("Deep supervision limit: " + str(deep_supervision_limit))
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
        warmup_steps = 1
        for g in optimizer.param_groups:
            g["lr"] = 0.0
    else:
        warmup_steps = 1
        model_checkpoint_path = os.path.join(prev_model_dir, "model.pt")
        optimizer_checkpoint_path = os.path.join(prev_model_dir, "optimizer.pt")

        model.load_state_dict(torch.load(model_checkpoint_path))
        optimizer.load_state_dict(torch.load(optimizer_checkpoint_path))

        for g in optimizer.param_groups:
            g["lr"] = 0.0
            if optimizer_type == "adam":
                g["betas"] = (momentum, second_momentum)
            elif optimizer_type == "sgd":
                g["momentum"] = momentum

    model_config = {
        "model": "Unet with attention segmentation",
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
        "attention_bottleneck": attention_bottleneck,
        "random_shift": random_shift,
        "random_flip": random_flip,
        "always_flip": always_flip,
        "expand": expand,
        "use_batch_norm": use_batch_norm,
        "do_not_exclude": do_not_exclude,
        "dropout": dropout,
        "dropout_pos_embeddings": dropout_pos_embeddings,
        "use_ce_loss": use_ce_loss,
        "use_deep_supervision": use_deep_supervision,
        "deep_supervision_limit": deep_supervision_limit,
        "use_cutmix": use_cutmix,
        "cutmix_skip": cutmix_skip,
        "cutmix_length": cutmix_length,
        "use_anglez_only": use_anglez_only,
        "batch_size": batch_size,
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
    train_metrics["onset_iou_metric"] = metrics.BinaryMetrics("train_onset_iou_metric")
    train_metrics["wakeup_iou_metric"] = metrics.BinaryMetrics("train_wakeup_iou_metric")
    if use_deep_supervision is not None:
        train_metrics["deep_loss"] = metrics.NumericalMetric("train_deep_loss")
        train_metrics["deep_metric"] = metrics.BinaryMetrics("train_deep_metric")
    val_metrics["loss"] = metrics.NumericalMetric("val_loss")
    val_metrics["metric"] = metrics.BinaryMetrics("val_metric")
    val_metrics["onset_metric"] = metrics.BinaryMetrics("val_onset_metric")
    val_metrics["wakeup_metric"] = metrics.BinaryMetrics("val_wakeup_metric")
    val_metrics["onset_iou_metric1"] = metrics.BinaryMetrics("val_onset_iou_metric1")
    val_metrics["wakeup_iou_metric1"] = metrics.BinaryMetrics("val_wakeup_iou_metric1")
    val_metrics["onset_iou_metric3"] = metrics.BinaryMetrics("val_onset_iou_metric3")
    val_metrics["wakeup_iou_metric3"] = metrics.BinaryMetrics("val_wakeup_iou_metric3")
    val_metrics["onset_iou_metric5"] = metrics.BinaryMetrics("val_onset_iou_metric5")
    val_metrics["wakeup_iou_metric5"] = metrics.BinaryMetrics("val_wakeup_iou_metric5")
    if use_deep_supervision is not None:
        val_metrics["deep_loss"] = metrics.NumericalMetric("val_deep_loss")
        val_metrics["deep_metric"] = metrics.BinaryMetrics("val_deep_metric")

    # Compile
    #single_training_step_compile = torch.compile(single_training_step)

    # Initialize the sampler
    print("Initializing the samplers...")
    print("Batch size: " + str(batch_size))
    print("Random shift: " + str(random_shift))
    print("Random flip: " + str(random_flip))
    print("Always flip: " + str(always_flip))
    print("Expand: " + str(expand))
    print("Use cutmix: " + str(use_cutmix))
    print("Use anglez only: " + str(use_anglez_only))
    if use_cutmix:
        print("Cutmix skip: " + str(cutmix_skip))
        print("Cutmix length: " + str(cutmix_length))
        training_sampler = convert_to_interval_events.SemiSyntheticIntervalEventsSampler(training_entries, all_data, all_interval_segmentations,
                                                                                         spliced_good_events=convert_to_good_events.GoodEventsSplicedSampler(
                                                                                             data_dict=convert_to_good_events.load_all_data_into_dict(),
                                                                                             entries_sublist=training_entries,
                                                                                             expected_length=cutmix_length
                                                                                         ),
                                                                                         cutmix_skip=cutmix_skip)
    else:
        training_sampler = convert_to_interval_events.IntervalEventsSampler(training_entries, all_data, all_interval_segmentations,
                                                                            train_or_test="train")
    val_sampler = convert_to_interval_events.IntervalEventsSampler(validation_entries, all_data, all_interval_segmentations,
                                                                        train_or_test="val")
    if use_deep_supervision is not None:
        training_sampler_deep = convert_to_good_events.GoodEvents(all_good_events_no_exclusion, training_entries, is_train=True, train_streaming=True)


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