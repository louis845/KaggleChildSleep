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

def masked_huber_loss(preds: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor):
    assert preds.shape == ground_truth.shape, "preds.shape = {}, ground_truth.shape = {}".format(preds.shape,
                                                                                                 ground_truth.shape)
    assert preds.shape == mask.shape, "preds.shape = {}, mask.shape = {}".format(preds.shape, mask.shape)
    mask_per_batch = torch.sum(mask, dim=(1, 2))
    loss_per_batch = torch.where(mask_per_batch > 0, torch.sum(torch.nn.functional.huber_loss(preds, ground_truth, reduction="none") * mask, dim=(1, 2)) / (mask_per_batch + 1e-7),
                                    torch.zeros_like(mask_per_batch))
    return torch.sum(loss_per_batch)

def masked_mae_loss(preds: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor):
    assert preds.shape == ground_truth.shape, "preds.shape = {}, ground_truth.shape = {}".format(preds.shape,
                                                                                                 ground_truth.shape)
    assert preds.shape == mask.shape, "preds.shape = {}, mask.shape = {}".format(preds.shape, mask.shape)
    mask_per_batch = torch.sum(mask, dim=(1, 2))
    loss_per_batch = torch.where(mask_per_batch > 0, torch.sum(torch.abs(preds - ground_truth) * mask, dim=(1, 2)) / mask_per_batch,
                                    torch.zeros_like(mask_per_batch))
    return torch.sum(loss_per_batch)


def single_training_step(model_: torch.nn.Module, optimizer_: torch.optim.Optimizer,
                            accel_data_batch: torch.Tensor, labels_batch: torch.Tensor,
                            event_regression_values: list[torch.Tensor], event_regression_masks: list[torch.Tensor]):
    optimizer_.zero_grad()
    pred_logits, pred_small, pred_mid, pred_large = model_(accel_data_batch, ret_type=("all" if use_deep_supervision is not None else "attn"))
    if use_ce_loss:
        loss = ce_loss(pred_logits, labels_batch)
    elif use_iou_loss:
        loss = 0.01 * dice_loss(pred_logits, labels_batch) + focal_loss(pred_logits, labels_batch)
    else:
        loss = focal_loss(pred_logits, labels_batch)

    if use_deep_supervision is not None:
        small_optim_loss = masked_huber_loss(pred_small, event_regression_values[0], event_regression_masks[0])
        mid_optim_loss = masked_huber_loss(pred_mid, event_regression_values[1], event_regression_masks[1])
        large_optim_loss = masked_huber_loss(pred_large, event_regression_values[2], event_regression_masks[2])
        floss = loss + small_optim_loss + mid_optim_loss + large_optim_loss
        floss.backward()
    else:
        loss.backward()
    optimizer_.step()

    with torch.no_grad():
        preds = pred_logits > 0.0
        if use_deep_supervision is not None:
            small_loss = masked_mae_loss(pred_small, event_regression_values[0], event_regression_masks[0]).item()
            mid_loss = masked_mae_loss(pred_mid, event_regression_values[1], event_regression_masks[1]).item()
            large_loss = masked_mae_loss(pred_large, event_regression_values[2], event_regression_masks[2]).item()

            num_events = torch.sum(torch.sum(event_regression_masks[0], dim=(1, 2)) > 0).item() # number of events in batch
        else:
            small_loss = mid_loss = large_loss = num_events = None

    return loss.item(), preds.to(torch.long), small_loss, mid_loss, large_loss, num_events

def training_step(record: bool):
    if record:
        for key in train_metrics:
            train_metrics[key].reset()

    # shuffle
    training_sampler.shuffle()

    # training
    with tqdm.tqdm(total=len(training_sampler)) as pbar:
        while training_sampler.entries_remaining() > 0:
            accel_data_batch, labels_batch, event_regression_values, event_regression_masks, increment =\
                training_sampler.sample(batch_size, random_shift=random_shift,
                                        random_flip=random_flip, always_flip=always_flip,
                                        expand=expand, event_regressions=use_deep_supervision)

            accel_data_batch_torch = torch.tensor(accel_data_batch, dtype=torch.float32, device=config.device)
            labels_batch_torch = torch.tensor(labels_batch, dtype=torch.float32, device=config.device)
            if use_deep_supervision is None:
                event_regression_values_torch = None
                event_regression_masks_torch = None
            else:
                event_regression_values_torch = [torch.tensor(event_regression_values[k], dtype=torch.float32, device=config.device)
                                                    for k in range(3)]
                event_regression_masks_torch = [torch.tensor(event_regression_masks[k], dtype=torch.float32, device=config.device)
                                                    for k in range(3)]

            if use_anglez_only:
                accel_data_batch_torch = accel_data_batch_torch[:, 0:1, :]
            elif use_enmo_only:
                accel_data_batch_torch = accel_data_batch_torch[:, 1:2, :]

            # train model now
            loss, preds, small_loss, mid_loss, large_loss, num_events = single_training_step(model, optimizer,
                                                                           accel_data_batch_torch,
                                                                           labels_batch_torch,
                                                                           event_regression_values_torch,
                                                                           event_regression_masks_torch)
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

                    if use_deep_supervision is not None:
                        train_metrics["mae_small"].add(small_loss, num_events)
                        train_metrics["mae_mid"].add(mid_loss, num_events)
                        train_metrics["mae_large"].add(large_loss, num_events)

            pbar.update(increment)

    if record:
        current_metrics = {}
        for key in train_metrics:
            train_metrics[key].write_to_dict(current_metrics)

        for key in current_metrics:
            train_history[key].append(current_metrics[key])

def single_validation_step(model_: torch.nn.Module, accel_data_batch: torch.Tensor,
                           labels_batch: torch.Tensor, event_regression_values: list[torch.Tensor],
                           event_regression_masks: list[torch.Tensor]):
    with torch.no_grad():
        pred_logits, pred_small, pred_mid, pred_large = model_(accel_data_batch, ret_type=("all" if use_deep_supervision is not None else "attn"))  # shape (batch_size, 2, T)
        if use_ce_loss:
            loss = ce_loss(pred_logits, labels_batch)
        elif use_iou_loss:
            loss = 0.01 * dice_loss(pred_logits, labels_batch) + focal_loss(pred_logits, labels_batch)
        else:
            loss = focal_loss(pred_logits, labels_batch)

        preds = pred_logits > 0.0
        if use_deep_supervision is not None:
            small_loss = masked_mae_loss(pred_small, event_regression_values[0], event_regression_masks[0]).item()
            mid_loss = masked_mae_loss(pred_mid, event_regression_values[1], event_regression_masks[1]).item()
            large_loss = masked_mae_loss(pred_large, event_regression_values[2], event_regression_masks[2]).item()

            num_events = torch.sum(torch.sum(event_regression_masks[0], dim=(1, 2)) > 0).item() # number of events in batch
        else:
            small_loss = mid_loss = large_loss = num_events = None

        return loss.item(), preds.to(torch.long), small_loss, mid_loss, large_loss, num_events

def validation_step():
    for key in val_metrics:
        val_metrics[key].reset()

    # validation
    val_sampler.shuffle()
    with (tqdm.tqdm(total=len(val_sampler)) as pbar):
        while val_sampler.entries_remaining() > 0:
            # load the batch
            accel_data_batch, labels_batch, event_regression_values, event_regression_masks, increment =\
                val_sampler.sample(batch_size, expand=expand, event_regressions=use_deep_supervision)
            accel_data_batch = torch.tensor(accel_data_batch, dtype=torch.float32, device=config.device)
            labels_batch = torch.tensor(labels_batch, dtype=torch.float32, device=config.device)
            if use_deep_supervision is None:
                event_regression_values = None
                event_regression_masks = None
            else:
                event_regression_values = [torch.tensor(event_regression_values[k], dtype=torch.float32, device=config.device)
                                                    for k in range(3)]
                event_regression_masks = [torch.tensor(event_regression_masks[k], dtype=torch.float32, device=config.device)
                                                    for k in range(3)]

            if use_anglez_only:
                accel_data_batch = accel_data_batch[:, 0:1, :]
            elif use_enmo_only:
                accel_data_batch = accel_data_batch[:, 1:2, :]

            # val model now
            loss, preds, small_loss, mid_loss, large_loss, num_events = single_validation_step(model, accel_data_batch, labels_batch, event_regression_values, event_regression_masks)

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

                if use_deep_supervision is not None:
                    val_metrics["mae_small"].add(small_loss, num_events)
                    val_metrics["mae_mid"].add(mid_loss, num_events)
                    val_metrics["mae_large"].add(large_loss, num_events)

            pbar.update(increment)

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
    parser.add_argument("--disable_deep_upconv_contraction", action="store_true", help="Whether to disable the deep upconv contraction. Default False.")
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
    parser.add_argument("--use_deep_supervision", type=int, nargs="+", default=None, help="Whether to use deep supervision. Default None. If specified, must be an integer indicating the length of deep supervision training.")
    parser.add_argument("--use_anglez_only", action="store_true", help="Whether to use only anglez. Default False.")
    parser.add_argument("--use_enmo_only", action="store_true", help="Whether to use only enmo. Default False.")
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
    disable_deep_upconv_contraction = args.disable_deep_upconv_contraction
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
    use_anglez_only = args.use_anglez_only
    use_enmo_only = args.use_enmo_only
    batch_size = args.batch_size
    num_extra_steps = args.num_extra_steps

    assert not (use_iou_loss and use_ce_loss), "Cannot use both IOU loss and cross entropy loss."
    assert not (use_anglez_only and use_enmo_only), "Cannot use both anglez only and enmo only."
    if use_deep_supervision is not None:
        assert isinstance(use_deep_supervision, list), "Deep supervision must be a list of integers."
        assert len(use_deep_supervision) == 3, "Deep supervision must be a list of length 3."
        assert all([isinstance(x, int) for x in use_deep_supervision]), "Deep supervision must be a list of integers."

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
    print("Hidden blocks: " + str(hidden_blocks))
    print("Kernel size: " + str(kernel_size))
    print("Attention bottleneck: " + str(attention_bottleneck))
    print("Deep upconv contraction: " + str(not disable_deep_upconv_contraction))
    model_unet.BATCH_NORM_MOMENTUM = 1 - momentum

    # initialize model
    in_channels = 1 if (use_anglez_only or use_enmo_only) else 2
    model = model_attention_unet.Unet3fDeepSupervision(in_channels, hidden_channels, kernel_size=kernel_size, blocks=hidden_blocks,
                            bottleneck_factor=bottleneck_factor, squeeze_excitation=squeeze_excitation,
                            squeeze_excitation_bottleneck_factor=4,
                            dropout=dropout, dropout_pos_embeddings=dropout_pos_embeddings,
                            use_batch_norm=use_batch_norm, out_channels=2, attn_out_channels=2, attention_bottleneck=attention_bottleneck,
                            expected_attn_input_length=17280 + (2 * expand),

                            deep_supervision_contraction=not disable_deep_upconv_contraction)
    model = model.to(config.device)

    # initialize optimizer
    loss_print = "Cross entropy" if use_ce_loss else ("IOU" if use_iou_loss else "Focal")
    print("Loss: " + loss_print)
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
        warmup_steps = 0
        for g in optimizer.param_groups:
            g["lr"] = learning_rate
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
        "use_iou_loss": use_iou_loss,
        "use_deep_supervision": use_deep_supervision,
        "use_anglez_only": use_anglez_only,
        "use_enmo_only": use_enmo_only,
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
    if use_deep_supervision is not None:
        train_metrics["mae_small"] = metrics.NumericalMetric("train_mae_small")
        train_metrics["mae_mid"] = metrics.NumericalMetric("train_mae_mid")
        train_metrics["mae_large"] = metrics.NumericalMetric("train_mae_large")
    train_metrics["onset_iou_metric"] = metrics.BinaryMetrics("train_onset_iou_metric")
    train_metrics["wakeup_iou_metric"] = metrics.BinaryMetrics("train_wakeup_iou_metric")
    val_metrics["loss"] = metrics.NumericalMetric("val_loss")
    val_metrics["metric"] = metrics.BinaryMetrics("val_metric")
    if use_deep_supervision is not None:
        val_metrics["mae_small"] = metrics.NumericalMetric("val_mae_small")
        val_metrics["mae_mid"] = metrics.NumericalMetric("val_mae_mid")
        val_metrics["mae_large"] = metrics.NumericalMetric("val_mae_large")
    val_metrics["onset_metric"] = metrics.BinaryMetrics("val_onset_metric")
    val_metrics["wakeup_metric"] = metrics.BinaryMetrics("val_wakeup_metric")
    val_metrics["onset_iou_metric1"] = metrics.BinaryMetrics("val_onset_iou_metric1")
    val_metrics["wakeup_iou_metric1"] = metrics.BinaryMetrics("val_wakeup_iou_metric1")
    val_metrics["onset_iou_metric3"] = metrics.BinaryMetrics("val_onset_iou_metric3")
    val_metrics["wakeup_iou_metric3"] = metrics.BinaryMetrics("val_wakeup_iou_metric3")
    val_metrics["onset_iou_metric5"] = metrics.BinaryMetrics("val_onset_iou_metric5")
    val_metrics["wakeup_iou_metric5"] = metrics.BinaryMetrics("val_wakeup_iou_metric5")

    # Compile
    #single_training_step_compile = torch.compile(single_training_step)

    # Initialize the sampler
    print("Initializing the samplers...")
    print("Batch size: " + str(batch_size))
    print("Random shift: " + str(random_shift))
    print("Random flip: " + str(random_flip))
    print("Always flip: " + str(always_flip))
    print("Expand: " + str(expand))
    print("Use anglez only: " + str(use_anglez_only))
    print("Use deep supervision: " + str(use_deep_supervision))
    training_sampler = convert_to_interval_events.IntervalEventsSampler(training_entries, all_data, all_interval_segmentations,
                                                                        train_or_test="train")
    val_sampler = convert_to_interval_events.IntervalEventsSampler(validation_entries, all_data, all_interval_segmentations,
                                                                        train_or_test="val")


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