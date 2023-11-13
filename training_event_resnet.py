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
import matplotlib.pyplot as plt

import bad_series_list
import config
import manager_folds
import manager_models
import metrics
import metrics_ap
import metrics_iou
import logging_memory_utils
import convert_to_npy_naive
import convert_to_interval_events
import convert_to_pred_events
import convert_to_seriesid_events
import model_unet
import model_event_unet

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
                            accel_data_batch: torch.Tensor, labels_batch: torch.Tensor, times_batch: np.ndarray):
    optimizer_.zero_grad()
    times_input = times_batch if use_time_information else None
    pred_logits, _, _, _ = model_(accel_data_batch, ret_type="attn", time=times_input)
    if predict_center_mode == "center":
        pred_logits = pred_logits[:, :, expand:-expand]
    if use_ce_loss:
        loss = ce_loss(pred_logits, labels_batch)
    elif use_iou_loss:
        loss = 0.01 * dice_loss(pred_logits, labels_batch) + focal_loss(pred_logits, labels_batch)
    elif use_ce_iou_loss:
        loss = ce_loss(pred_logits, labels_batch) + 0.01 * dice_loss(pred_logits, labels_batch)
    else:
        loss = focal_loss(pred_logits, labels_batch)

    loss.backward()
    optimizer_.step()

    with torch.no_grad():
        preds = pred_logits > 0.0
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
            accel_data_batch, labels_batch, times_batch, increment =\
                training_sampler.sample(batch_size, random_shift=random_shift,
                                        random_flip=random_flip, random_vflip=flip_value, always_flip=always_flip,
                                        expand=expand, elastic_deformation=use_elastic_deformation, v_elastic_deformation=use_velastic_deformation,
                                        include_all_events=include_all_events, include_events_in_extension=(predict_center_mode == "expanded"))
            assert labels_batch.shape[-1] == 2 * expand + prediction_length, "labels_batch.shape = {}".format(labels_batch.shape)
            assert accel_data_batch.shape[-1] == 2 * expand + prediction_length, "accel_data_batch.shape = {}".format(accel_data_batch.shape)

            accel_data_batch_torch = torch.tensor(accel_data_batch, dtype=torch.float32, device=config.device)
            labels_batch_torch = torch.tensor(labels_batch, dtype=torch.float32, device=config.device)

            if use_anglez_only:
                accel_data_batch_torch = accel_data_batch_torch[:, 0:1, :]
            elif use_enmo_only:
                accel_data_batch_torch = accel_data_batch_torch[:, 1:2, :]
            elif mix_anglez_enmo:
                if np.random.rand() < 0.5:
                    accel_data_batch_torch = accel_data_batch_torch[:, 0:1, :]
                else:
                    accel_data_batch_torch = accel_data_batch_torch[:, 1:2, :]

            if predict_center_mode == "center":
                labels_batch_torch = labels_batch_torch[:, :, expand:-expand]

            # train model now
            loss, preds, small_loss, mid_loss, large_loss, num_events = single_training_step(model, optimizer,
                                                                           accel_data_batch_torch,
                                                                           labels_batch_torch, times_batch)
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
                           labels_batch: torch.Tensor, times_batch: np.ndarray):
    with torch.no_grad():
        times_input = times_batch if use_time_information else None
        pred_logits, _, _, _ = model_(accel_data_batch, ret_type="attn", time=times_input)
        if (predict_center_mode == "center" or predict_center_mode == "expanded"):
            pred_logits = pred_logits[:, :, expand:-expand]
        if use_ce_loss:
            loss = ce_loss(pred_logits, labels_batch)
        elif use_iou_loss:
            loss = 0.01 * dice_loss(pred_logits, labels_batch) + focal_loss(pred_logits, labels_batch)
        elif use_ce_iou_loss:
            loss = ce_loss(pred_logits, labels_batch) + 0.01 * dice_loss(pred_logits, labels_batch)
        else:
            loss = focal_loss(pred_logits, labels_batch)

        preds = pred_logits > 0.0
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
            accel_data_batch, labels_batch, times_batch, increment = val_sampler.sample(batch_size, expand=expand, include_all_events=include_all_events,
                                                                           include_events_in_extension=(predict_center_mode == "expanded"))
            accel_data_batch = torch.tensor(accel_data_batch, dtype=torch.float32, device=config.device)
            labels_batch = torch.tensor(labels_batch, dtype=torch.float32, device=config.device)

            if use_anglez_only:
                accel_data_batch = accel_data_batch[:, 0:1, :]
            elif use_enmo_only:
                accel_data_batch = accel_data_batch[:, 1:2, :]
            elif mix_anglez_enmo:
                accel_data_batch = accel_data_batch[:, 0:1, :]

            if (predict_center_mode == "center" or predict_center_mode == "expanded"):
                labels_batch = labels_batch[:, :, expand:-expand]

            # val model now
            loss, preds, small_loss, mid_loss, large_loss, num_events = single_validation_step(model, accel_data_batch, labels_batch, times_batch)

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

    current_metrics = {}
    for key in val_metrics:
        val_metrics[key].write_to_dict(current_metrics)

    for key in current_metrics:
        val_history[key].append(current_metrics[key])

def plot_single_precision_recall_curve(ax, precisions, recalls, ap, title):
    ax.plot(recalls, precisions)
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, "AP: {:.4f}".format(ap), horizontalalignment="center", verticalalignment="center")

validation_AP_tolerances = [1, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30][::-1]
def validation_ap(epoch, ap_log_dir, predicted_events, gt_events):
    ap_onset_metrics = [metrics_ap.EventMetrics(name="", tolerance=tolerance * 12) for tolerance in validation_AP_tolerances]
    ap_wakeup_metrics = [metrics_ap.EventMetrics(name="", tolerance=tolerance * 12) for tolerance in validation_AP_tolerances]

    with torch.no_grad():
        for series_id in tqdm.tqdm(validation_entries):
            # load the batch
            accel_data = all_data[series_id]["accel"]
            if use_anglez_only:
                accel_data = accel_data[0:1, :]
            elif use_enmo_only:
                accel_data = accel_data[1:2, :]
            elif mix_anglez_enmo:
                accel_data = accel_data[0:1, :]

            # load the times if required
            if use_time_information:
                times = {"hours": all_data[series_id]["hours"], "mins": all_data[series_id]["mins"], "secs": all_data[series_id]["secs"]}
            else:
                times = None

            proba_preds = model_event_unet.event_confidence_inference(model=model, time_series=accel_data,
                                                                    batch_size=batch_size * 5,
                                                                    prediction_length=prediction_length,
                                                                    expand=expand, times=times)
            onset_IOU_probas = iou_score_converter.convert(proba_preds[0, :])
            wakeup_IOU_probas = iou_score_converter.convert(proba_preds[1, :])

            # load the regression predictions
            preds_locs = predicted_events[series_id]
            preds_locs_onset = preds_locs["onset"]
            preds_locs_wakeup = preds_locs["wakeup"]

            # restrict
            onset_IOU_probas = onset_IOU_probas[preds_locs_onset]
            wakeup_IOU_probas = wakeup_IOU_probas[preds_locs_wakeup]

            # get the ground truth
            gt_onset_locs = gt_events[series_id]["onset"]
            gt_wakeup_locs = gt_events[series_id]["wakeup"]

            # add info
            for ap_onset_metric, ap_wakeup_metric in zip(ap_onset_metrics, ap_wakeup_metrics):
                ap_onset_metric.add(pred_locs=preds_locs_onset, pred_probas=onset_IOU_probas, gt_locs=gt_onset_locs)
                ap_wakeup_metric.add(pred_locs=preds_locs_wakeup, pred_probas=wakeup_IOU_probas, gt_locs=gt_wakeup_locs)

    # compute average precision
    ctime = time.time()
    ap_onset_precisions, ap_onset_recalls, ap_onset_average_precisions = [], [], []
    ap_wakeup_precisions, ap_wakeup_recalls, ap_wakeup_average_precisions = [], [], []
    for ap_onset_metric, ap_wakeup_metric in zip(ap_onset_metrics, ap_wakeup_metrics):
        ap_onset_precision, ap_onset_recall, ap_onset_average_precision = ap_onset_metric.get()
        ap_wakeup_precision, ap_wakeup_recall, ap_wakeup_average_precision = ap_wakeup_metric.get()
        ap_onset_precisions.append(ap_onset_precision)
        ap_onset_recalls.append(ap_onset_recall)
        ap_onset_average_precisions.append(ap_onset_average_precision)
        ap_wakeup_precisions.append(ap_wakeup_precision)
        ap_wakeup_recalls.append(ap_wakeup_recall)
        ap_wakeup_average_precisions.append(ap_wakeup_average_precision)
    print("AP computation time: {:.2f} seconds".format(time.time() - ctime))

    # write to dict
    ctime = time.time()
    for k in range(len(validation_AP_tolerances)):
        val_history["val_onset_ap{}".format(validation_AP_tolerances[k])].append(ap_onset_average_precisions[k])
        val_history["val_wakeup_ap{}".format(validation_AP_tolerances[k])].append(ap_wakeup_average_precisions[k])
    val_history["val_onset_mAP"].append(np.mean(ap_onset_average_precisions))
    val_history["val_wakeup_mAP"].append(np.mean(ap_wakeup_average_precisions))

    # draw the precision-recall curve using matplotlib onto file "epoch{}_AP.png".format(epoch) inside the ap_log_dir
    fig, axes=  plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle("Epoch {} (Onset mAP: {}, Wakeup mAP: {})".format(epoch, np.mean(ap_onset_average_precisions), np.mean(ap_wakeup_average_precisions)))
    for k in range(len(validation_AP_tolerances)):
        ax = axes[k // 5, k % 5]
        plot_single_precision_recall_curve(ax, ap_onset_precisions[k], ap_onset_recalls[k], ap_onset_average_precisions[k], "Onset AP{}".format(validation_AP_tolerances[k]))
        ax = axes[(k + 10) // 5, (k + 10) % 5]
        plot_single_precision_recall_curve(ax, ap_wakeup_precisions[k], ap_wakeup_recalls[k], ap_wakeup_average_precisions[k], "Wakeup AP{}".format(validation_AP_tolerances[k]))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(os.path.join(ap_log_dir, "epoch{}_AP.png".format(epoch)))
    plt.close()

    print("Plotting time: {:.2f} seconds".format(time.time() - ctime))

def print_history(metrics_history):
    for key in metrics_history:
        print("{}      {}".format(key, metrics_history[key][-1]))

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    all_data = convert_to_npy_naive.load_all_data_into_dict()

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
    parser.add_argument("--attention_blocks", type=int, default=4, help="Number of attention blocks to use. Default 4.")
    parser.add_argument("--attention_bottleneck", type=int, default=None, help="The bottleneck factor of the attention module. Default None.")
    parser.add_argument("--attention_mode", type=str, default="learned", help="Attention mode. Default 'learned'. Must be 'learned', 'length' or 'pairwise_length'.")
    parser.add_argument("--upconv_channels_override", type=int, default=None, help="Number of fixed channels for the upsampling path. Default None, do not override.")
    parser.add_argument("--random_shift", type=int, default=0, help="Randomly shift the intervals by at most this amount. Default 0.")
    parser.add_argument("--random_flip", action="store_true", help="Randomly flip the intervals. Default False.")
    parser.add_argument("--always_flip", action="store_true", help="Always flip the intervals. Default False.")
    parser.add_argument("--flip_value", action="store_true", help="Whether to flip the value of the intervals. Default False.")
    parser.add_argument("--expand", type=int, default=0, help="Expand the intervals by this amount. Default 0.")
    parser.add_argument("--use_batch_norm", action="store_true", help="Whether to use batch norm. Default False.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate. Default 0.0.")
    parser.add_argument("--dropout_pos_embeddings", action="store_true", help="Whether to dropout the positional embeddings. Default False.")
    parser.add_argument("--attn_dropout", type=float, default=0.0, help="Attention dropout rate. Default 0.0.")
    parser.add_argument("--use_time_information", action="store_true", help="Whether to use time information. Default False.")
    parser.add_argument("--use_elastic_deformation", action="store_true", help="Whether to use elastic deformation. Default False.")
    parser.add_argument("--use_velastic_deformation", action="store_true", help="Whether to use velastic deformation (only available for anglez). Default False.")
    parser.add_argument("--use_ce_loss", action="store_true", help="Whether to use cross entropy loss. Default False.")
    parser.add_argument("--use_iou_loss", action="store_true", help="Whether to use IOU loss. Default False.")
    parser.add_argument("--use_ce_iou_loss", action="store_true", help="Whether to use a combination of cross entropy and IOU loss. Default False.")
    parser.add_argument("--use_anglez_only", action="store_true", help="Whether to use only anglez. Default False.")
    parser.add_argument("--use_enmo_only", action="store_true", help="Whether to use only enmo. Default False.")
    parser.add_argument("--mix_anglez_enmo", action="store_true", help="Whether to mix anglez and enmo. Default False.")
    parser.add_argument("--include_all_events", action="store_true", help="Whether to include all events. Default False.")
    parser.add_argument("--exclude_bad_series_from_training", action="store_true", help="Whether to exclude bad series from training. Default False.")
    parser.add_argument("--prediction_length", type=int, default=17280, help="Number of timesteps to predict. Default 17280.")
    parser.add_argument("--prediction_stride", type=int, default=4320, help="Number of timesteps to stride when predicting. Default 4320.")
    parser.add_argument("--predict_center_mode", type=str, default="all", help="Optimization target for the center (expanded parts omitted) and the expanded parts of the intervals. Default all.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size. Default 512.")
    parser.add_argument("--num_extra_steps", type=int, default=0, help="Extra steps of gradient descent before the usual step in an epoch. Default 0.")
    parser.add_argument("--log_average_precision", action="store_true", help="Whether to log the average precision. Default False.")
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
    validation_entries = [series_id for series_id in validation_entries if series_id not in bad_series_list.noisy_bad_segmentations] # exclude

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
    attention_blocks = args.attention_blocks
    attention_bottleneck = args.attention_bottleneck
    attention_mode = args.attention_mode
    upconv_channels_override = args.upconv_channels_override
    random_shift = args.random_shift
    random_flip = args.random_flip
    always_flip = args.always_flip
    flip_value = args.flip_value
    expand = args.expand
    use_batch_norm = args.use_batch_norm
    dropout = args.dropout
    dropout_pos_embeddings = args.dropout_pos_embeddings
    attn_dropout = args.attn_dropout
    use_time_information = args.use_time_information
    use_elastic_deformation = args.use_elastic_deformation
    use_velastic_deformation = args.use_velastic_deformation
    use_ce_loss = args.use_ce_loss
    use_iou_loss = args.use_iou_loss
    use_ce_iou_loss = args.use_ce_iou_loss
    use_anglez_only = args.use_anglez_only
    use_enmo_only = args.use_enmo_only
    mix_anglez_enmo = args.mix_anglez_enmo
    include_all_events = args.include_all_events
    exclude_bad_series_from_training = args.exclude_bad_series_from_training
    prediction_length = args.prediction_length
    prediction_stride = args.prediction_stride
    predict_center_mode = args.predict_center_mode
    batch_size = args.batch_size
    num_extra_steps = args.num_extra_steps
    log_average_precision = args.log_average_precision

    assert not (use_iou_loss and use_ce_loss), "Cannot use both IOU loss and cross entropy loss."
    assert sum([use_anglez_only, use_enmo_only, mix_anglez_enmo]) <= 1, "Cannot use more than one of anglez only, enmo only, and mix anglez and enmo."
    assert predict_center_mode in ["all", "center", "expanded"], "predict_center_mode must be one of all, center, expanded."
    assert not (use_time_information and random_flip), "Cannot use time information and random flip at the same time."
    # all      - predict all parts of the interval, events only happen in the center, and the expanded parts will be predicted as negative (for both training and validation)
    # center   - predict only the center parts of the interval. No optimization and predictions will be made for the expanded parts (for both training and validation)
    # expanded - use the entire interval (expanded parts included) for optimization, and only the center part for prediction (validation)
    if exclude_bad_series_from_training:
        training_entries = [series_id for series_id in training_entries if series_id not in bad_series_list.noisy_bad_segmentations]
    if log_average_precision:
        assert os.path.isdir("./inference_regression_statistics/regression_preds/"), "Must generate regression predictions first. See inference_regression_statistics folder."
        per_series_id_events = convert_to_seriesid_events.get_events_per_seriesid()
        regression_predicted_events = convert_to_pred_events.load_all_pred_events_into_dict()
        iou_score_converter = model_event_unet.ProbasIOUScoreConverter(intersection_width=30 * 12, union_width=60 * 12, device=config.device)
        ap_log_dir = os.path.join(model_dir, "ap_log")
        os.mkdir(ap_log_dir)
    else:
        per_series_id_events = None
        regression_predicted_events = None
        iou_score_converter = None
        ap_log_dir = None

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
    print("Attention dropout: " + str(attn_dropout))
    print("Batch norm: " + str(use_batch_norm))
    print("Squeeze excitation: " + str(squeeze_excitation))
    print("Bottleneck factor: " + str(bottleneck_factor))
    print("Hidden channels: " + str(hidden_channels))
    print("Hidden blocks: " + str(hidden_blocks))
    print("Kernel size: " + str(kernel_size))
    print("Attention bottleneck: " + str(attention_bottleneck))
    print("Attention mode: " + str(attention_mode))
    print("Upconv channels: " + str(upconv_channels_override))
    model_unet.BATCH_NORM_MOMENTUM = 1 - momentum

    # initialize model
    in_channels = 1 if (use_anglez_only or use_enmo_only or mix_anglez_enmo) else 2
    model = model_event_unet.EventConfidenceUnet(in_channels, hidden_channels, kernel_size=kernel_size, blocks=hidden_blocks,
                            bottleneck_factor=bottleneck_factor, squeeze_excitation=squeeze_excitation,
                            squeeze_excitation_bottleneck_factor=4,
                            dropout=dropout, dropout_pos_embeddings=dropout_pos_embeddings,
                            use_batch_norm=use_batch_norm, attn_out_channels=2, attention_bottleneck=attention_bottleneck,
                            expected_attn_input_length=17280 + (2 * expand), attention_blocks=attention_blocks,
                            upconv_channels_override=upconv_channels_override, attention_mode=attention_mode,
                            attention_dropout=attn_dropout, use_time_input=use_time_information)
    model = model.to(config.device)

    # initialize optimizer
    loss_print = "Cross entropy" if use_ce_loss else ("IOU" if use_iou_loss else ("Cross entropy + IOU" if use_ce_iou_loss else "Focal"))
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
        "attention_blocks": attention_blocks,
        "attention_bottleneck": attention_bottleneck,
        "attention_mode": attention_mode,
        "upconv_channels_override": upconv_channels_override,
        "random_shift": random_shift,
        "random_flip": random_flip,
        "always_flip": always_flip,
        "flip_value": flip_value,
        "expand": expand,
        "use_batch_norm": use_batch_norm,
        "dropout": dropout,
        "dropout_pos_embeddings": dropout_pos_embeddings,
        "attn_dropout": attn_dropout,
        "use_time_information": use_time_information,
        "use_elastic_deformation": use_elastic_deformation,
        "use_velastic_deformation": use_velastic_deformation,
        "use_ce_loss": use_ce_loss,
        "use_iou_loss": use_iou_loss,
        "use_ce_iou_loss": use_ce_iou_loss,
        "use_anglez_only": use_anglez_only,
        "use_enmo_only": use_enmo_only,
        "mix_anglez_enmo": mix_anglez_enmo,
        "include_all_events": include_all_events,
        "exclude_bad_series_from_training": exclude_bad_series_from_training,
        "prediction_length": prediction_length,
        "prediction_stride": prediction_stride,
        "predict_center_mode": predict_center_mode,
        "batch_size": batch_size,
        "num_extra_steps": num_extra_steps,
        "log_average_precision": log_average_precision,
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

    # Compile
    #single_training_step_compile = torch.compile(single_training_step)

    # Initialize the sampler
    print("Initializing the samplers...")
    print("Batch size: " + str(batch_size))
    print("Random shift: " + str(random_shift))
    print("Random flip: " + str(random_flip))
    print("Always flip: " + str(always_flip))
    print("Elastic deformation: " + str(use_elastic_deformation))
    print("Expand: " + str(expand))
    print("Prediction mode:" + str(predict_center_mode))
    print("Use anglez only: " + str(use_anglez_only))
    print("Use enmo only: " + str(use_enmo_only))
    print("Mix anglez and enmo: " + str(mix_anglez_enmo))
    training_sampler = convert_to_interval_events.IntervalEventsSampler(training_entries, all_data,
                                                                        train_or_test="train",
                                                                        prediction_length=prediction_length,
                                                                        prediction_stride=prediction_stride)
    val_sampler = convert_to_interval_events.IntervalEventsSampler(validation_entries, all_data,
                                                                        train_or_test="val",
                                                                        prediction_length=prediction_length,
                                                                        prediction_stride=prediction_stride)


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
                if log_average_precision:
                    validation_ap(epoch=epoch, ap_log_dir=ap_log_dir, predicted_events=regression_predicted_events,
                                  gt_events=per_series_id_events)

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