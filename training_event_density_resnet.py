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
import matplotlib.pyplot as plt
import torch

import bad_series_list
import config
import manager_folds
import manager_models
import metrics
import metrics_ap
import logging_memory_utils
import convert_to_npy_naive
import convert_to_interval_density_events
import convert_to_pred_events
import convert_to_seriesid_events
import model_unet
import model_event_density_unet
import postprocessing

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


def single_training_step(model_: torch.nn.Module, optimizer_: torch.optim.Optimizer,
                            accel_data_batch: torch.Tensor, labels_density_batch: torch.Tensor,
                            labels_occurrence_batch: torch.Tensor, times_batch: np.ndarray):
    optimizer_.zero_grad()
    times_input = times_batch if use_time_information else None
    pred_density_logits, pred_occurences = model_(accel_data_batch, time=times_input)
    entropy_loss = torch.nn.functional.cross_entropy(pred_density_logits, labels_density_batch, reduction="none").mean(dim=-1).sum()
    class_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_occurences, labels_occurrence_batch, reduction="none").mean(dim=-1).sum()
    loss = entropy_loss + class_loss

    loss.backward()
    optimizer_.step()

    with torch.no_grad():
        pred_occurences = pred_occurences > 0.0

    return loss.item(), class_loss.item(), pred_occurences.to(torch.long), entropy_loss.item()

def training_step(record: bool):
    if record:
        for key in train_metrics:
            train_metrics[key].reset()

    # shuffle
    training_sampler.shuffle()

    # training
    with tqdm.tqdm(total=len(training_sampler)) as pbar:
        while training_sampler.entries_remaining() > 0:
            accel_data_batch, event_infos, times_batch, increment =\
                training_sampler.sample(batch_size, random_shift=random_shift,
                                        random_flip=random_flip, random_vflip=flip_value, always_flip=always_flip,
                                        expand=expand, elastic_deformation=use_elastic_deformation, v_elastic_deformation=use_velastic_deformation,

                                        return_mode="expanded_interval_density")
            labels_density_batch = event_infos["density"]
            labels_occurrence_batch = event_infos["occurrence"]
            assert labels_density_batch.shape[-1] == 2 * expand + prediction_length, "labels_batch.shape = {}".format(labels_density_batch.shape)
            assert labels_density_batch.shape[-2] == 2, "labels_batch.shape = {}".format(labels_density_batch.shape)
            assert len(labels_occurrence_batch.shape) == 2, "labels_occurrence_batch.shape = {}".format(labels_occurrence_batch.shape)
            assert labels_occurrence_batch.shape[-1] == 2, "labels_occurrence_batch.shape = {}".format(labels_occurrence_batch.shape)
            assert accel_data_batch.shape[-1] == 2 * expand + prediction_length, "accel_data_batch.shape = {}".format(accel_data_batch.shape)

            accel_data_batch_torch = torch.tensor(accel_data_batch, dtype=torch.float32, device=config.device)
            labels_density_batch_torch = torch.tensor(labels_density_batch, dtype=torch.float32, device=config.device)
            labels_occurrence_batch_torch = torch.tensor(labels_occurrence_batch, dtype=torch.float32, device=config.device)
            with torch.no_grad():
                labels_density_batch_torch = labels_density_batch_torch.permute(0, 2, 1)

            if use_anglez_only:
                accel_data_batch_torch = accel_data_batch_torch[:, 0:1, :]
            elif use_enmo_only:
                accel_data_batch_torch = accel_data_batch_torch[:, 1:2, :]

            # train model now
            loss, class_loss, pred_occurences, entropy_loss = single_training_step(model, optimizer,
                                                                           accel_data_batch_torch,
                                                                           labels_density_batch_torch,
                                                                           labels_occurrence_batch_torch, times_batch)
            #time.sleep(0.2)

            # record
            if record:
                with torch.no_grad():
                    labels_occurrence_long = (labels_occurrence_batch_torch > 0.5).to(torch.long)
                    train_metrics["loss"].add(loss, increment)
                    train_metrics["class_loss"].add(class_loss, increment)
                    train_metrics["class_metric"].add(pred_occurences, labels_occurrence_long)
                    train_metrics["entropy_loss"].add(entropy_loss, increment)

            pbar.update(increment)

    if record:
        current_metrics = {}
        for key in train_metrics:
            train_metrics[key].write_to_dict(current_metrics)

        for key in current_metrics:
            train_history[key].append(current_metrics[key])

def single_validation_step(model_: torch.nn.Module, accel_data_batch: torch.Tensor,
                           labels_density_batch: torch.Tensor, labels_occurrence_batch: torch.Tensor,
                           times_batch: np.ndarray):
    with torch.no_grad():
        times_input = times_batch if use_time_information else None
        pred_density_logits, pred_occurences = model_(accel_data_batch, time=times_input, return_as_training=True)
        entropy_loss = torch.nn.functional.cross_entropy(pred_density_logits, labels_density_batch,
                                                         reduction="none").mean(dim=-1).sum()
        class_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_occurences, labels_occurrence_batch,
                                                                          reduction="none").mean(dim=-1).sum()
        loss = entropy_loss + class_loss
        pred_occurences = pred_occurences > 0.0
        return loss.item(), class_loss.item(), pred_occurences.to(torch.long), entropy_loss.item()

def validation_step():
    for key in val_metrics:
        val_metrics[key].reset()
    use_model = swa_model if (use_swa and (epoch > swa_start)) else model

    # validation
    val_sampler.shuffle()
    with (tqdm.tqdm(total=len(val_sampler)) as pbar):
        while val_sampler.entries_remaining() > 0:
            # load the batch
            accel_data_batch, event_infos, times_batch, increment = val_sampler.sample(batch_size, expand=expand,
                                                                                        return_mode="expanded_interval_density")
            labels_density_batch = event_infos["density"]
            labels_occurrence_batch = event_infos["occurrence"]

            accel_data_batch = torch.tensor(accel_data_batch, dtype=torch.float32, device=config.device)
            labels_density_batch = torch.tensor(labels_density_batch, dtype=torch.float32, device=config.device)
            labels_occurrence_batch = torch.tensor(labels_occurrence_batch, dtype=torch.float32, device=config.device)
            with torch.no_grad():
                labels_density_batch = labels_density_batch.permute(0, 2, 1)

            if use_anglez_only:
                accel_data_batch = accel_data_batch[:, 0:1, :]
            elif use_enmo_only:
                accel_data_batch = accel_data_batch[:, 1:2, :]

            # val model now
            loss, class_loss, pred_occurences, entropy_loss = single_validation_step(use_model, accel_data_batch,
                                                                                     labels_density_batch,
                                                                                     labels_occurrence_batch,
                                                                                     times_batch)

            # record
            with torch.no_grad():
                labels_occurrence_long = (labels_occurrence_batch > 0.5).to(torch.long)
                val_metrics["loss"].add(loss, increment)
                val_metrics["class_loss"].add(class_loss, increment)
                val_metrics["class_metric"].add(pred_occurences, labels_occurrence_long)
                val_metrics["entropy_loss"].add(entropy_loss, increment)

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
def validation_ap(epoch, ap_log_dir, ap_log_dilated_dir, ap_log_aligned_dir, ap_log_augpruned_dir, predicted_events, gt_events):
    use_model = swa_model if (use_swa and (epoch > swa_start)) else model

    ap_onset_metrics = [metrics_ap.EventMetrics(name="", tolerance=tolerance * 12) for tolerance in validation_AP_tolerances]
    ap_wakeup_metrics = [metrics_ap.EventMetrics(name="", tolerance=tolerance * 12) for tolerance in validation_AP_tolerances]
    ap_onset_metrics_dilated = [metrics_ap.EventMetrics(name="", tolerance=tolerance * 12) for tolerance in validation_AP_tolerances]
    ap_wakeup_metrics_dilated = [metrics_ap.EventMetrics(name="", tolerance=tolerance * 12) for tolerance in validation_AP_tolerances]
    ap_onset_metrics_augpruned = [metrics_ap.EventMetrics(name="", tolerance=tolerance * 12) for tolerance in validation_AP_tolerances]
    ap_wakeup_metrics_augpruned = [metrics_ap.EventMetrics(name="", tolerance=tolerance * 12) for tolerance in validation_AP_tolerances]
    ap_onset_metrics_aligned = [metrics_ap.EventMetrics(name="", tolerance=tolerance * 12) for tolerance in validation_AP_tolerances]
    ap_wakeup_metrics_aligned = [metrics_ap.EventMetrics(name="", tolerance=tolerance * 12) for tolerance in validation_AP_tolerances]

    with torch.no_grad():
        for series_id in tqdm.tqdm(validation_entries):
            # load the batch
            accel_data = all_data[series_id]["accel"]
            if use_anglez_only:
                accel_data = accel_data[0:1, :]
            elif use_enmo_only:
                accel_data = accel_data[1:2, :]

            # load the times if required
            if use_time_information:
                times = {"hours": all_data[series_id]["hours"], "mins": all_data[series_id]["mins"], "secs": all_data[series_id]["secs"]}
            else:
                times = None

            stride_count = 8
            if use_swa and (epoch > swa_start) and ((epoch - swa_start) % 3 != 0):
                stride_count = 1
            proba_preds = model_event_density_unet.event_density_inference(model=use_model, time_series=accel_data,
                                                                      batch_size=batch_size * 5,
                                                                      prediction_length=prediction_length,
                                                                      expand=expand, times=times,
                                                                      device=config.device,
                                                                      use_time_input=use_time_information,
                                                                      stride_count=stride_count)
            onset_IOU_probas = proba_preds[0, :]
            wakeup_IOU_probas = proba_preds[1, :]
            onset_IOU_probas_dilated = dilation_converter.convert(onset_IOU_probas)
            wakeup_IOU_probas_dilated = dilation_converter.convert(wakeup_IOU_probas)

            # load the regression predictions
            preds_locs = predicted_events[series_id]
            preds_locs_onset = preds_locs["onset"]
            preds_locs_wakeup = preds_locs["wakeup"]

            # compute augmentation pruned predictions
            preds_locs_onset_aug = (onset_IOU_probas[1:-1] >= onset_IOU_probas[0:-2]) & (onset_IOU_probas[1:-1] >= onset_IOU_probas[2:])
            preds_locs_onset_aug = np.argwhere(preds_locs_onset_aug).flatten() + 1
            preds_locs_wakeup_aug = (wakeup_IOU_probas[1:-1] >= wakeup_IOU_probas[0:-2]) & (wakeup_IOU_probas[1:-1] >= wakeup_IOU_probas[2:])
            preds_locs_wakeup_aug = np.argwhere(preds_locs_wakeup_aug).flatten() + 1
            if len(preds_locs_onset_aug) > 0:
                preds_locs_onset_aug = postprocessing.prune(preds_locs_onset_aug, onset_IOU_probas, pruning_radius=60)
            if len(preds_locs_wakeup_aug) > 0:
                preds_locs_wakeup_aug = postprocessing.prune(preds_locs_wakeup_aug, wakeup_IOU_probas, pruning_radius=60)
            if len(preds_locs_onset_aug) > 0:
                onset_relative_locs = postprocessing.prune_relative(preds_locs_onset_aug, preds_locs_onset, pruning_radius=60)
                preds_locs_onset_augpruned = np.unique(np.concatenate((preds_locs_onset_aug, onset_relative_locs)))
            else:
                preds_locs_onset_augpruned = preds_locs_onset
            if len(preds_locs_wakeup_aug) > 0:
                wakeup_relative_locs = postprocessing.prune_relative(preds_locs_wakeup_aug, preds_locs_wakeup, pruning_radius=60)
                preds_locs_wakeup_augpruned = np.unique(np.concatenate((preds_locs_wakeup_aug, wakeup_relative_locs)))
            else:
                preds_locs_wakeup_augpruned = preds_locs_wakeup

            # restrict
            onset_IOU_aligned_probas = postprocessing.align_index_probas(preds_locs_onset, onset_IOU_probas)
            wakeup_IOU_aligned_probas = postprocessing.align_index_probas(preds_locs_wakeup, wakeup_IOU_probas)
            onset_IOU_augpruned_probas = postprocessing.align_index_probas(preds_locs_onset_augpruned, onset_IOU_probas)
            wakeup_IOU_augpruned_probas = postprocessing.align_index_probas(preds_locs_wakeup_augpruned, wakeup_IOU_probas)
            onset_IOU_probas = onset_IOU_probas[preds_locs_onset]
            wakeup_IOU_probas = wakeup_IOU_probas[preds_locs_wakeup]
            onset_IOU_probas_dilated = onset_IOU_probas_dilated[preds_locs_onset]
            wakeup_IOU_probas_dilated = wakeup_IOU_probas_dilated[preds_locs_wakeup]

            # get the ground truth
            gt_onset_locs = gt_events[series_id]["onset"]
            gt_wakeup_locs = gt_events[series_id]["wakeup"]

            # add info
            for ap_onset_metric, ap_wakeup_metric, ap_onset_metric_dilated, ap_wakeup_metric_dilated \
                    in zip(ap_onset_metrics, ap_wakeup_metrics, ap_onset_metrics_dilated, ap_wakeup_metrics_dilated):
                ap_onset_metric.add(pred_locs=preds_locs_onset, pred_probas=onset_IOU_probas, gt_locs=gt_onset_locs)
                ap_wakeup_metric.add(pred_locs=preds_locs_wakeup, pred_probas=wakeup_IOU_probas, gt_locs=gt_wakeup_locs)
                ap_onset_metric_dilated.add(pred_locs=preds_locs_onset, pred_probas=onset_IOU_probas_dilated, gt_locs=gt_onset_locs)
                ap_wakeup_metric_dilated.add(pred_locs=preds_locs_wakeup, pred_probas=wakeup_IOU_probas_dilated, gt_locs=gt_wakeup_locs)

            # add aligned and augmented pruned info
            for ap_onset_metrics_aligned, ap_wakeup_metrics_aligned, ap_onset_metrics_augpruned, ap_wakeup_metrics_augpruned \
                    in zip(ap_onset_metrics_aligned, ap_wakeup_metrics_aligned, ap_onset_metrics_augpruned, ap_wakeup_metrics_augpruned):
                ap_onset_metrics_aligned.add(pred_locs=preds_locs_onset, pred_probas=onset_IOU_aligned_probas, gt_locs=gt_onset_locs)
                ap_wakeup_metrics_aligned.add(pred_locs=preds_locs_wakeup, pred_probas=wakeup_IOU_aligned_probas, gt_locs=gt_wakeup_locs)
                ap_onset_metrics_augpruned.add(pred_locs=preds_locs_onset_augpruned, pred_probas=onset_IOU_augpruned_probas, gt_locs=gt_onset_locs)
                ap_wakeup_metrics_augpruned.add(pred_locs=preds_locs_wakeup_augpruned, pred_probas=wakeup_IOU_augpruned_probas, gt_locs=gt_wakeup_locs)



    # compute average precision
    ctime = time.time()
    ap_onset_precisions, ap_onset_recalls, ap_onset_average_precisions = [], [], []
    ap_wakeup_precisions, ap_wakeup_recalls, ap_wakeup_average_precisions = [], [], []
    ap_onset_dilated_precisions, ap_onset_dilated_recalls, ap_onset_dilated_average_precisions = [], [], []
    ap_wakeup_dilated_precisions, ap_wakeup_dilated_recalls, ap_wakeup_dilated_average_precisions = [], [], []
    ap_onset_aligned_precisions, ap_onset_aligned_recalls, ap_onset_aligned_average_precisions = [], [], []
    ap_wakeup_aligned_precisions, ap_wakeup_aligned_recalls, ap_wakeup_aligned_average_precisions = [], [], []
    ap_onset_augpruned_precisions, ap_onset_augpruned_recalls, ap_onset_augpruned_average_precisions = [], [], []
    ap_wakeup_augpruned_precisions, ap_wakeup_augpruned_recalls, ap_wakeup_augpruned_average_precisions = [], [], []
    for ap_onset_metric, ap_wakeup_metric in zip(ap_onset_metrics, ap_wakeup_metrics):
        ap_onset_precision, ap_onset_recall, ap_onset_average_precision, _ = ap_onset_metric.get()
        ap_wakeup_precision, ap_wakeup_recall, ap_wakeup_average_precision, _ = ap_wakeup_metric.get()
        ap_onset_precisions.append(ap_onset_precision)
        ap_onset_recalls.append(ap_onset_recall)
        ap_onset_average_precisions.append(ap_onset_average_precision)
        ap_wakeup_precisions.append(ap_wakeup_precision)
        ap_wakeup_recalls.append(ap_wakeup_recall)
        ap_wakeup_average_precisions.append(ap_wakeup_average_precision)

    for ap_onset_metric_dilated, ap_wakeup_metric_dilated in zip(ap_onset_metrics_dilated, ap_wakeup_metrics_dilated):
        ap_onset_dilated_precision, ap_onset_dilated_recall, ap_onset_dilated_average_precision, _ = ap_onset_metric_dilated.get()
        ap_wakeup_dilated_precision, ap_wakeup_dilated_recall, ap_wakeup_dilated_average_precision, _ = ap_wakeup_metric_dilated.get()
        ap_onset_dilated_precisions.append(ap_onset_dilated_precision)
        ap_onset_dilated_recalls.append(ap_onset_dilated_recall)
        ap_onset_dilated_average_precisions.append(ap_onset_dilated_average_precision)
        ap_wakeup_dilated_precisions.append(ap_wakeup_dilated_precision)
        ap_wakeup_dilated_recalls.append(ap_wakeup_dilated_recall)
        ap_wakeup_dilated_average_precisions.append(ap_wakeup_dilated_average_precision)

    for ap_onset_metric_aligned, ap_wakeup_metric_aligned in zip(ap_onset_metrics_aligned, ap_wakeup_metrics_aligned):
        ap_onset_aligned_precision, ap_onset_aligned_recall, ap_onset_aligned_average_precision, _ = ap_onset_metric_aligned.get()
        ap_wakeup_aligned_precision, ap_wakeup_aligned_recall, ap_wakeup_aligned_average_precision, _ = ap_wakeup_metric_aligned.get()
        ap_onset_aligned_precisions.append(ap_onset_aligned_precision)
        ap_onset_aligned_recalls.append(ap_onset_aligned_recall)
        ap_onset_aligned_average_precisions.append(ap_onset_aligned_average_precision)
        ap_wakeup_aligned_precisions.append(ap_wakeup_aligned_precision)
        ap_wakeup_aligned_recalls.append(ap_wakeup_aligned_recall)
        ap_wakeup_aligned_average_precisions.append(ap_wakeup_aligned_average_precision)

    for ap_onset_metric_augpruned, ap_wakeup_metric_augpruned in zip(ap_onset_metrics_augpruned, ap_wakeup_metrics_augpruned):
        ap_onset_augpruned_precision, ap_onset_augpruned_recall, ap_onset_augpruned_average_precision, _ = ap_onset_metric_augpruned.get()
        ap_wakeup_augpruned_precision, ap_wakeup_augpruned_recall, ap_wakeup_augpruned_average_precision, _ = ap_wakeup_metric_augpruned.get()
        ap_onset_augpruned_precisions.append(ap_onset_augpruned_precision)
        ap_onset_augpruned_recalls.append(ap_onset_augpruned_recall)
        ap_onset_augpruned_average_precisions.append(ap_onset_augpruned_average_precision)
        ap_wakeup_augpruned_precisions.append(ap_wakeup_augpruned_precision)
        ap_wakeup_augpruned_recalls.append(ap_wakeup_augpruned_recall)
        ap_wakeup_augpruned_average_precisions.append(ap_wakeup_augpruned_average_precision)

    print("AP computation time: {:.2f} seconds".format(time.time() - ctime))

    # write to dict
    ctime = time.time()
    val_history["val_onset_mAP"].append(np.mean(ap_onset_average_precisions))
    val_history["val_wakeup_mAP"].append(np.mean(ap_wakeup_average_precisions))
    val_history["val_onset_dilated_mAP"].append(np.mean(ap_onset_dilated_average_precisions))
    val_history["val_wakeup_dilated_mAP"].append(np.mean(ap_wakeup_dilated_average_precisions))
    val_history["val_onset_aligned_mAP"].append(np.mean(ap_onset_aligned_average_precisions))
    val_history["val_wakeup_aligned_mAP"].append(np.mean(ap_wakeup_aligned_average_precisions))
    val_history["val_onset_augpruned_mAP"].append(np.mean(ap_onset_augpruned_average_precisions))
    val_history["val_wakeup_augpruned_mAP"].append(np.mean(ap_wakeup_augpruned_average_precisions))

    # draw the precision-recall curve using matplotlib onto file "epoch{}_AP.png".format(epoch) inside the ap_log_dir
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle("Epoch {} (Onset mAP: {}, Wakeup mAP: {})".format(epoch, np.mean(ap_onset_average_precisions), np.mean(ap_wakeup_average_precisions)))
    for k in range(len(validation_AP_tolerances)):
        ax = axes[k // 5, k % 5]
        plot_single_precision_recall_curve(ax, ap_onset_precisions[k], ap_onset_recalls[k], ap_onset_average_precisions[k], "Onset AP{}".format(validation_AP_tolerances[k]))
        ax = axes[(k + 10) // 5, (k + 10) % 5]
        plot_single_precision_recall_curve(ax, ap_wakeup_precisions[k], ap_wakeup_recalls[k], ap_wakeup_average_precisions[k], "Wakeup AP{}".format(validation_AP_tolerances[k]))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(os.path.join(ap_log_dir, "epoch{}_AP.png".format(epoch)))
    plt.close()

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle("Epoch {} (Onset Dilated mAP: {}, Wakeup Dilated mAP: {})".format(epoch, np.mean(ap_onset_dilated_average_precisions), np.mean(ap_wakeup_dilated_average_precisions)))
    for k in range(len(validation_AP_tolerances)):
        ax = axes[k // 5, k % 5]
        plot_single_precision_recall_curve(ax, ap_onset_dilated_precisions[k], ap_onset_dilated_recalls[k], ap_onset_dilated_average_precisions[k], "Onset Dilated AP{}".format(validation_AP_tolerances[k]))
        ax = axes[(k + 10) // 5, (k + 10) % 5]
        plot_single_precision_recall_curve(ax, ap_wakeup_dilated_precisions[k], ap_wakeup_dilated_recalls[k], ap_wakeup_dilated_average_precisions[k], "Wakeup Dilated AP{}".format(validation_AP_tolerances[k]))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(os.path.join(ap_log_dilated_dir, "epoch{}_AP.png".format(epoch)))
    plt.close()

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle("Epoch {} (Onset Aligned mAP: {}, Wakeup Aligned mAP: {})".format(epoch, np.mean(ap_onset_aligned_average_precisions), np.mean(ap_wakeup_aligned_average_precisions)))
    for k in range(len(validation_AP_tolerances)):
        ax = axes[k // 5, k % 5]
        plot_single_precision_recall_curve(ax, ap_onset_aligned_precisions[k], ap_onset_aligned_recalls[k], ap_onset_aligned_average_precisions[k], "Onset Aligned AP{}".format(validation_AP_tolerances[k]))
        ax = axes[(k + 10) // 5, (k + 10) % 5]
        plot_single_precision_recall_curve(ax, ap_wakeup_aligned_precisions[k], ap_wakeup_aligned_recalls[k], ap_wakeup_aligned_average_precisions[k], "Wakeup Aligned AP{}".format(validation_AP_tolerances[k]))

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(os.path.join(ap_log_aligned_dir, "epoch{}_AP.png".format(epoch)))
    plt.close()

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle("Epoch {} (Onset Augpruned mAP: {}, Wakeup Augpruned mAP: {})".format(epoch, np.mean(ap_onset_augpruned_average_precisions), np.mean(ap_wakeup_augpruned_average_precisions)))
    for k in range(len(validation_AP_tolerances)):
        ax = axes[k // 5, k % 5]
        plot_single_precision_recall_curve(ax, ap_onset_augpruned_precisions[k], ap_onset_augpruned_recalls[k], ap_onset_augpruned_average_precisions[k], "Onset Augpruned AP{}".format(validation_AP_tolerances[k]))
        ax = axes[(k + 10) // 5, (k + 10) % 5]
        plot_single_precision_recall_curve(ax, ap_wakeup_augpruned_precisions[k], ap_wakeup_augpruned_recalls[k], ap_wakeup_augpruned_average_precisions[k], "Wakeup Augpruned AP{}".format(validation_AP_tolerances[k]))

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(os.path.join(ap_log_augpruned_dir, "epoch{}_AP.png".format(epoch)))
    plt.close()

    print("Plotting time: {:.2f} seconds".format(time.time() - ctime))

def print_history(metrics_history):
    for key in metrics_history:
        print("{}      {}".format(key, metrics_history[key][-1]))

def update_SWA_bn(swa_model):
    # get previous momentum
    momenta = {}
    for module in swa_model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            momenta[module] = module.momentum

    if not momenta: # no batchnorm layers
        return

    # set to average over full batch
    was_training = swa_model.training
    swa_model.train()
    for module in momenta.keys():
        module.momentum = None

    # forward pass to update batchnorm
    training_sampler.shuffle()

    with torch.no_grad():
        with tqdm.tqdm(total=len(training_sampler)) as pbar:
            while training_sampler.entries_remaining() > 0:
                accel_data_batch, event_infos, times_batch, increment = \
                    training_sampler.sample(batch_size, random_shift=random_shift,
                                            random_flip=random_flip, random_vflip=flip_value, always_flip=always_flip,
                                            expand=expand, elastic_deformation=use_elastic_deformation,
                                            v_elastic_deformation=use_velastic_deformation,

                                            return_mode="expanded_interval_density")
                assert accel_data_batch.shape[
                           -1] == 2 * expand + prediction_length, "accel_data_batch.shape = {}".format(
                    accel_data_batch.shape)

                accel_data_batch_torch = torch.tensor(accel_data_batch, dtype=torch.float32, device=config.device)

                if use_anglez_only:
                    accel_data_batch_torch = accel_data_batch_torch[:, 0:1, :]
                elif use_enmo_only:
                    accel_data_batch_torch = accel_data_batch_torch[:, 1:2, :]

                # run forward pass
                times_input = times_batch if use_time_information else None
                swa_model(accel_data_batch_torch, time=times_input)

                pbar.update(increment)

    # recover momentum
    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    swa_model.train(was_training)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    all_data = convert_to_npy_naive.load_all_data_into_dict()

    parser = argparse.ArgumentParser(description="Train a sleeping prediction model with only clean data.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for. Default 50.")
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="Learning rate to use. Default 5e-3.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum to use. Default 0.9. This would be the momentum for SGD, and beta1 for Adam.")
    parser.add_argument("--second_momentum", type=float, default=0.999, help="Second momentum to use. Default 0.999. This would be beta2 for Adam. Ignored if SGD.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use. Default 0.0.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Which optimizer to use. Available options: adam, sgd. Default adam.")
    parser.add_argument("--epochs_per_save", type=int, default=1, help="Number of epochs between saves. Default 1.")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[1, 6, 8, 23, 8],
                        help="Number of hidden 2d blocks for ResNet backbone.")
    parser.add_argument("--hidden_channels", type=int, nargs="+", default=[2], help="Number of hidden channels. Default 2. Can be a list to specify num channels in each downsampled layer.")
    parser.add_argument("--bottleneck_factor", type=int, default=4, help="The bottleneck factor of the ResNet backbone. Default 4.")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for the first layer. Default 3.")
    parser.add_argument("--attention_blocks", type=int, default=4, help="Number of attention blocks to use. Default 4.")
    parser.add_argument("--upconv_channels_override", type=int, default=8, help="Number of fixed channels for the upsampling path. Default 8. If None, do not override.")
    parser.add_argument("--random_shift", type=int, default=0, help="Randomly shift the intervals by at most this amount. Default 0.")
    parser.add_argument("--random_flip", action="store_true", help="Randomly flip the intervals. Default False.")
    parser.add_argument("--always_flip", action="store_true", help="Always flip the intervals. Default False.")
    parser.add_argument("--flip_value", action="store_true", help="Whether to flip the value of the intervals. Default False.")
    parser.add_argument("--expand", type=int, default=8640, help="Expand the intervals by this amount. Default 8640.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate. Default 0.1.")
    parser.add_argument("--attn_dropout", type=float, default=0.1, help="Attention dropout rate. Default 0.1.")
    parser.add_argument("--use_time_information", action="store_true", help="Whether to use time information. Default False.")
    parser.add_argument("--use_elastic_deformation", action="store_true", help="Whether to use elastic deformation. Default False.")
    parser.add_argument("--use_velastic_deformation", action="store_true", help="Whether to use velastic deformation (only available for anglez). Default False.")
    parser.add_argument("--use_anglez_only", action="store_true", help="Whether to use only anglez. Default False.")
    parser.add_argument("--use_enmo_only", action="store_true", help="Whether to use only enmo. Default False.")
    parser.add_argument("--use_swa", action="store_true", help="Whether to use SWA. Default False.")
    parser.add_argument("--swa_start", type=int, default=10, help="Epoch to start SWA. Default 10.")
    parser.add_argument("--donot_exclude_bad_series_from_training", action="store_true", help="Whether to not exclude bad series from training. Default False.")
    parser.add_argument("--prediction_length", type=int, default=17280, help="Number of timesteps to predict. Default 17280.")
    parser.add_argument("--prediction_stride", type=int, default=4320, help="Number of timesteps to stride when predicting. Default 4320.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Default 32.")
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
    validation_entries = [series_id for series_id in validation_entries if series_id not in bad_series_list.noisy_bad_segmentations] # exclude

    # initialize gpu
    config.parse_args(args)

    # get model directories
    model_dir, prev_model_dir = manager_models.parse_args(args)

    # obtain model and training parameters
    epochs = args.epochs
    learning_rate = args.learning_rate
    momentum = args.momentum
    second_momentum = args.second_momentum
    weight_decay = args.weight_decay
    optimizer_type = args.optimizer
    epochs_per_save = args.epochs_per_save
    hidden_blocks = args.hidden_blocks
    hidden_channels = args.hidden_channels
    bottleneck_factor = args.bottleneck_factor
    kernel_size = args.kernel_size
    attention_blocks = args.attention_blocks
    upconv_channels_override = args.upconv_channels_override
    random_shift = args.random_shift
    random_flip = args.random_flip
    always_flip = args.always_flip
    flip_value = args.flip_value
    expand = args.expand
    dropout = args.dropout
    attn_dropout = args.attn_dropout
    use_time_information = args.use_time_information
    use_elastic_deformation = args.use_elastic_deformation
    use_velastic_deformation = args.use_velastic_deformation
    use_anglez_only = args.use_anglez_only
    use_enmo_only = args.use_enmo_only
    use_swa = args.use_swa
    swa_start = args.swa_start
    donot_exclude_bad_series_from_training = args.donot_exclude_bad_series_from_training
    prediction_length = args.prediction_length
    prediction_stride = args.prediction_stride
    batch_size = args.batch_size
    num_extra_steps = args.num_extra_steps

    assert sum([use_anglez_only, use_enmo_only]) <= 1, "Cannot use more than one of anglez only, enmo only"
    assert not (use_time_information and random_flip), "Cannot use time information and random flip at the same time."
    if not donot_exclude_bad_series_from_training:
        training_entries = [series_id for series_id in training_entries if series_id not in bad_series_list.noisy_bad_segmentations]

    assert os.path.isdir("./inference_regression_statistics/regression_preds/"), "Must generate regression predictions first. See inference_regression_statistics folder."
    per_series_id_events = convert_to_seriesid_events.get_events_per_seriesid()
    regression_predicted_events = convert_to_pred_events.load_all_pred_events_into_dict()
    dilation_converter = model_event_density_unet.ProbasDilationConverter(sigma=30 * 12, device=config.device)
    ap_log_dir = os.path.join(model_dir, "ap_log")
    ap_log_dilated_dir = os.path.join(model_dir, "ap_log_dilated")
    ap_log_aligned_dir = os.path.join(model_dir, "ap_log_aligned")
    ap_log_augpruned_dir = os.path.join(model_dir, "ap_log_augpruned")
    os.mkdir(ap_log_dir)
    os.mkdir(ap_log_dilated_dir)
    os.mkdir(ap_log_aligned_dir)
    os.mkdir(ap_log_augpruned_dir)

    if isinstance(hidden_channels, int):
        hidden_channels = [hidden_channels]
    if len(hidden_channels) == 1:
        chnls = hidden_channels[0]
        hidden_channels = []
        for k in range(len(hidden_blocks)):
            hidden_channels.append(chnls * (2 ** k))

    print("Epochs: " + str(epochs))
    print("Dropout: " + str(dropout))
    print("Attention dropout: " + str(attn_dropout))
    print("Bottleneck factor: " + str(bottleneck_factor))
    print("Hidden channels: " + str(hidden_channels))
    print("Hidden blocks: " + str(hidden_blocks))
    print("Kernel size: " + str(kernel_size))
    print("Upconv channels: " + str(upconv_channels_override))
    print("Use SWA: " + str(use_swa))
    model_unet.BATCH_NORM_MOMENTUM = 1 - momentum

    # initialize model
    in_channels = 1 if (use_anglez_only or use_enmo_only) else 2
    model = model_event_density_unet.EventDensityUnet(in_channels, hidden_channels, kernel_size=kernel_size, blocks=hidden_blocks,
                            bottleneck_factor=bottleneck_factor, dropout=dropout,
                            attention_blocks=attention_blocks,
                            upconv_channels_override=upconv_channels_override, attention_mode="length",
                            attention_dropout=attn_dropout,

                            use_time_input=use_time_information, training_strategy="density_only",
                            input_interval_length=prediction_length, input_expand_radius=expand)
    model = model.to(config.device)

    # initialize optimizer
    print("Loss: Day KL Divergence")
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

    # create SWA if necessary
    if use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=5 * learning_rate, anneal_strategy="linear", anneal_epochs=5)
        swa_start = args.swa_start

    model_config = {
        "model": "Unet with attention density",
        "training_dataset": train_dset_name,
        "validation_dataset": val_dset_name,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "second_momentum": second_momentum,
        "weight_decay": weight_decay,
        "optimizer": optimizer_type,
        "epochs_per_save": epochs_per_save,
        "hidden_blocks": hidden_blocks,
        "hidden_channels": hidden_channels,
        "bottleneck_factor": bottleneck_factor,
        "kernel_size": kernel_size,
        "attention_blocks": attention_blocks,
        "upconv_channels_override": upconv_channels_override,
        "random_shift": random_shift,
        "random_flip": random_flip,
        "always_flip": always_flip,
        "flip_value": flip_value,
        "expand": expand,
        "dropout": dropout,
        "attn_dropout": attn_dropout,
        "use_time_information": use_time_information,
        "use_elastic_deformation": use_elastic_deformation,
        "use_velastic_deformation": use_velastic_deformation,
        "use_anglez_only": use_anglez_only,
        "use_enmo_only": use_enmo_only,
        "use_swa": use_swa,
        "swa_start": swa_start,
        "prediction_length": prediction_length,
        "prediction_stride": prediction_stride,
        "batch_size": batch_size,
        "num_extra_steps": num_extra_steps
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
    train_metrics["class_loss"] = metrics.NumericalMetric("train_class_loss")
    train_metrics["class_metric"] = metrics.BinaryMetricsTPRFPR("train_class_metric")
    train_metrics["entropy_loss"] = metrics.NumericalMetric("train_entropy_loss")
    val_metrics["loss"] = metrics.NumericalMetric("val_loss")
    val_metrics["class_loss"] = metrics.NumericalMetric("val_class_loss")
    val_metrics["class_metric"] = metrics.BinaryMetricsTPRFPR("val_class_metric")
    val_metrics["entropy_loss"] = metrics.NumericalMetric("val_entropy_loss")

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
    print("Use anglez only: " + str(use_anglez_only))
    print("Use enmo only: " + str(use_enmo_only))
    training_sampler = convert_to_interval_density_events.IntervalDensityEventsSampler(training_entries, all_data,
                                                                        train_or_test="train",
                                                                        prediction_length=prediction_length,
                                                                        prediction_stride=prediction_stride,
                                                                        is_enmo_only=use_enmo_only)
    val_sampler = convert_to_interval_density_events.IntervalDensityEventsSampler(validation_entries, all_data,
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

            # manage SWA
            if use_swa and (epoch > swa_start):
                swa_model.update_parameters(model)
                swa_scheduler.step()
                if ((epoch - swa_start) % 3 == 0) and not args.train_all:
                    update_SWA_bn(swa_model)

            # switch model to eval mode
            model.eval()
            if use_swa and (epoch > swa_start):
                swa_model.eval()
            with torch.no_grad():
                validation_step()
                validation_ap(epoch=epoch, ap_log_dir=ap_log_dir, ap_log_dilated_dir=ap_log_dilated_dir,
                              ap_log_aligned_dir=ap_log_aligned_dir, ap_log_augpruned_dir=ap_log_augpruned_dir,
                              predicted_events=regression_predicted_events, gt_events=per_series_id_events)

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
                if use_swa and (epoch > swa_start):
                    torch.save(swa_model.state_dict(), os.path.join(model_dir, "swa_model_{}.pt".format(epoch)))
                    torch.save(swa_scheduler.state_dict(), os.path.join(model_dir, "swa_scheduler_{}.pt".format(epoch)))

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
    if use_swa:
        update_SWA_bn(swa_model)
        torch.save(swa_model.state_dict(), os.path.join(model_dir, "swa_model.pt"))
        torch.save(swa_scheduler.state_dict(), os.path.join(model_dir, "swa_scheduler.pt"))

    # save metrics
    if len(train_history) > 0:
        train_df = pd.DataFrame(train_history)
        val_df = pd.DataFrame(val_history)
        train_df.to_csv(os.path.join(model_dir, "train_metrics.csv"), index=True)
        val_df.to_csv(os.path.join(model_dir, "val_metrics.csv"), index=True)

    memory_logger.close()