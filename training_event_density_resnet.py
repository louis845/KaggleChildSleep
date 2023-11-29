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

def focal_loss(preds: torch.Tensor, ground_truth: torch.Tensor):
    assert preds.shape == ground_truth.shape, "preds.shape = {}, ground_truth.shape = {}".format(preds.shape, ground_truth.shape)

    ce = torch.nn.functional.binary_cross_entropy_with_logits(preds, ground_truth, reduction="none")
    preds_probas = torch.sigmoid(preds)

    return (((preds_probas - ground_truth) ** 2) * ce).mean(dim=-1).sum()


def single_training_step(model_: torch.nn.Module, optimizer_: torch.optim.Optimizer,
                            accel_data_batch: torch.Tensor, labels_density_batch: torch.Tensor,
                            labels_occurrence_batch: torch.Tensor,
                            labels_segmentation_batch: torch.Tensor,
                            times_batch: np.ndarray):
    optimizer_.zero_grad()
    times_input = times_batch if use_time_information else None
    if use_center_softmax:
        pred_density_logits, pred_occurences, pred_token_confidence = model_(accel_data_batch, time=times_input)
        deep_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_token_confidence, labels_segmentation_batch, reduction="none").mean(dim=-1).sum()
    else:
        pred_density_logits, pred_occurences = model_(accel_data_batch, time=times_input)
    entropy_loss = torch.nn.functional.cross_entropy(pred_density_logits, labels_density_batch, reduction="none").mean(dim=-1).sum()
    if use_focal_loss:
        class_loss = focal_loss(pred_occurences, labels_occurrence_batch)
    else:
        class_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_occurences, labels_occurrence_batch, reduction="none").mean(dim=-1).sum()
    if use_center_softmax:
        loss = entropy_loss + class_loss + deep_loss
        deep_loss_ret = deep_loss.item()
        with torch.no_grad():
            pred_token_confidence_ret = (pred_token_confidence > 0.0).to(torch.long)
    else:
        loss = entropy_loss + class_loss
        deep_loss_ret = None
        pred_token_confidence_ret = None

    loss.backward()
    optimizer_.step()

    with torch.no_grad():
        pred_occurences = pred_occurences > 0.0

    return loss.item(), class_loss.item(), deep_loss_ret, pred_occurences.to(torch.long), pred_token_confidence_ret, entropy_loss.item()

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

                                        return_mode="interval_density_and_expanded_events" if use_center_softmax else "expanded_interval_density")
            labels_density_batch = event_infos["density"]
            labels_occurrence_batch = event_infos["occurrence"]
            if use_center_softmax:
                assert labels_density_batch.shape[-1] == prediction_length, "labels_batch.shape = {}".format(labels_density_batch.shape)
            else:
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

            if use_center_softmax:
                labels_segmentation_batch = event_infos["segmentation"]
                assert labels_segmentation_batch.shape[-1] == (2 * expand + prediction_length) // model.input_length_multiple, "labels_segmentation_batch.shape = {}".format(labels_segmentation_batch.shape)
                assert labels_segmentation_batch.shape[-2] == 2, "labels_segmentation_batch.shape = {}".format(labels_segmentation_batch.shape)
                labels_segmentation_batch_torch = torch.tensor(labels_segmentation_batch, dtype=torch.float32, device=config.device)
            else:
                labels_segmentation_batch_torch = None

            # train model now
            loss, class_loss, deep_loss, pred_occurences, pred_token_confidence, entropy_loss =\
                single_training_step(model, optimizer,
                                       accel_data_batch_torch,
                                       labels_density_batch_torch,
                                       labels_occurrence_batch_torch,
                                       labels_segmentation_batch_torch,
                                       times_batch)
            #time.sleep(0.2)

            # record
            if record:
                with torch.no_grad():
                    labels_occurrence_long = (labels_occurrence_batch_torch > 0.5).to(torch.long)
                    train_metrics["loss"].add(loss, increment)
                    train_metrics["class_loss"].add(class_loss, increment)
                    train_metrics["class_metric"].add(pred_occurences, labels_occurrence_long)
                    train_metrics["entropy_loss"].add(entropy_loss, increment)
                    if use_center_softmax:
                        train_metrics["deep_loss"].add(deep_loss, increment)
                        train_metrics["deep_metric"].add(pred_token_confidence, labels_segmentation_batch_torch.to(torch.long))

            pbar.update(increment)

    if record:
        current_metrics = {}
        for key in train_metrics:
            train_metrics[key].write_to_dict(current_metrics)

        for key in current_metrics:
            train_history[key].append(current_metrics[key])

def single_validation_step(model_: torch.nn.Module, accel_data_batch: torch.Tensor,
                           labels_density_batch: torch.Tensor, labels_occurrence_batch: torch.Tensor,
                           labels_segmentation_batch: torch.Tensor, times_batch: np.ndarray):
    with torch.no_grad():
        times_input = times_batch if use_time_information else None
        if use_center_softmax:
            pred_density_logits, pred_occurences, pred_token_confidence = model_(accel_data_batch, time=times_input, return_as_training=True)
            deep_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_token_confidence, labels_segmentation_batch, reduction="none").mean(dim=-1).sum()
            pred_token_confidence = (pred_token_confidence > 0.0).to(torch.long)
        else:
            pred_density_logits, pred_occurences = model_(accel_data_batch, time=times_input, return_as_training=True)
            deep_loss = None
            pred_token_confidence = None

        entropy_loss = torch.nn.functional.cross_entropy(pred_density_logits, labels_density_batch,
                                                         reduction="none").mean(dim=-1).sum()
        class_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_occurences, labels_occurrence_batch,
                                                                          reduction="none").mean(dim=-1).sum()
        if use_center_softmax:
            loss = entropy_loss + class_loss + deep_loss
            deep_loss = deep_loss.item()
        else:
            loss = entropy_loss + class_loss
        pred_occurences = pred_occurences > 0.0
        return loss.item(), class_loss.item(), deep_loss, pred_occurences.to(torch.long), pred_token_confidence, entropy_loss.item()

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
                return_mode="interval_density_and_expanded_events" if use_center_softmax else "expanded_interval_density")
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

            if use_center_softmax:
                labels_segmentation_batch = event_infos["segmentation"]
                labels_segmentation_batch = torch.tensor(labels_segmentation_batch, dtype=torch.float32,
                                                               device=config.device)
            else:
                labels_segmentation_batch = None

            # val model now
            loss, class_loss, deep_loss, pred_occurences, pred_token_confidence, entropy_loss =\
                single_validation_step(use_model, accel_data_batch,
                                             labels_density_batch,
                                             labels_occurrence_batch,
                                             labels_segmentation_batch,
                                             times_batch)

            # record
            with torch.no_grad():
                labels_occurrence_long = (labels_occurrence_batch > 0.5).to(torch.long)
                val_metrics["loss"].add(loss, increment)
                val_metrics["class_loss"].add(class_loss, increment)
                val_metrics["class_metric"].add(pred_occurences, labels_occurrence_long)
                val_metrics["entropy_loss"].add(entropy_loss, increment)
                if use_center_softmax:
                    val_metrics["deep_loss"].add(deep_loss, increment)
                    val_metrics["deep_metric"].add(pred_token_confidence, labels_segmentation_batch.to(torch.long))

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
def validation_ap(epoch, ap_log_dir, ap_log_aligned_dir, ap_log_loc_softmax_dir,
                  ap_dense_log_dir, ap_dense_log_aligned_dir, ap_dense_log_loc_softmax_dir,
                  ap_very_dense_log_dir, ap_very_dense_log_aligned_dir, ap_very_dense_log_loc_softmax_dir,
                  predicted_events, dense_predicted_events, very_dense_predicted_events, gt_events):
    use_model = swa_model if (use_swa and (epoch > swa_start)) else model

    keys = ["usual", "aligned", "loc_softmax", "dense_usual", "dense_aligned", "dense_loc_softmax",
                "very_dense_usual", "very_dense_aligned", "very_dense_loc_softmax"]

    dirs = {"usual": ap_log_dir, "aligned": ap_log_aligned_dir, "loc_softmax": ap_log_loc_softmax_dir,
            "dense_usual": ap_dense_log_dir, "dense_aligned": ap_dense_log_aligned_dir, "dense_loc_softmax": ap_dense_log_loc_softmax_dir,
            "very_dense_usual": ap_very_dense_log_dir, "very_dense_aligned": ap_very_dense_log_aligned_dir, "very_dense_loc_softmax": ap_very_dense_log_loc_softmax_dir}
    all_ap_onset_metrics = {key: [metrics_ap.EventMetrics(name="", tolerance=tolerance * 12) for tolerance in validation_AP_tolerances]
                             for key in keys}
    all_ap_wakeup_metrics = {key: [metrics_ap.EventMetrics(name="", tolerance=tolerance * 12) for tolerance in validation_AP_tolerances]
                              for key in keys}

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

            # load the regression predictions
            preds_locs = predicted_events[series_id]
            preds_locs_onset = preds_locs["onset"]
            preds_locs_wakeup = preds_locs["wakeup"]

            dense_preds_locs = dense_predicted_events[series_id]
            dense_preds_locs_onset = dense_preds_locs["onset"]
            dense_preds_locs_wakeup = dense_preds_locs["wakeup"]

            very_dense_preds_locs = very_dense_predicted_events[series_id]
            very_dense_preds_locs_onset = very_dense_preds_locs["onset"]
            very_dense_preds_locs_wakeup = very_dense_preds_locs["wakeup"]

            # run inference
            stride_count = 8
            if use_swa and (epoch > swa_start) and ((epoch - swa_start) % 3 != 0):
                stride_count = 1
            proba_preds, all_onset_IOU_loc_softmax_probas, all_wakeup_IOU_loc_softmax_probas =\
                model_event_density_unet.event_density_inference(model=use_model, time_series=accel_data,
                                                                      predicted_locations=[{
                                                                          "onset": preds_locs_onset,
                                                                          "wakeup": preds_locs_wakeup
                                                                      },
                                                                      {
                                                                          "onset": dense_preds_locs_onset,
                                                                          "wakeup": dense_preds_locs_wakeup
                                                                      },
                                                                      {
                                                                          "onset": very_dense_preds_locs_onset,
                                                                          "wakeup": very_dense_preds_locs_wakeup
                                                                      }],
                                                                      batch_size=batch_size * 5,
                                                                      prediction_length=prediction_length,
                                                                      expand=expand, times=times,
                                                                      device=config.device,
                                                                      use_time_input=use_time_information,
                                                                      stride_count=stride_count)
            onset_IOU_probas = proba_preds[0, :]
            wakeup_IOU_probas = proba_preds[1, :]

            # restrict
            onset_IOU_usual_probas = onset_IOU_probas[preds_locs_onset]
            wakeup_IOU_usual_probas = wakeup_IOU_probas[preds_locs_wakeup]
            onset_IOU_aligned_probas = postprocessing.align_index_probas(preds_locs_onset, onset_IOU_probas)
            wakeup_IOU_aligned_probas = postprocessing.align_index_probas(preds_locs_wakeup, wakeup_IOU_probas)
            onset_IOU_loc_softmax_probas = all_onset_IOU_loc_softmax_probas[0]
            wakeup_IOU_loc_softmax_probas = all_wakeup_IOU_loc_softmax_probas[0]

            onset_IOU_dense_usual_probas = onset_IOU_probas[dense_preds_locs_onset]
            wakeup_IOU_dense_usual_probas = wakeup_IOU_probas[dense_preds_locs_wakeup]
            onset_IOU_dense_aligned_probas = postprocessing.align_index_probas(dense_preds_locs_onset, onset_IOU_probas)
            wakeup_IOU_dense_aligned_probas = postprocessing.align_index_probas(dense_preds_locs_wakeup, wakeup_IOU_probas)
            onset_IOU_dense_loc_softmax_probas = all_onset_IOU_loc_softmax_probas[1]
            wakeup_IOU_dense_loc_softmax_probas = all_wakeup_IOU_loc_softmax_probas[1]

            onset_IOU_very_dense_usual_probas = onset_IOU_probas[very_dense_preds_locs_onset]
            wakeup_IOU_very_dense_usual_probas = wakeup_IOU_probas[very_dense_preds_locs_wakeup]
            onset_IOU_very_dense_aligned_probas = postprocessing.align_index_probas(very_dense_preds_locs_onset, onset_IOU_probas)
            wakeup_IOU_very_dense_aligned_probas = postprocessing.align_index_probas(very_dense_preds_locs_wakeup, wakeup_IOU_probas)
            onset_IOU_very_dense_loc_softmax_probas = all_onset_IOU_loc_softmax_probas[2]
            wakeup_IOU_very_dense_loc_softmax_probas = all_wakeup_IOU_loc_softmax_probas[2]

            onset_all_probas = {"usual": onset_IOU_usual_probas, "aligned": onset_IOU_aligned_probas, "loc_softmax": onset_IOU_loc_softmax_probas,
                                "dense_usual": onset_IOU_dense_usual_probas, "dense_aligned": onset_IOU_dense_aligned_probas, "dense_loc_softmax": onset_IOU_dense_loc_softmax_probas,
                                "very_dense_usual": onset_IOU_very_dense_usual_probas, "very_dense_aligned": onset_IOU_very_dense_aligned_probas, "very_dense_loc_softmax": onset_IOU_very_dense_loc_softmax_probas}
            wakeup_all_probas = {"usual": wakeup_IOU_usual_probas, "aligned": wakeup_IOU_aligned_probas, "loc_softmax": wakeup_IOU_loc_softmax_probas,
                                    "dense_usual": wakeup_IOU_dense_usual_probas, "dense_aligned": wakeup_IOU_dense_aligned_probas, "dense_loc_softmax": wakeup_IOU_dense_loc_softmax_probas,
                                    "very_dense_usual": wakeup_IOU_very_dense_usual_probas, "very_dense_aligned": wakeup_IOU_very_dense_aligned_probas, "very_dense_loc_softmax": wakeup_IOU_very_dense_loc_softmax_probas}

            # get the ground truth
            gt_onset_locs = gt_events[series_id]["onset"]
            gt_wakeup_locs = gt_events[series_id]["wakeup"]

            # add info
            for key in keys:
                for key_onset_metric, key_wakeup_metric in zip(all_ap_onset_metrics[key], all_ap_wakeup_metrics[key]):
                    if "very_dense" in key:
                        key_onset_metric.add(pred_locs=very_dense_preds_locs_onset, pred_probas=onset_all_probas[key], gt_locs=gt_onset_locs)
                        key_wakeup_metric.add(pred_locs=very_dense_preds_locs_wakeup, pred_probas=wakeup_all_probas[key], gt_locs=gt_wakeup_locs)
                    elif "dense" in key:
                        key_onset_metric.add(pred_locs=dense_preds_locs_onset, pred_probas=onset_all_probas[key], gt_locs=gt_onset_locs)
                        key_wakeup_metric.add(pred_locs=dense_preds_locs_wakeup, pred_probas=wakeup_all_probas[key], gt_locs=gt_wakeup_locs)
                    else:
                        key_onset_metric.add(pred_locs=preds_locs_onset, pred_probas=onset_all_probas[key], gt_locs=gt_onset_locs)
                        key_wakeup_metric.add(pred_locs=preds_locs_wakeup, pred_probas=wakeup_all_probas[key], gt_locs=gt_wakeup_locs)

    # compute average precision
    ctime = time.time()
    all_ap_onset_precisions, all_ap_onset_recalls, all_ap_onset_average_precisions = {key: [] for key in keys}, {key: [] for key in keys}, {key: [] for key in keys}
    all_ap_wakeup_precisions, all_ap_wakeup_recalls, all_ap_wakeup_average_precisions = {key: [] for key in keys}, {key: [] for key in keys}, {key: [] for key in keys}

    for key in keys:
        for ap_onset_metric, ap_wakeup_metric in zip(all_ap_onset_metrics[key], all_ap_wakeup_metrics[key]):
            ap_onset_precision, ap_onset_recall, ap_onset_average_precision, _ = ap_onset_metric.get()
            ap_wakeup_precision, ap_wakeup_recall, ap_wakeup_average_precision, _ = ap_wakeup_metric.get()
            all_ap_onset_precisions[key].append(ap_onset_precision)
            all_ap_onset_recalls[key].append(ap_onset_recall)
            all_ap_onset_average_precisions[key].append(ap_onset_average_precision)
            all_ap_wakeup_precisions[key].append(ap_wakeup_precision)
            all_ap_wakeup_recalls[key].append(ap_wakeup_recall)
            all_ap_wakeup_average_precisions[key].append(ap_wakeup_average_precision)
    print("AP computation time: {:.2f} seconds".format(time.time() - ctime))

    # write to dict
    ctime = time.time()
    for key in keys:
        val_history["val_onset_{}_mAP".format(key)].append(np.mean(all_ap_onset_average_precisions[key]))
        val_history["val_wakeup_{}_mAP".format(key)].append(np.mean(all_ap_wakeup_average_precisions[key]))

    # draw the precision-recall curve using matplotlib onto file "epoch{}_AP.png".format(epoch) inside the ap_log_dir(s)
    for key in keys:
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        fig.suptitle("Epoch {} (Onset mAP: {}, Wakeup mAP: {})".format(epoch, np.mean(all_ap_onset_average_precisions[key]), np.mean(all_ap_wakeup_average_precisions[key])))
        for k in range(len(validation_AP_tolerances)):
            ax = axes[k // 5, k % 5]
            plot_single_precision_recall_curve(ax, all_ap_onset_precisions[key][k], all_ap_onset_recalls[key][k], all_ap_onset_average_precisions[key][k], "Onset AP{}".format(validation_AP_tolerances[k]))
            ax = axes[(k + 10) // 5, (k + 10) % 5]
            plot_single_precision_recall_curve(ax, all_ap_wakeup_precisions[key][k], all_ap_wakeup_recalls[key][k], all_ap_wakeup_average_precisions[key][k], "Wakeup AP{}".format(validation_AP_tolerances[k]))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.savefig(os.path.join(dirs[key], "epoch{}_AP.png".format(epoch)))
        plt.close()

    print("Plotting time: {:.2f} seconds".format(time.time() - ctime))

def print_history(metrics_history):
    for key in metrics_history:
        print("{}      {}".format(key, metrics_history[key][-1]))

def update_SWA_bn(swa_model): # modified from PyTorch swa_utils source code.
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

                                            return_mode="interval_density_and_expanded_events" if use_center_softmax else "expanded_interval_density")
                assert accel_data_batch.shape[-1] == 2 * expand + prediction_length, "accel_data_batch.shape = {}".format(accel_data_batch.shape)

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
    parser.add_argument("--use_focal_loss", action="store_true", help="Whether to use focal loss. Default False.")
    parser.add_argument("--use_time_information", action="store_true", help="Whether to use time information. Default False.")
    parser.add_argument("--use_elastic_deformation", action="store_true", help="Whether to use elastic deformation. Default False.")
    parser.add_argument("--use_velastic_deformation", action="store_true", help="Whether to use velastic deformation (only available for anglez). Default False.")
    parser.add_argument("--use_anglez_only", action="store_true", help="Whether to use only anglez. Default False.")
    parser.add_argument("--use_enmo_only", action="store_true", help="Whether to use only enmo. Default False.")
    parser.add_argument("--use_center_softmax", action="store_true", help="Whether to train the model by predicting the softmax only in the center. Default False.")
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
    #validation_entries = [series_id for series_id in validation_entries if series_id not in bad_series_list.noisy_bad_segmentations] # exclude

    # initialize gpu
    config.parse_args(args)

    # get model directories
    model_dir, prev_model_dir, load_model_epoch = manager_models.parse_args(args)

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
    use_focal_loss = args.use_focal_loss
    use_time_information = args.use_time_information
    use_elastic_deformation = args.use_elastic_deformation
    use_velastic_deformation = args.use_velastic_deformation
    use_anglez_only = args.use_anglez_only
    use_enmo_only = args.use_enmo_only
    use_center_softmax = args.use_center_softmax
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
    assert os.path.isdir("./inference_regression_statistics/regression_preds_dense/"), "Must generate regression predictions first. See inference_regression_statistics folder."
    assert os.path.isdir("./inference_regression_statistics/regression_preds_very_dense/"), "Must generate regression predictions first. See inference_regression_statistics folder."
    per_series_id_events = convert_to_seriesid_events.get_events_per_seriesid()
    regression_predicted_events = convert_to_pred_events.load_all_pred_events_into_dict("regression_preds")
    regression_dense_predicted_events = convert_to_pred_events.load_all_pred_events_into_dict("regression_preds_dense")
    regression_very_dense_predicted_events = convert_to_pred_events.load_all_pred_events_into_dict("regression_preds_very_dense")
    dilation_converter = model_event_density_unet.ProbasDilationConverter(sigma=30 * 12, device=config.device)
    ap_log_dir = os.path.join(model_dir, "ap_log")
    ap_log_aligned_dir = os.path.join(model_dir, "ap_log_aligned")
    ap_log_loc_softmax_dir = os.path.join(model_dir, "ap_log_loc_softmax")
    ap_dense_log_dir = os.path.join(model_dir, "ap_dense_log")
    ap_dense_log_aligned_dir = os.path.join(model_dir, "ap_dense_log_aligned")
    ap_dense_log_loc_softmax_dir = os.path.join(model_dir, "ap_dense_log_loc_softmax")
    ap_very_dense_log_dir = os.path.join(model_dir, "ap_very_dense_log")
    ap_very_dense_log_aligned_dir = os.path.join(model_dir, "ap_very_dense_log_aligned")
    ap_very_dense_log_loc_softmax_dir = os.path.join(model_dir, "ap_very_dense_log_loc_softmax")
    os.mkdir(ap_log_dir)
    os.mkdir(ap_log_aligned_dir)
    os.mkdir(ap_log_loc_softmax_dir)
    os.mkdir(ap_dense_log_dir)
    os.mkdir(ap_dense_log_aligned_dir)
    os.mkdir(ap_dense_log_loc_softmax_dir)
    os.mkdir(ap_very_dense_log_dir)
    os.mkdir(ap_very_dense_log_aligned_dir)
    os.mkdir(ap_very_dense_log_loc_softmax_dir)

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

                            use_time_input=use_time_information, training_strategy="density_and_confidence" if use_center_softmax else "density_only",
                            input_interval_length=prediction_length, input_expand_radius=expand)
    model = model.to(config.device)

    # initialize optimizer
    print("Loss: Day KL Divergence (Focal)" if use_focal_loss else "Loss: Day KL Divergence")
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
        if load_model_epoch == -1:
            print("Loading previous model (final).")
            model_checkpoint_path = os.path.join(prev_model_dir, "model.pt")
            optimizer_checkpoint_path = os.path.join(prev_model_dir, "optimizer.pt")
        else:
            print("Loading previous model (epoch {}).".format(load_model_epoch))
            model_checkpoint_path = os.path.join(prev_model_dir, "model_{}.pt".format(load_model_epoch))
            optimizer_checkpoint_path = os.path.join(prev_model_dir, "optimizer_{}.pt".format(load_model_epoch))

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
        "use_focal_loss": use_focal_loss,
        "use_time_information": use_time_information,
        "use_elastic_deformation": use_elastic_deformation,
        "use_velastic_deformation": use_velastic_deformation,
        "use_anglez_only": use_anglez_only,
        "use_enmo_only": use_enmo_only,
        "use_center_softmax": use_center_softmax,
        "use_swa": use_swa,
        "swa_start": swa_start,
        "donot_exclude_bad_series_from_training": donot_exclude_bad_series_from_training,
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
    if use_center_softmax:
        train_metrics["deep_loss"] = metrics.NumericalMetric("train_deep_loss")
        train_metrics["deep_metric"] = metrics.BinaryMetrics("train_deep_metric")
    train_metrics["entropy_loss"] = metrics.NumericalMetric("train_entropy_loss")
    val_metrics["loss"] = metrics.NumericalMetric("val_loss")
    val_metrics["class_loss"] = metrics.NumericalMetric("val_class_loss")
    val_metrics["class_metric"] = metrics.BinaryMetricsTPRFPR("val_class_metric")
    if use_center_softmax:
        val_metrics["deep_loss"] = metrics.NumericalMetric("val_deep_loss")
        val_metrics["deep_metric"] = metrics.BinaryMetrics("val_deep_metric")
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
                                                                        input_length_multiple=model.input_length_multiple,
                                                                        train_or_test="train",
                                                                        prediction_length=prediction_length,
                                                                        prediction_stride=prediction_stride,
                                                                        is_enmo_only=use_enmo_only,
                                                                        donot_exclude_bad_series_from_training=donot_exclude_bad_series_from_training)
    val_sampler = convert_to_interval_density_events.IntervalDensityEventsSampler(validation_entries, all_data,
                                                                        input_length_multiple=model.input_length_multiple,
                                                                        train_or_test="val",
                                                                        prediction_length=prediction_length,
                                                                        prediction_stride=prediction_stride,
                                                                        donot_exclude_bad_series_from_training=True)


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
                validation_ap(epoch=epoch, ap_log_dir=ap_log_dir,  ap_log_aligned_dir=ap_log_aligned_dir, ap_log_loc_softmax_dir=ap_log_loc_softmax_dir,
                              ap_dense_log_dir=ap_dense_log_dir, ap_dense_log_aligned_dir=ap_dense_log_aligned_dir,
                              ap_dense_log_loc_softmax_dir=ap_dense_log_loc_softmax_dir,
                              ap_very_dense_log_dir=ap_very_dense_log_dir, ap_very_dense_log_aligned_dir=ap_very_dense_log_aligned_dir,
                              ap_very_dense_log_loc_softmax_dir=ap_very_dense_log_loc_softmax_dir,
                              predicted_events=regression_predicted_events,
                              dense_predicted_events=regression_dense_predicted_events,
                              very_dense_predicted_events=regression_very_dense_predicted_events,
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

    memory_logger.close()

    # save metrics
    if len(train_history) > 0:
        train_df = pd.DataFrame(train_history)
        val_df = pd.DataFrame(val_history)
        train_df.to_csv(os.path.join(model_dir, "train_metrics.csv"), index=True)
        val_df.to_csv(os.path.join(model_dir, "val_metrics.csv"), index=True)