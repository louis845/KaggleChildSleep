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
import convert_to_npy_naive
import convert_to_regression_events
import model_event_unet
import model_unet
import model_attention_unet
import kernel_utils

def masked_ce_loss(preds: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor = None):
    assert preds.shape == ground_truth.shape, "preds.shape = {}, ground_truth.shape = {}".format(preds.shape,
                                                                                                 ground_truth.shape)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(preds, ground_truth, reduction="none")
    mask_per_batch = torch.sum(mask, dim=(1, 2))
    loss_per_batch = torch.where(mask_per_batch > 0, torch.sum(bce * mask, dim=(1, 2)) / (mask_per_batch + 1e-7),
                                 torch.zeros_like(mask_per_batch))
    return torch.sum(loss_per_batch)

def masked_focal_loss(preds: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor = None):
    assert preds.shape == ground_truth.shape, "preds.shape = {}, ground_truth.shape = {}".format(preds.shape, ground_truth.shape)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(preds, ground_truth, reduction="none")
    focal_bce = ((torch.sigmoid(preds) - ground_truth) ** 2) * bce
    mask_per_batch = torch.sum(mask, dim=(1, 2))
    loss_per_batch = torch.where(mask_per_batch > 0, torch.sum(focal_bce * mask, dim=(1, 2)) / (mask_per_batch + 1e-7),
                                 torch.zeros_like(mask_per_batch))
    return torch.sum(loss_per_batch)

def masked_huber_mse_loss(preds: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor):
    assert preds.shape == ground_truth.shape, "preds.shape = {}, ground_truth.shape = {}".format(preds.shape,
                                                                                                 ground_truth.shape)
    assert preds.shape == mask.shape, "preds.shape = {}, mask.shape = {}".format(preds.shape, mask.shape)
    mask_per_batch = torch.sum(mask, dim=(1, 2))
    mse_per_batch = torch.sum(torch.nn.functional.mse_loss(preds, ground_truth, reduction="none") * mask, dim=(1, 2)) / (mask_per_batch + 1e-7)
    huber_mse_per_batch = torch.where(mse_per_batch < 1, 0.5 * mse_per_batch, torch.sqrt(mse_per_batch) - 0.5)
    loss_per_batch = torch.where(mask_per_batch > 0, huber_mse_per_batch, torch.zeros_like(mask_per_batch))
    return torch.sum(loss_per_batch)

def masked_huber_loss(preds: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor):
    assert preds.shape == ground_truth.shape, "preds.shape = {}, ground_truth.shape = {}".format(preds.shape,
                                                                                                 ground_truth.shape)
    assert preds.shape == mask.shape, "preds.shape = {}, mask.shape = {}".format(preds.shape, mask.shape)
    mask_per_batch = torch.sum(mask, dim=(1, 2))
    loss_per_batch = torch.where(mask_per_batch > 0, torch.sum(torch.nn.functional.huber_loss(preds, ground_truth, reduction="none") * mask, dim=(1, 2)) / (mask_per_batch + 1e-5),
                                    torch.zeros_like(mask_per_batch))
    return torch.sum(loss_per_batch)

def masked_huber_sigma_loss(preds: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor):
    assert preds.shape[1] == 4, "preds.shape = {}".format(preds.shape)
    assert (preds.shape[0] == ground_truth.shape[0] and preds.shape[2] == ground_truth.shape[2]), "preds.shape = {}, ground_truth.shape = {}".format(preds.shape, ground_truth.shape)
    assert ground_truth.shape == mask.shape, "ground_truth.shape = {}, mask.shape = {}".format(ground_truth.shape, mask.shape)
    mask_per_batch = torch.sum(mask, dim=(1, 2))

    values = preds[:, :2, :]
    assert values.shape == mask.shape, "values.shape = {}, mask.shape = {}".format(values.shape, mask.shape)
    sigmas = preds[:, 2:, :]
    k = sigmas / np.sqrt(2)
    elemwise_loss = (torch.nn.functional.huber_loss(values, ground_truth, reduction="none") + 0.5) / k + torch.log(k)

    loss_per_batch = torch.where(mask_per_batch > 0, torch.sum(elemwise_loss * mask, dim=(1, 2)) / (mask_per_batch + 1e-5),
                                    torch.zeros_like(mask_per_batch))
    return torch.sum(loss_per_batch)

def masked_mse_loss(preds: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor):
    assert preds.shape == ground_truth.shape, "preds.shape = {}, ground_truth.shape = {}".format(preds.shape,
                                                                                                 ground_truth.shape)
    assert preds.shape == mask.shape, "preds.shape = {}, mask.shape = {}".format(preds.shape, mask.shape)
    mask_per_batch = torch.sum(mask, dim=(1, 2))
    loss_per_batch = torch.where(mask_per_batch > 0, torch.sum(torch.nn.functional.mse_loss(preds, ground_truth, reduction="none") * mask, dim=(1, 2)) / (mask_per_batch + 1e-7),
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

def masked_mae_loss_raw(preds: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor):
    assert preds.shape == ground_truth.shape, "preds.shape = {}, ground_truth.shape = {}".format(preds.shape,
                                                                                                 ground_truth.shape)
    assert preds.shape == mask.shape, "preds.shape = {}, mask.shape = {}".format(preds.shape, mask.shape)
    loss = torch.sum(torch.abs(preds - ground_truth) * mask)
    return loss

def masked_ce_loss_raw(preds: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor):
    assert preds.shape == ground_truth.shape, "preds.shape = {}, ground_truth.shape = {}".format(preds.shape,
                                                                                                 ground_truth.shape)
    assert preds.shape == mask.shape, "preds.shape = {}, mask.shape = {}".format(preds.shape, mask.shape)
    loss = torch.sum(torch.nn.functional.binary_cross_entropy_with_logits(preds, ground_truth, reduction="none") * mask)
    return loss

def masked_focal_loss_raw(preds: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor):
    assert preds.shape == ground_truth.shape, "preds.shape = {}, ground_truth.shape = {}".format(preds.shape,
                                                                                                 ground_truth.shape)
    assert preds.shape == mask.shape, "preds.shape = {}, mask.shape = {}".format(preds.shape, mask.shape)
    loss = torch.sum(((torch.sigmoid(preds) - ground_truth) ** 2) *
                     torch.nn.functional.binary_cross_entropy_with_logits(preds, ground_truth, reduction="none") * mask)
    return loss

def local_median_ae_loss(preds: np.ndarray, event: int, width: int):
    low = max(0, event - width)
    high = min(preds.shape[0], event + width + 1)

    width = high - low
    intervals = [(low, high),
                 (low, low + int(width * 0.8)),
                 (low + int(width * 0.2), high),
                 (low, low + int(width * 0.6)),
                 (low + int(width * 0.2), low + int(width * 0.8)),
                 (low + int(width * 0.4), high),
                 (low, low + int(width * 0.4)),
                 (low + int(width * 0.2), low + int(width * 0.6)),
                 (low + int(width * 0.4), low + int(width * 0.8)),
                 (low + int(width * 0.6), high)]
    return np.max(
        [np.abs(np.mean(np.arange(low, high) - preds[low:high]) - event) for low, high in intervals]
    )

def local_mean_ae_loss(preds: np.ndarray, event: int, width: int):
    low = max(0, event - width)
    high = min(preds.shape[0], event + width + 1)

    width = high - low
    intervals = [(low, high),
                 (low, low + int(width * 0.8)),
                 (low + int(width * 0.2), high),
                 (low, low + int(width * 0.6)),
                 (low + int(width * 0.2), low + int(width * 0.8)),
                 (low + int(width * 0.4), high),
                 (low, low + int(width * 0.4)),
                 (low + int(width * 0.2), low + int(width * 0.6)),
                 (low + int(width * 0.4), low + int(width * 0.8)),
                 (low + int(width * 0.6), high)]
    return np.max(
        [np.abs(np.mean(np.arange(low, high) - preds[low:high]) - event) for low, high in intervals]
    )

def local_argmax_kernel_loss(preds: torch.Tensor, event: int, width: int, use_L1_kernel=False):
    expanded_width = width * 3
    low = max(0, event - width)
    high = min(preds.shape[0], event + width + 1)
    low_expanded = max(0, event - expanded_width)
    high_expanded = min(preds.shape[0], event + expanded_width + 1)

    locations = torch.arange(start=low_expanded, end=high_expanded, step=1, dtype=torch.float32, device=preds.device) - preds[low_expanded:high_expanded] - low

    if use_L1_kernel:
        kernel_preds = kernel_utils.generate_kernel_preds_huber(high - low, locations, kernel_radius=6).cpu().numpy()
    else:
        kernel_preds = kernel_utils.generate_kernel_preds(high - low, locations, kernel_radius=6).cpu().numpy()
    assert kernel_preds.shape == (high - low,), "kernel_preds.shape = {}, high - low = {}".format(kernel_preds.shape, high - low)

    return np.abs(np.argmax(kernel_preds) + low - event)

def local_argmax_sigma_kernel_loss(preds: torch.Tensor, sigmas: torch.Tensor, event: int, width: int):
    expanded_width = width * 3
    low = max(0, event - width)
    high = min(preds.shape[0], event + width + 1)
    low_expanded = max(0, event - expanded_width)
    high_expanded = min(preds.shape[0], event + expanded_width + 1)

    locations = torch.arange(start=low_expanded, end=high_expanded, step=1, dtype=torch.float32, device=preds.device) - preds[low_expanded:high_expanded] - low

    kernel_preds = kernel_utils.generate_kernel_preds_huber_sigmas(high - low, locations, sigmas=sigmas[low_expanded:high_expanded]).cpu().numpy()
    assert kernel_preds.shape == (high - low,), "kernel_preds.shape = {}, high - low = {}".format(kernel_preds.shape, high - low)

    return np.abs(np.argmax(kernel_preds) + low - event)

def local_argmax_loss(preds: np.ndarray, event: int, width: int):
    low = max(0, event - width)
    high = min(preds.shape[0], event + width + 1)

    return np.abs(np.argmax(preds[low:high]) + low - event)

def single_training_step(model_: torch.nn.Module, optimizer_: torch.optim.Optimizer,
                            accel_data_batch: torch.Tensor,
                            event_regression_values: list[torch.Tensor], event_regression_masks: list[torch.Tensor]):
    optimizer_.zero_grad()
    pred_small, pred_mid, pred_large = model_(accel_data_batch, ret_type="deep")

    if loss_type == "huber":
        small_optim_loss = masked_huber_loss(pred_small, event_regression_values[0], event_regression_masks[0])
        mid_optim_loss = masked_huber_loss(pred_mid, event_regression_values[1], event_regression_masks[1])
        large_optim_loss = masked_huber_loss(pred_large, event_regression_values[2], event_regression_masks[2])
    elif loss_type == "mse":
        small_optim_loss = masked_mse_loss(pred_small, event_regression_values[0], event_regression_masks[0])
        mid_optim_loss = masked_mse_loss(pred_mid, event_regression_values[1], event_regression_masks[1])
        large_optim_loss = masked_mse_loss(pred_large, event_regression_values[2], event_regression_masks[2])
    elif loss_type == "huber_mse":
        small_optim_loss = masked_huber_mse_loss(pred_small, event_regression_values[0], event_regression_masks[0])
        mid_optim_loss = masked_huber_mse_loss(pred_mid, event_regression_values[1], event_regression_masks[1])
        large_optim_loss = masked_huber_mse_loss(pred_large, event_regression_values[2], event_regression_masks[2])
    elif loss_type == "ce":
        small_optim_loss = masked_ce_loss(pred_small, event_regression_values[0], event_regression_masks[0])
        mid_optim_loss = masked_ce_loss(pred_mid, event_regression_values[1], event_regression_masks[1])
        large_optim_loss = masked_ce_loss(pred_large, event_regression_values[2], event_regression_masks[2])
    elif loss_type == "focal":
        small_optim_loss = masked_focal_loss(pred_small, event_regression_values[0], event_regression_masks[0])
        mid_optim_loss = masked_focal_loss(pred_mid, event_regression_values[1], event_regression_masks[1])
        large_optim_loss = masked_focal_loss(pred_large, event_regression_values[2], event_regression_masks[2])
    else:
        raise ValueError("Unknown loss type {}".format(loss_type))
    loss = small_optim_loss + mid_optim_loss + large_optim_loss
    loss.backward()
    optimizer_.step()

    with torch.no_grad():
        if loss_type in ["ce", "focal"]:
            small_loss = small_optim_loss.detach().item()
            mid_loss = mid_optim_loss.detach().item()
            large_loss = large_optim_loss.detach().item()
        else:
            small_loss = masked_mae_loss(pred_small, event_regression_values[0], event_regression_masks[0]).item()
            mid_loss = masked_mae_loss(pred_mid, event_regression_values[1], event_regression_masks[1]).item()
            large_loss = masked_mae_loss(pred_large, event_regression_values[2], event_regression_masks[2]).item()

        num_events = torch.sum(torch.sum(event_regression_masks[0], dim=(1, 2)) > 0).item() # number of events in batch

    return loss.item(), small_loss, mid_loss, large_loss, num_events

def single_training_step_standard(model_: torch.nn.Module, optimizer_: torch.optim.Optimizer,
                            accel_data_batch: torch.Tensor,
                            event_regression_values: list[torch.Tensor], event_regression_masks: list[torch.Tensor]):
    optimizer_.zero_grad()
    preds = model_(accel_data_batch, ret_type="deep")

    assert loss_type == "huber" or loss_type == "huber_sigma" or loss_type == "mse", "loss_type must be huber or huber_sigma or mse"
    if loss_type == "huber":
        loss = masked_huber_loss(preds, event_regression_values[0], event_regression_masks[0])
    elif loss_type == "huber_sigma":
        loss = masked_huber_sigma_loss(preds, event_regression_values[0], event_regression_masks[0])
    else:
        loss = masked_mse_loss(preds, event_regression_values[0], event_regression_masks[0])
    loss.backward()
    optimizer_.step()

    with torch.no_grad():
        if loss_type == "huber" or loss_type == "mse":
            mae_loss = masked_mae_loss(preds, event_regression_values[0], event_regression_masks[0]).item()
        else:
            mae_loss = masked_mae_loss(preds[:, :2, :], event_regression_values[0], event_regression_masks[0]).item()

        num_events = torch.sum(torch.sum(event_regression_masks[0], dim=(1, 2)) > 0).item() # number of events in batch

    return loss.item(), mae_loss, num_events

def training_step(record: bool):
    if record:
        for key in train_metrics:
            train_metrics[key].reset()

    # shuffle
    training_sampler.shuffle()

    # training
    with tqdm.tqdm(total=len(training_sampler)) as pbar:
        while training_sampler.entries_remaining() > 0:
            accel_data_batch, event_regression_values, event_regression_masks, increment =\
                training_sampler.sample(batch_size, target_length=target_length, elastic_deformation=use_elastic_deformation,
                                        v_elastic_deformation=use_velastic_deformation, random_vflip=flip_value)

            accel_data_batch_torch = torch.tensor(accel_data_batch, dtype=torch.float32, device=config.device)
            event_regression_values_torch = [torch.tensor(event_regression_values[k], dtype=torch.float32, device=config.device)
                                                for k in range(3)]
            event_regression_masks_torch = [torch.tensor(event_regression_masks[k], dtype=torch.float32, device=config.device)
                                                for k in range(3)]

            if use_anglez_only:
                accel_data_batch_torch = accel_data_batch_torch[:, 0:1, :]
            elif use_enmo_only:
                accel_data_batch_torch = accel_data_batch_torch[:, 1:2, :]

            # train model now
            if use_standard_model:
                loss, mae_loss, num_events = single_training_step_standard(model, optimizer,
                                                                               accel_data_batch_torch,
                                                                               event_regression_values_torch,
                                                                               event_regression_masks_torch)
            else:
                loss, small_loss, mid_loss, large_loss, num_events = single_training_step(model, optimizer,
                                                                               accel_data_batch_torch,
                                                                               event_regression_values_torch,
                                                                               event_regression_masks_torch)

            # record
            if record:
                with torch.no_grad():
                    if use_standard_model:
                        train_metrics["mae"].add(mae_loss, num_events)
                    else:
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
                           event_regression_values: list[torch.Tensor],
                           event_regression_masks: list[torch.Tensor]):
    with torch.no_grad():
        pred_small, pred_mid, pred_large = model_(accel_data_batch, ret_type="deep")

        if loss_type == "ce":
            small_loss = masked_ce_loss_raw(pred_small, event_regression_values[0], event_regression_masks[0]).item()
            mid_loss = masked_ce_loss_raw(pred_mid, event_regression_values[1], event_regression_masks[1]).item()
            large_loss = masked_ce_loss_raw(pred_large, event_regression_values[2], event_regression_masks[2]).item()
        elif loss_type == "focal":
            small_loss = masked_focal_loss_raw(pred_small, event_regression_values[0], event_regression_masks[0]).item()
            mid_loss = masked_focal_loss_raw(pred_mid, event_regression_values[1], event_regression_masks[1]).item()
            large_loss = masked_focal_loss_raw(pred_large, event_regression_values[2], event_regression_masks[2]).item()
        else:
            small_loss = masked_mae_loss_raw(pred_small, event_regression_values[0], event_regression_masks[0]).item()
            mid_loss = masked_mae_loss_raw(pred_mid, event_regression_values[1], event_regression_masks[1]).item()
            large_loss = masked_mae_loss_raw(pred_large, event_regression_values[2], event_regression_masks[2]).item()

        small_mask_num = torch.sum(event_regression_masks[0]).item()
        mid_mask_num = torch.sum(event_regression_masks[1]).item()
        large_mask_num = torch.sum(event_regression_masks[2]).item()

        return small_loss, mid_loss, large_loss, small_mask_num, mid_mask_num, large_mask_num, pred_small, pred_mid, pred_large

def single_validation_step_standard(model_: torch.nn.Module, accel_data_batch: torch.Tensor,
                           event_regression_values: list[torch.Tensor],
                           event_regression_masks: list[torch.Tensor]):
    with torch.no_grad():
        preds = model_(accel_data_batch, ret_type="deep")
        if loss_type == "huber" or loss_type == "mse":
            loss = masked_mae_loss_raw(preds, event_regression_values[0], event_regression_masks[0]).item()
        else:
            loss = masked_mae_loss_raw(preds[:, :2, :], event_regression_values[0], event_regression_masks[0]).item()

        mask_num = torch.sum(event_regression_masks[0]).item()

        return loss, mask_num, preds

def validation_step():
    use_model = swa_model if (use_swa and (epoch > swa_start)) else model

    for key in val_metrics:
        val_metrics[key].reset()

    # validation
    val_sampler.shuffle()
    with (tqdm.tqdm(total=len(val_sampler)) as pbar):
        while val_sampler.entries_remaining() > 0:
            # load the batch
            accel_data_batch, event_regression_values, event_regression_masks, event_locs = val_sampler.sample_series(target_multiple)
            accel_data_batch = torch.tensor(accel_data_batch, dtype=torch.float32, device=config.device)
            event_regression_values = [torch.tensor(event_regression_values[k], dtype=torch.float32, device=config.device)
                                                for k in range(3)]
            event_regression_masks = [torch.tensor(event_regression_masks[k], dtype=torch.float32, device=config.device)
                                                for k in range(3)]

            if use_anglez_only:
                accel_data_batch = accel_data_batch[:, 0:1, :]
            elif use_enmo_only:
                accel_data_batch = accel_data_batch[:, 1:2, :]

            # val model now
            if use_standard_model:
                loss, mask_num, preds = single_validation_step_standard(use_model, accel_data_batch,
                                                                            event_regression_values, event_regression_masks)
            else:
                small_loss, mid_loss, large_loss, small_mask_num, mid_mask_num, large_mask_num,\
                    pred_small, pred_mid, pred_large = single_validation_step(use_model, accel_data_batch,
                                                                                event_regression_values, event_regression_masks)

            # record
            with torch.no_grad():
                if use_standard_model:
                    val_metrics["mae"].add(loss, mask_num)

                    pred_torch = preds[0, ...]

                    for event in event_locs:
                        onset = event["onset"]
                        wakeup = event["wakeup"]

                        width = int(regression_width[0] * 1.0)
                        if loss_type == "huber" or loss_type == "mse":
                            onset_argmax_ae = local_argmax_kernel_loss(pred_torch[0, :], onset, width)
                            wakeup_argmax_ae = local_argmax_kernel_loss(pred_torch[1, :], wakeup, width)
                        else:
                            onset_argmax_ae = local_argmax_sigma_kernel_loss(pred_torch[0, :], pred_torch[2, :], onset, width)
                            wakeup_argmax_ae = local_argmax_sigma_kernel_loss(pred_torch[1, :], pred_torch[3, :], wakeup, width)
                        val_metrics["argmax_ae"].add(onset_argmax_ae, 1)
                        val_metrics["argmax_ae"].add(wakeup_argmax_ae, 1)
                else:
                    val_metrics["mae_small"].add(small_loss, small_mask_num)
                    val_metrics["mae_mid"].add(mid_loss, mid_mask_num)
                    val_metrics["mae_large"].add(large_loss, large_mask_num)

                    pred_small_torch = pred_small[0, ...]
                    pred_mid_torch = pred_mid[0, ...]
                    pred_large_torch = pred_large[0, ...]
                    preds_torch = [pred_small_torch, pred_mid_torch, pred_large_torch]
                    pred_small = pred_small[0, ...].cpu().numpy()
                    pred_mid = pred_mid[0, ...].cpu().numpy()
                    pred_large = pred_large[0, ...].cpu().numpy()
                    assert pred_small.shape == pred_mid.shape == pred_large.shape, "pred_small.shape = {}, pred_mid.shape = {}, pred_large.shape = {}".format(pred_small.shape, pred_mid.shape, pred_large.shape)
                    preds = [pred_small, pred_mid, pred_large]
                    metric_labels = ["median_ae_small", "median_ae_mid", "median_ae_large"]
                    metric_labels_mean = ["mean_ae_small", "mean_ae_mid", "mean_ae_large"]
                    metric_labels_argmax = ["argmax_ae_small", "argmax_ae_mid", "argmax_ae_large"]

                    for event in event_locs:
                        onset = event["onset"]
                        wakeup = event["wakeup"]

                        for k in range(len(regression_width)):
                            width = int(regression_width[k] * 1.0)
                            if loss_type in ["ce", "focal"]:
                                onset_argmax_ae = local_argmax_loss(preds[k][0, :], onset, width)
                                wakeup_argmax_ae = local_argmax_loss(preds[k][1, :], wakeup, width)
                                val_metrics[metric_labels_argmax[k]].add(onset_argmax_ae, 1)
                                val_metrics[metric_labels_argmax[k]].add(wakeup_argmax_ae, 1)
                            else:
                                onset_median_ae = local_median_ae_loss(preds[k][0, :], onset, width)
                                wakeup_median_ae = local_median_ae_loss(preds[k][1, :], wakeup, width)
                                val_metrics[metric_labels[k]].add(onset_median_ae, 1)
                                val_metrics[metric_labels[k]].add(wakeup_median_ae, 1)

                                onset_mean_ae = local_mean_ae_loss(preds[k][0, :], onset, width)
                                wakeup_mean_ae = local_mean_ae_loss(preds[k][1, :], wakeup, width)
                                val_metrics[metric_labels_mean[k]].add(onset_mean_ae, 1)
                                val_metrics[metric_labels_mean[k]].add(wakeup_mean_ae, 1)

                                onset_argmax_ae = local_argmax_kernel_loss(preds_torch[k][0, :], onset, width)
                                wakeup_argmax_ae = local_argmax_kernel_loss(preds_torch[k][1, :], wakeup, width)
                                val_metrics[metric_labels_argmax[k]].add(onset_argmax_ae, 1)
                                val_metrics[metric_labels_argmax[k]].add(wakeup_argmax_ae, 1)


            pbar.update(1)

    current_metrics = {}
    for key in val_metrics:
        val_metrics[key].write_to_dict(current_metrics)

    for key in current_metrics:
        val_history[key].append(current_metrics[key])

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

    if not momenta:  # no batchnorm layers
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
                accel_data_batch, event_regression_values, event_regression_masks, increment =\
                    training_sampler.sample(batch_size, target_length=target_length, elastic_deformation=use_elastic_deformation,
                                            v_elastic_deformation=use_velastic_deformation, random_vflip=flip_value)

                accel_data_batch_torch = torch.tensor(accel_data_batch, dtype=torch.float32, device=config.device)

                if use_anglez_only:
                    accel_data_batch_torch = accel_data_batch_torch[:, 0:1, :]
                elif use_enmo_only:
                    accel_data_batch_torch = accel_data_batch_torch[:, 1:2, :]

                # forward pass
                swa_model(accel_data_batch_torch, ret_type="deep")

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
    parser.add_argument("--loss", type=str, default="huber", help="Loss to use. Default huber. Available options: huber, mse, huber_mse, ce, focal, huber_sigma")
    parser.add_argument("--regression_width", type=int, nargs="+", default=[60, 120, 240], help="Number of regression values to predict for each event. Default [60, 120, 240].")
    parser.add_argument("--regression_kernel", type=int, default=None, help="Kernel size for the regression layers. Default None, which means usual regression is performed.")
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="Learning rate to use. Default 5e-3.")
    parser.add_argument("--use_swa", action="store_true", help="Whether to use SWA. Default False.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum to use. Default 0.9. This would be the momentum for SGD, and beta1 for Adam.")
    parser.add_argument("--second_momentum", type=float, default=0.999, help="Second momentum to use. Default 0.999. This would be beta2 for Adam. Ignored if SGD.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use. Default 0.0.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Which optimizer to use. Available options: adam, sgd. Default adam.")
    parser.add_argument("--epochs_per_save", type=int, default=1, help="Number of epochs between saves. Default 1.")
    parser.add_argument("--use_standard_model", action="store_true", help="Whether to use the standard model. Default False. This overrides the options below.")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[2, 2, 2, 2, 3],
                        help="Number of hidden 2d blocks for ResNet backbone.")
    parser.add_argument("--hidden_channels", type=int, nargs="+", default=[4, 4, 8, 16, 32], help="Number of hidden channels. Default 2. Can be a list to specify num channels in each downsampled layer.")
    parser.add_argument("--bottleneck_factor", type=int, default=4, help="The bottleneck factor of the ResNet backbone. Default 4.")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for the first layer. Default 3.")
    parser.add_argument("--disable_deep_upconv_contraction", action="store_true", help="Whether to disable the deep upconv contraction. Default False.")
    parser.add_argument("--deep_upconv_kernel", type=int, default=5, help="Kernel size for the deep upconv layers. Default 5.")
    parser.add_argument("--deep_upconv_channels_override", type=int, default=None, help="Override the number of channels for the deep upconv layers. Default None.")
    parser.add_argument("--flip_value", action="store_true", help="Whether to flip the value of the intervals. Default False.")
    parser.add_argument("--expand", type=int, default=360, help="Expand the intervals by this amount. Default 360.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate. Default 0.0.")
    parser.add_argument("--use_anglez_only", action="store_true", help="Whether to use only anglez. Default False.")
    parser.add_argument("--use_enmo_only", action="store_true", help="Whether to use only enmo. Default False.")
    parser.add_argument("--use_elastic_deformation", action="store_true", help="Whether to use elastic deformation. Default False.")
    parser.add_argument("--use_velastic_deformation", action="store_true", help="Whether to use velastic deformation (only available for anglez). Default False.")
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
    loss_type = args.loss
    regression_width = args.regression_width
    regression_kernel = args.regression_kernel
    learning_rate = args.learning_rate
    use_swa = args.use_swa
    momentum = args.momentum
    second_momentum = args.second_momentum
    weight_decay = args.weight_decay
    optimizer_type = args.optimizer
    epochs_per_save = args.epochs_per_save
    use_standard_model = args.use_standard_model
    hidden_blocks = args.hidden_blocks
    hidden_channels = args.hidden_channels
    bottleneck_factor = args.bottleneck_factor
    kernel_size = args.kernel_size
    disable_deep_upconv_contraction = args.disable_deep_upconv_contraction
    deep_upconv_kernel = args.deep_upconv_kernel
    deep_upconv_channels_override = args.deep_upconv_channels_override
    flip_value = args.flip_value
    expand = args.expand
    dropout = args.dropout
    use_anglez_only = args.use_anglez_only
    use_enmo_only = args.use_enmo_only
    use_elastic_deformation = args.use_elastic_deformation
    use_velastic_deformation = args.use_velastic_deformation
    batch_size = args.batch_size
    num_extra_steps = args.num_extra_steps

    assert not (use_anglez_only and use_enmo_only), "Cannot use both anglez only and enmo only."
    assert isinstance(regression_width, list), "Regression width must be a list."
    assert len(regression_width) == 3, "Regression width must be a list of length 3."
    assert loss_type in ["huber", "mse", "huber_mse", "ce", "focal", "huber_sigma"], "Invalid loss type."
    if regression_kernel is not None:
        assert loss_type in ["ce", "focal"], "Must use probability loss type if using regression kernel."
    else:
        assert loss_type in ["huber", "mse", "huber_mse", "huber_sigma"], "Must use regression loss type if not using regression kernel."
    if use_standard_model:
        assert use_anglez_only, "Must use anglez only if using standard model."
        assert loss_type == "huber" or loss_type == "huber_sigma" or loss_type == "mse", "Must use huber loss/MSE loss if using standard model."
    if loss_type == "huber_sigma":
        assert use_standard_model, "Must use standard model if using huber sigma loss."

    if isinstance(hidden_channels, int):
        hidden_channels = [hidden_channels]
    if len(hidden_channels) == 1:
        chnls = hidden_channels[0]
        hidden_channels = []
        for k in range(len(hidden_blocks)):
            hidden_channels.append(chnls * (2 ** k))

    print("Epochs: " + str(epochs))
    print("Loss type: " + loss_type)
    if use_standard_model:
        print("Regression width: " + str(regression_width[0]))
        print("--- Using standard model ---")
    else:
        print("Regression width: " + str(regression_width))
        print("Dropout: " + str(dropout))
        print("Bottleneck factor: " + str(bottleneck_factor))
        print("Hidden channels: " + str(hidden_channels))
        print("Hidden blocks: " + str(hidden_blocks))
        print("Kernel size: " + str(kernel_size))
        print("Deep upconv contraction: " + str(not disable_deep_upconv_contraction))
        print("Deep upconv kernel: " + str(deep_upconv_kernel))
        print("Deep upconv channels override: " + str(deep_upconv_channels_override))
    model_unet.BATCH_NORM_MOMENTUM = 1 - momentum

    # initialize model
    in_channels = 1 if (use_anglez_only or use_enmo_only) else 2
    if use_standard_model:
        model = model_event_unet.EventRegressorUnet(use_learnable_sigma=(loss_type == "huber_sigma"), hidden_channels=hidden_channels,
                                                    blocks=hidden_blocks)
    else:
        model = model_attention_unet.Unet3fDeepSupervision(in_channels, hidden_channels, kernel_size=kernel_size, blocks=hidden_blocks,
                                bottleneck_factor=bottleneck_factor, squeeze_excitation=False,
                                squeeze_excitation_bottleneck_factor=4,
                                dropout=dropout,
                                use_batch_norm=True, out_channels=2, attn_out_channels=2,

                                deep_supervision_contraction=not disable_deep_upconv_contraction, deep_supervision_kernel_size=deep_upconv_kernel,
                                deep_supervision_channels_override=deep_upconv_channels_override)
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

    # Create swa model
    if use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=10.0 * learning_rate, anneal_strategy="linear", anneal_epochs=5)
        swa_start = 3

    model_config = {
        "model": "Unet with attention segmentation",
        "training_dataset": train_dset_name,
        "validation_dataset": val_dset_name,
        "epochs": epochs,
        "loss_type": loss_type,
        "regression_width": regression_width,
        "regression_kernel": regression_kernel,
        "learning_rate": learning_rate,
        "use_swa": use_swa,
        "momentum": momentum,
        "second_momentum": second_momentum,
        "weight_decay": weight_decay,
        "optimizer": optimizer_type,
        "epochs_per_save": epochs_per_save,
        "hidden_blocks": hidden_blocks,
        "hidden_channels": hidden_channels,
        "bottleneck_factor": bottleneck_factor,
        "kernel_size": kernel_size,
        "disable_deep_upconv_contraction": disable_deep_upconv_contraction,
        "deep_upconv_kernel": deep_upconv_kernel,
        "deep_upconv_channels_override": deep_upconv_channels_override,
        "flip_value": flip_value,
        "expand": expand,
        "dropout": dropout,
        "use_anglez_only": use_anglez_only,
        "use_enmo_only": use_enmo_only,
        "use_elastic_deformation": use_elastic_deformation,
        "use_velastic_deformation": use_velastic_deformation,
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
    if use_standard_model:
        train_metrics["mae"] = metrics.NumericalMetric("train_mae")
        val_metrics["mae"] = metrics.NumericalMetric("val_mae")
        val_metrics["argmax_ae"] = metrics.NumericalMetric("val_argmax_ae", require_median=True)
    else:
        train_metrics["mae_small"] = metrics.NumericalMetric("train_mae_small")
        train_metrics["mae_mid"] = metrics.NumericalMetric("train_mae_mid")
        train_metrics["mae_large"] = metrics.NumericalMetric("train_mae_large")
        val_metrics["mae_small"] = metrics.NumericalMetric("val_mae_small")
        val_metrics["mae_mid"] = metrics.NumericalMetric("val_mae_mid")
        val_metrics["mae_large"] = metrics.NumericalMetric("val_mae_large")
        if loss_type not in ["ce", "focal"]:
            val_metrics["median_ae_small"] = metrics.NumericalMetric("val_median_ae_small")
            val_metrics["median_ae_mid"] = metrics.NumericalMetric("val_median_ae_mid")
            val_metrics["median_ae_large"] = metrics.NumericalMetric("val_median_ae_large")
            val_metrics["mean_ae_small"] = metrics.NumericalMetric("val_mean_ae_small")
            val_metrics["mean_ae_mid"] = metrics.NumericalMetric("val_mean_ae_mid")
            val_metrics["mean_ae_large"] = metrics.NumericalMetric("val_mean_ae_large")
        val_metrics["argmax_ae_small"] = metrics.NumericalMetric("val_argmax_ae_small", require_median=True)
        val_metrics["argmax_ae_mid"] = metrics.NumericalMetric("val_argmax_ae_mid", require_median=True)
        val_metrics["argmax_ae_large"] = metrics.NumericalMetric("val_argmax_ae_large", require_median=True)

    # Compile
    #single_training_step_compile = torch.compile(single_training_step)

    # Initialize the sampler
    input_type = "anglez" if use_anglez_only else ("enmo" if use_enmo_only else "both")
    print("Initializing the samplers...")
    print("Batch size: " + str(batch_size))
    print("Expand: " + str(expand))
    print("Input type: " + input_type)
    training_sampler = convert_to_regression_events.IntervalRegressionSampler(training_entries, all_data, event_regressions=regression_width,
                                                                              train_or_test="train", use_kernel=regression_kernel)
    val_sampler = convert_to_regression_events.IntervalRegressionSampler(validation_entries, all_data, event_regressions=regression_width,
                                                                        train_or_test="test", use_kernel=regression_kernel)
    target_multiple = 3 * (2 ** (len(hidden_blocks) - 2))
    target_length = training_sampler.max_length + (2 * expand)
    target_length = int(np.ceil((target_length + 0.0) / target_multiple)) * int(target_multiple)
    print("Target length: " + str(target_length))


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
                if use_swa and epoch > swa_start:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                training_step(record=False)

            print("Running the usual step of gradient descent.")
            training_step(record=True)
            ref_metric = "train_mae" if use_standard_model else "train_mae_small"
            if np.isnan(train_history[ref_metric][-1]):
                print("Is nan. Reverting...")
                model.load_state_dict(torch.load(os.path.join(model_dir, "latest_model.pt")))
                optimizer.load_state_dict(torch.load(os.path.join(model_dir, "latest_optimizer.pt")))
            else:
                torch.save(model.state_dict(), os.path.join(model_dir, "latest_model.pt"))
                torch.save(optimizer.state_dict(), os.path.join(model_dir, "latest_optimizer.pt"))

            if use_swa and epoch > swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
                update_SWA_bn(swa_model)

            # switch model to eval mode, and fix all running stats for batchnorm layers
            model.eval()
            if use_swa and (epoch > swa_start):
                swa_model.eval()
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