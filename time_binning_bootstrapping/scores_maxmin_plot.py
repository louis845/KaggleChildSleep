import shutil
import sys
import os
import json
from typing import Iterator

import numpy as np
import h5py
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import convert_to_seriesid_events
import postprocessing
import metrics_ap_bootstrap
import model_event_density_unet
import bad_series_list


validation_AP_tolerances = [1, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30][::-1]

def plot_single_precision_recall_curve(ax, precisions, recalls, ap, title):
    ax.plot(recalls, precisions)
    #ax.scatter(recalls, precisions, c=proba, cmap="coolwarm")
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, "AP: {:.4f}".format(ap), horizontalalignment="center", verticalalignment="center")

def plot_ap_curve(title: str, out_file: str,
         ap_onset_precisions: list[np.ndarray], ap_onset_recalls: list[np.ndarray], ap_onset_average_precisions: list[float],
         ap_wakeup_precisions: list[np.ndarray], ap_wakeup_recalls: list[np.ndarray], ap_wakeup_average_precisions: list[float]):
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    plt.suptitle(title)
    for k in range(len(validation_AP_tolerances)):
        ax = axes[k // 5, k % 5]
        plot_single_precision_recall_curve(ax, ap_onset_precisions[k], ap_onset_recalls[k],
                                           ap_onset_average_precisions[k],
                                           "Onset AP{}".format(validation_AP_tolerances[k]))
        ax = axes[(k + 10) // 5, (k + 10) % 5]
        plot_single_precision_recall_curve(ax, ap_wakeup_precisions[k], ap_wakeup_recalls[k],
                                           ap_wakeup_average_precisions[k],
                                           "Wakeup AP{}".format(validation_AP_tolerances[k]))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(out_file)
    plt.close()

def convert_to_regression_folder(regression_labels_name: str, kernel_shape: str, kernel_width: int):
    regression_folder = os.path.join("./inference_regression_statistics/regression_labels", regression_labels_name,
                        "{}_kernel{}".format(kernel_shape, kernel_width))
    assert os.path.isdir(regression_folder)
    return regression_folder

def convert_to_density_folder(density_labels_name: str):
    density_folder = os.path.join("./inference_density_statistics/density_labels", density_labels_name)
    assert os.path.isdir(density_folder)
    return density_folder

def get_regression_preds_locs(selected_regression_folders: list[str], series_id: str, alignment=True,
                              cutoff=0.01, pruning=72):
    # compute locs here
    num_regression_preds = 0
    onset_kernel_vals, wakeup_kernel_vals = None, None
    for folder in selected_regression_folders:
        onset_kernel = np.load(os.path.join(folder, "{}_onset.npy".format(series_id)))
        wakeup_kernel = np.load(os.path.join(folder, "{}_wakeup.npy".format(series_id)))
        if onset_kernel_vals is None:
            onset_kernel_vals = onset_kernel
            wakeup_kernel_vals = wakeup_kernel
        else:
            onset_kernel_vals = onset_kernel_vals + onset_kernel
            wakeup_kernel_vals = wakeup_kernel_vals + wakeup_kernel
        num_regression_preds += 1

    onset_kernel_vals = onset_kernel_vals / num_regression_preds
    wakeup_kernel_vals = wakeup_kernel_vals / num_regression_preds

    onset_locs = (onset_kernel_vals[1:-1] > onset_kernel_vals[0:-2]) & (onset_kernel_vals[1:-1] > onset_kernel_vals[2:])
    onset_locs = np.argwhere(onset_locs).flatten() + 1
    wakeup_locs = (wakeup_kernel_vals[1:-1] > wakeup_kernel_vals[0:-2]) & (wakeup_kernel_vals[1:-1] > wakeup_kernel_vals[2:])
    wakeup_locs = np.argwhere(wakeup_locs).flatten() + 1

    onset_values = onset_kernel_vals[onset_locs]
    wakeup_values = wakeup_kernel_vals[wakeup_locs]

    onset_locs = onset_locs[onset_values > cutoff]
    wakeup_locs = wakeup_locs[wakeup_values > cutoff]

    if pruning > 0:
        if len(onset_locs) > 0:
            onset_locs = postprocessing.prune(onset_locs, onset_values[onset_values > cutoff], pruning)
        if len(wakeup_locs) > 0:
            wakeup_locs = postprocessing.prune(wakeup_locs, wakeup_values[wakeup_values > cutoff], pruning)

    if alignment:
        seconds_values = np.load("./data_naive/{}/secs.npy".format(series_id))
        first_zero = postprocessing.compute_first_zero(seconds_values)
        if len(onset_locs) > 0:
            onset_locs = postprocessing.align_predictions(onset_locs, onset_kernel_vals, first_zero=first_zero)
        if len(wakeup_locs) > 0:
            wakeup_locs = postprocessing.align_predictions(wakeup_locs, wakeup_kernel_vals, first_zero=first_zero)

    return onset_locs, wakeup_locs, onset_kernel_vals, wakeup_kernel_vals

def event_density_file_logit_iterator(selected_density_folder, series_id) -> Iterator[dict[str, np.ndarray]]:
    # load the logits
    logits_file = os.path.join(selected_density_folder, series_id, "intervals.h5")
    with h5py.File(logits_file, "r") as f:
        intervals_start = f["intervals_start"][:]
        intervals_end = f["intervals_end"][:]
        intervals_logits = f["intervals_logits"][:]
        intervals_event_presence = f["intervals_event_presence"][:]

    for k in range(len(intervals_start)):
        interval_start = intervals_start[k]
        interval_end = intervals_end[k]
        interval_logits = intervals_logits[k]
        interval_event_presence = intervals_event_presence[k]
        yield {
            "interval_start": interval_start,
            "interval_end": interval_end,
            "interval_logits": interval_logits,
            "interval_event_presence": interval_event_presence
        }

def validation_ap(gt_events, all_series_ids,
                  selected_density_folders: list[str], selected_regression_folders: list[str],
                  cutoff, regression_pruning,

                  binning_cutoffs: np.ndarray,

                  exclude_bad_segmentations: bool=True):
    selected_series_ids = all_series_ids

    ap_onset_metrics = [metrics_ap_bootstrap.EventMetricsBootstrap(name="", tolerance=tolerance * 12) for tolerance in validation_AP_tolerances]
    ap_wakeup_metrics = [metrics_ap_bootstrap.EventMetricsBootstrap(name="", tolerance=tolerance * 12) for tolerance in validation_AP_tolerances]

    cutoff_onset_values = 0
    cutoff_wakeup_values = 0
    total_onset_values = 0
    total_wakeup_values = 0

    for series_id in selected_series_ids:
        if exclude_bad_segmentations and series_id in bad_series_list.noisy_bad_segmentations:
            continue

        # compute the regression predictions
        preds_locs_onset, preds_locs_wakeup, onset_kernel_vals, wakeup_kernel_vals = get_regression_preds_locs(selected_regression_folders, series_id,
                                                                                                               alignment=False, pruning=regression_pruning)
        total_length = len(onset_kernel_vals)

        # load and compute the probas
        onset_locs_all_probas, wakeup_locs_all_probas = None, None
        for k in range(len(selected_density_folders)):
            logit_loader = event_density_file_logit_iterator(selected_density_folders[k], series_id)
            _, onset_locs_probas, wakeup_locs_probas = model_event_density_unet.event_density_probas_from_interval_info(
                                                                             interval_info_stream=logit_loader,
                                                                             total_length=total_length,
                                                                             predicted_locations=[{
                                                                                    "onset": preds_locs_onset,
                                                                                    "wakeup": preds_locs_wakeup
                                                                             }],
                                                                             return_probas=False)
            if onset_locs_all_probas is None:
                onset_locs_all_probas = onset_locs_probas[0]
                wakeup_locs_all_probas = wakeup_locs_probas[0]
            else:
                onset_locs_all_probas += onset_locs_probas[0]
                wakeup_locs_all_probas += wakeup_locs_probas[0]
        onset_locs_all_probas /= len(selected_density_folders)
        wakeup_locs_all_probas /= len(selected_density_folders)

        # prune using cutoff. also compute cutoff stats
        total_onset_values += len(preds_locs_onset)
        total_wakeup_values += len(preds_locs_wakeup)
        if cutoff > 0:
            cutoff_onset_values += np.sum(onset_locs_all_probas > cutoff)
            cutoff_wakeup_values += np.sum(wakeup_locs_all_probas > cutoff)

            preds_locs_onset = preds_locs_onset[onset_locs_all_probas > cutoff]
            onset_locs_all_probas = onset_locs_all_probas[onset_locs_all_probas > cutoff]
            preds_locs_wakeup = preds_locs_wakeup[wakeup_locs_all_probas > cutoff]
            wakeup_locs_all_probas = wakeup_locs_all_probas[wakeup_locs_all_probas > cutoff]

        # align
        seconds_values = np.load("./data_naive/{}/secs.npy".format(series_id))
        first_zero = postprocessing.compute_first_zero(seconds_values)
        if len(preds_locs_onset) > 0:
            preds_locs_onset = postprocessing.align_predictions(preds_locs_onset, onset_kernel_vals, first_zero=first_zero,
                                                                return_sorted=False)
            if not np.all(preds_locs_onset[1:] >= preds_locs_onset[:-1]):
                idxsort = np.argsort(preds_locs_onset)
                preds_locs_onset = preds_locs_onset[idxsort]
                onset_locs_all_probas = onset_locs_all_probas[idxsort]
        if len(preds_locs_wakeup) > 0:
            preds_locs_wakeup = postprocessing.align_predictions(preds_locs_wakeup, wakeup_kernel_vals, first_zero=first_zero,
                                                                 return_sorted=False)
            if not np.all(preds_locs_wakeup[1:] >= preds_locs_wakeup[:-1]):
                idxsort = np.argsort(preds_locs_wakeup)
                preds_locs_wakeup = preds_locs_wakeup[idxsort]
                wakeup_locs_all_probas = wakeup_locs_all_probas[idxsort]

        # get the ground truth
        gt_onset_locs = gt_events[series_id]["onset"]
        gt_wakeup_locs = gt_events[series_id]["wakeup"]

        # exclude end if excluding bad segmentations
        if exclude_bad_segmentations and series_id in bad_series_list.bad_segmentations_tail:
            if len(gt_onset_locs) > 0 and len(gt_wakeup_locs) > 0:
                last = gt_wakeup_locs[-1] + 8640

                onset_cutoff = np.searchsorted(preds_locs_onset, last, side="right")
                wakeup_cutoff = np.searchsorted(preds_locs_wakeup, last, side="right")
                preds_locs_onset = preds_locs_onset[:onset_cutoff]
                preds_locs_wakeup = preds_locs_wakeup[:wakeup_cutoff]
                onset_locs_all_probas = onset_locs_all_probas[:onset_cutoff]
                wakeup_locs_all_probas = wakeup_locs_all_probas[:wakeup_cutoff]

        # add info
        for ap_onset_metric, ap_wakeup_metric in zip(ap_onset_metrics, ap_wakeup_metrics):
            ap_onset_metric.add(pred_locs=preds_locs_onset, pred_probas=onset_locs_all_probas, gt_locs=gt_onset_locs)
            ap_wakeup_metric.add(pred_locs=preds_locs_wakeup, pred_probas=wakeup_locs_all_probas, gt_locs=gt_wakeup_locs)

    # get matches, probas and num positive
    all_ap_onset_matches, all_ap_onset_probas, all_ap_onset_numpositive = [], [], []
    all_ap_wakeup_matches, all_ap_wakeup_probas, all_ap_wakeup_numpositive = [], [], []
    for ap_onset_metric, ap_wakeup_metric in zip(ap_onset_metrics, ap_wakeup_metrics):
        ap_onset_matches, ap_onset_probas = ap_onset_metric.get_sorted_matches_probas()
        ap_wakeup_matches, ap_wakeup_probas = ap_wakeup_metric.get_sorted_matches_probas()

        all_ap_onset_matches.append(ap_onset_matches)
        all_ap_onset_probas.append(ap_onset_probas)
        all_ap_wakeup_matches.append(ap_wakeup_matches)
        all_ap_wakeup_probas.append(ap_wakeup_probas)

        all_ap_onset_numpositive.append(ap_onset_metric.num_positive)
        all_ap_wakeup_numpositive.append(ap_wakeup_metric.num_positive)

    # compute mAP, mAP lower bound, mAP upper bound
    num_tolerances = len(validation_AP_tolerances)
    all_onset_aps, all_onset_min_aps, all_onset_max_aps, all_onset_mean_aps = [], [], [], []
    all_onset_recalls, all_onset_min_recalls, all_onset_max_recalls, all_onset_mean_recalls = [], [], [], []
    all_onset_precisions, all_onset_min_precisions, all_onset_max_precisions, all_onset_mean_precisions = [], [], [], []

    all_wakeup_aps, all_wakeup_min_aps, all_wakeup_max_aps, all_wakeup_mean_aps = [], [], [], []
    all_wakeup_recalls, all_wakeup_min_recalls, all_wakeup_max_recalls, all_wakeup_mean_recalls = [], [], [], []
    all_wakeup_precisions, all_wakeup_min_precisions, all_wakeup_max_precisions, all_wakeup_mean_precisions = [], [], [], []

    for k in range(num_tolerances):
        # get matches and probas, and resp. for lower and upper bounds
        onset_matches_per_tolerance = all_ap_onset_matches[k]
        onset_probas_per_tolerance = all_ap_onset_probas[k]
        onset_numpositive_per_tolerance = all_ap_onset_numpositive[k]
        onset_matches_per_tolerance_lower, onset_probas_per_tolerance_lower = metrics_ap_bootstrap.get_lower_bound(
            onset_matches_per_tolerance, onset_probas_per_tolerance, binning_cutoffs)
        onset_matches_per_tolerance_upper, onset_probas_per_tolerance_upper = metrics_ap_bootstrap.get_upper_bound(
            onset_matches_per_tolerance, onset_probas_per_tolerance, binning_cutoffs)
        onset_matches_per_tolerance_mean, onset_probas_per_tolerance_mean = metrics_ap_bootstrap.get_mean(
            onset_matches_per_tolerance, onset_probas_per_tolerance, binning_cutoffs)


        wakeup_matches_per_tolerance = all_ap_wakeup_matches[k]
        wakeup_probas_per_tolerance = all_ap_wakeup_probas[k]
        wakeup_numpositive_per_tolerance = all_ap_wakeup_numpositive[k]
        wakeup_matches_per_tolerance_lower, wakeup_probas_per_tolerance_lower = metrics_ap_bootstrap.get_lower_bound(
            wakeup_matches_per_tolerance, wakeup_probas_per_tolerance, binning_cutoffs)
        wakeup_matches_per_tolerance_upper, wakeup_probas_per_tolerance_upper = metrics_ap_bootstrap.get_upper_bound(
            wakeup_matches_per_tolerance, wakeup_probas_per_tolerance, binning_cutoffs)
        wakeup_matches_per_tolerance_mean, wakeup_probas_per_tolerance_mean = metrics_ap_bootstrap.get_mean(
            wakeup_matches_per_tolerance, wakeup_probas_per_tolerance, binning_cutoffs)


        # compute precision and recall curves
        precision, recall, average_precision, _ = metrics_ap_bootstrap.compute_precision_recall_curve(
                                                            onset_numpositive_per_tolerance,
                                                            onset_matches_per_tolerance,
                                                            onset_probas_per_tolerance)
        all_onset_aps.append(average_precision)
        all_onset_recalls.append(recall)
        all_onset_precisions.append(precision)
        precision, recall, average_precision, _ = metrics_ap_bootstrap.compute_precision_recall_curve(
                                                            onset_numpositive_per_tolerance,
                                                            onset_matches_per_tolerance_lower,
                                                            onset_probas_per_tolerance_lower)
        all_onset_min_aps.append(average_precision)
        all_onset_min_recalls.append(recall)
        all_onset_min_precisions.append(precision)
        precision, recall, average_precision, _ = metrics_ap_bootstrap.compute_precision_recall_curve(
                                                            onset_numpositive_per_tolerance,
                                                            onset_matches_per_tolerance_upper,
                                                            onset_probas_per_tolerance_upper)
        all_onset_max_aps.append(average_precision)
        all_onset_max_recalls.append(recall)
        all_onset_max_precisions.append(precision)
        precision, recall, average_precision, _ = metrics_ap_bootstrap.compute_precision_recall_curve(
                                                            onset_numpositive_per_tolerance,
                                                            onset_matches_per_tolerance_mean,
                                                            onset_probas_per_tolerance_mean)
        all_onset_mean_aps.append(average_precision)
        all_onset_mean_recalls.append(recall)
        all_onset_mean_precisions.append(precision)

        precision, recall, average_precision, _ = metrics_ap_bootstrap.compute_precision_recall_curve(
                                                            wakeup_numpositive_per_tolerance,
                                                            wakeup_matches_per_tolerance,
                                                            wakeup_probas_per_tolerance)
        all_wakeup_aps.append(average_precision)
        all_wakeup_recalls.append(recall)
        all_wakeup_precisions.append(precision)
        precision, recall, average_precision, _ = metrics_ap_bootstrap.compute_precision_recall_curve(
                                                            wakeup_numpositive_per_tolerance,
                                                            wakeup_matches_per_tolerance_lower,
                                                            wakeup_probas_per_tolerance_lower)
        all_wakeup_min_aps.append(average_precision)
        all_wakeup_min_recalls.append(recall)
        all_wakeup_min_precisions.append(precision)
        precision, recall, average_precision, _ = metrics_ap_bootstrap.compute_precision_recall_curve(
                                                            wakeup_numpositive_per_tolerance,
                                                            wakeup_matches_per_tolerance_upper,
                                                            wakeup_probas_per_tolerance_upper)
        all_wakeup_max_aps.append(average_precision)
        all_wakeup_max_recalls.append(recall)
        all_wakeup_max_precisions.append(precision)
        precision, recall, average_precision, _ = metrics_ap_bootstrap.compute_precision_recall_curve(
                                                            wakeup_numpositive_per_tolerance,
                                                            wakeup_matches_per_tolerance_mean,
                                                            wakeup_probas_per_tolerance_mean)
        all_wakeup_mean_aps.append(average_precision)
        all_wakeup_mean_recalls.append(recall)
        all_wakeup_mean_precisions.append(precision)

    # compute mAPs
    onset_mAP, onset_min_mAP, onset_max_mAP, onset_mean_mAP = np.mean(all_onset_aps), np.mean(all_onset_min_aps), np.mean(all_onset_max_aps), np.mean(all_onset_mean_aps)
    wakeup_mAP, wakeup_min_mAP, wakeup_max_mAP, wakeup_mean_mAP = np.mean(all_wakeup_aps), np.mean(all_wakeup_min_aps), np.mean(all_wakeup_max_aps), np.mean(all_wakeup_mean_aps)

    return {"onset_mAP": onset_mAP, "onset_min_mAP": onset_min_mAP, "onset_max_mAP": onset_max_mAP, "onset_mean_mAP": onset_mean_mAP,
            "wakeup_mAP": wakeup_mAP, "wakeup_min_mAP": wakeup_min_mAP, "wakeup_max_mAP": wakeup_max_mAP, "wakeup_mean_mAP": wakeup_mean_mAP,

            "onset_aps": all_onset_aps, "onset_min_aps": all_onset_min_aps, "onset_max_aps": all_onset_max_aps, "onset_mean_aps": all_onset_mean_aps,
            "wakeup_aps": all_wakeup_aps, "wakeup_min_aps": all_wakeup_min_aps, "wakeup_max_aps": all_wakeup_max_aps, "wakeup_mean_aps": all_wakeup_mean_aps,

            "onset_recalls": all_onset_recalls, "onset_min_recalls": all_onset_min_recalls, "onset_max_recalls": all_onset_max_recalls, "onset_mean_recalls": all_onset_mean_recalls,
            "wakeup_recalls": all_wakeup_recalls, "wakeup_min_recalls": all_wakeup_min_recalls, "wakeup_max_recalls": all_wakeup_max_recalls, "wakeup_mean_recalls": all_wakeup_mean_recalls,

            "onset_precisions": all_onset_precisions, "onset_min_precisions": all_onset_min_precisions, "onset_max_precisions": all_onset_max_precisions, "onset_mean_precisions": all_onset_mean_precisions,
            "wakeup_precisions": all_wakeup_precisions, "wakeup_min_precisions": all_wakeup_min_precisions, "wakeup_max_precisions": all_wakeup_max_precisions, "wakeup_mean_precisions": all_wakeup_mean_precisions,}

"""
gt_events, all_series_ids,
                  selected_density_folders: list[str], selected_regression_folders: list[str],
                  cutoff, regression_pruning,

                  binning_cutoffs: np.ndarray,

                  exclude_bad_segmentations: bool=True
"""


def run_min_max_AP(config, all_series_ids, binning_cutoffs: np.ndarray,
                   gt_events: np.ndarray):
    regression_cfg_content = config["regression_cfg_content"]
    density_cfg_content = config["density_cfg_content"]

    regression_kernels = regression_cfg_content["regression_kernels"]
    regression_pruning = regression_cfg_content["pruning"]

    density_results = density_cfg_content["density_results"]

    result_graphs = validation_ap(gt_events=gt_events, all_series_ids=all_series_ids,
                  selected_density_folders=density_results,
                  selected_regression_folders=regression_kernels,
                  cutoff=0.0, regression_pruning=regression_pruning,

                  binning_cutoffs=binning_cutoffs,

                  exclude_bad_segmentations=True)

    result_graphs_noexclude = validation_ap(gt_events=gt_events, all_series_ids=all_series_ids,
                                  selected_density_folders=density_results,
                                  selected_regression_folders=regression_kernels,
                                  cutoff=0.0, regression_pruning=regression_pruning,

                                  binning_cutoffs=binning_cutoffs,

                                  exclude_bad_segmentations=False)

    return {
        "result": result_graphs,
        "result_noexclude": result_graphs_noexclude
    }

if __name__ == "__main__":
    # load gt values and list of series ids
    all_series_ids = [filename.split(".")[0] for filename in os.listdir("./individual_train_series")]
    gt_events = convert_to_seriesid_events.get_events_per_seriesid()

    # file locations
    config_file = "./time_binning_bootstrapping/scores_maxmin_plot_config.json"
    output_folder = "./time_binning_bootstrapping/maxmin_risk"
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)

    # load files
    with open(config_file, "r") as f:
        config = json.load(f)

    # convert config
    regression_names = []
    entry = config["regressions"]
    regression_kernel_list = entry["regression_kernels"]
    regression_kernel_width = entry["kernel_width"]
    for k in range(len(regression_kernel_list)):
        regression_kernel_list[k] = convert_to_regression_folder(regression_kernel_list[k], kernel_width=regression_kernel_width,
                                                                 kernel_shape="gaussian")

    for entry_name in config["densities"]:
        entry = config["densities"][entry_name]
        density_list = entry["density_results"]
        for k in range(len(density_list)):
            density_list[k] = convert_to_density_folder(density_list[k])

    # generate general config
    run_configs = []
    for entry_name in config["densities"]:
        cfg = {
            "regression_cfg_content": config["regressions"],
            "density_cfg_content": config["densities"][entry_name]
        }
        run_configs.append(cfg)
    run_cuts = [20, 15, 10, 5, 3, 2, 1]

    # get scores now
    tasks = []
    for i in range(len(run_configs)):
        for j in range(len(run_cuts)):
            tasks.append(
                delayed(run_min_max_AP)(run_configs[i], all_series_ids,
                                    np.arange(0, 61, run_cuts[j], dtype=np.float32),
                                    gt_events)
            )
    results = Parallel(n_jobs=10, verbose=50)(tasks)

    # save the score to output folder
    result_gaps_matrix = np.zeros((len(config["densities"]), len(run_cuts)), dtype=np.float32)
    result_gains_matrix = np.zeros((len(config["densities"]), len(run_cuts)), dtype=np.float32)
    result_deviance_matrix = np.zeros((len(config["densities"]), len(run_cuts)), dtype=np.float32)
    result_noexcl_gaps_matrix = np.zeros((len(config["densities"]), len(run_cuts)), dtype=np.float32)
    result_noexcl_gains_matrix = np.zeros((len(config["densities"]), len(run_cuts)), dtype=np.float32)
    result_noexcl_deviance_matrix = np.zeros((len(config["densities"]), len(run_cuts)), dtype=np.float32)

    for i, entry_name in tqdm.tqdm(enumerate(config["densities"])):
        entry_name_formatted = entry_name.replace(" ", "_")
        for j in range(len(run_cuts)):
            cut = run_cuts[j]
            maxmin_risk_info = results[i * len(run_cuts) + j]

            out_distribution_file = os.path.join(output_folder, "{}_cut{}_distribution.png".format(entry_name_formatted, cut))
            out_min_distribution_file = os.path.join(output_folder, "{}_cut{}_distribution_min.png".format(entry_name_formatted, cut))
            out_max_distribution_file = os.path.join(output_folder, "{}_cut{}_distribution_max.png".format(entry_name_formatted, cut))
            out_mean_distribution_file = os.path.join(output_folder, "{}_cut{}_distribution_mean.png".format(entry_name_formatted, cut))

            out_noexcl_distribution_file = os.path.join(output_folder, "{}_cut{}_noexcl_distribution.png".format(entry_name_formatted, cut))
            out_noexcl_min_distribution_file = os.path.join(output_folder, "{}_cut{}_noexcl_distribution_min.png".format(entry_name_formatted, cut))
            out_noexcl_max_distribution_file = os.path.join(output_folder, "{}_cut{}_noexcl_distribution_max.png".format(entry_name_formatted, cut))
            out_noexcl_mean_distribution_file = os.path.join(output_folder, "{}_cut{}_noexcl_distribution_mean.png".format(entry_name_formatted, cut))

            # plot the distributions
            titlestr = "{} Cut {} (Onset: {}, Wakeup: {})".format(entry_name, cut, maxmin_risk_info["result"]["onset_mAP"], maxmin_risk_info["result"]["wakeup_mAP"])
            plot_ap_curve(title=titlestr,
                          out_file=out_distribution_file,
                          ap_onset_precisions=maxmin_risk_info["result"]["onset_precisions"],
                          ap_onset_recalls=maxmin_risk_info["result"]["onset_recalls"],
                          ap_onset_average_precisions=maxmin_risk_info["result"]["onset_aps"],
                          ap_wakeup_precisions=maxmin_risk_info["result"]["wakeup_precisions"],
                          ap_wakeup_recalls=maxmin_risk_info["result"]["wakeup_recalls"],
                          ap_wakeup_average_precisions=maxmin_risk_info["result"]["wakeup_aps"])
            titlestr = "{} Cut {} Min (Onset Gap: {}, Wakeup Gap: {})".format(entry_name, cut,
                                                                 maxmin_risk_info["result"]["onset_mAP"] - maxmin_risk_info["result"]["onset_min_mAP"],
                                                                 maxmin_risk_info["result"]["wakeup_mAP"] - maxmin_risk_info["result"]["wakeup_min_mAP"])
            plot_ap_curve(title=titlestr,
                          out_file=out_min_distribution_file,
                          ap_onset_precisions=maxmin_risk_info["result"]["onset_min_precisions"],
                          ap_onset_recalls=maxmin_risk_info["result"]["onset_min_recalls"],
                          ap_onset_average_precisions=maxmin_risk_info["result"]["onset_min_aps"],
                          ap_wakeup_precisions=maxmin_risk_info["result"]["wakeup_min_precisions"],
                          ap_wakeup_recalls=maxmin_risk_info["result"]["wakeup_min_recalls"],
                          ap_wakeup_average_precisions=maxmin_risk_info["result"]["wakeup_min_aps"])
            titlestr = "{} Cut {} Max (Onset Gain: {}, Wakeup Gain: {})".format(entry_name, cut,
                                                                          maxmin_risk_info["result"]["onset_max_mAP"] - maxmin_risk_info["result"]["onset_mAP"],
                                                                          maxmin_risk_info["result"]["wakeup_max_mAP"] - maxmin_risk_info["result"]["wakeup_mAP"])
            plot_ap_curve(title=titlestr,
                          out_file=out_max_distribution_file,
                          ap_onset_precisions=maxmin_risk_info["result"]["onset_max_precisions"],
                          ap_onset_recalls=maxmin_risk_info["result"]["onset_max_recalls"],
                          ap_onset_average_precisions=maxmin_risk_info["result"]["onset_max_aps"],
                          ap_wakeup_precisions=maxmin_risk_info["result"]["wakeup_max_precisions"],
                          ap_wakeup_recalls=maxmin_risk_info["result"]["wakeup_max_recalls"],
                          ap_wakeup_average_precisions=maxmin_risk_info["result"]["wakeup_max_aps"])
            titlestr = "{} Cut {} Max (Onset Deviance: {}, Wakeup Deviance: {})".format(entry_name, cut,
                                                                          maxmin_risk_info["result"]["onset_mean_mAP"] - maxmin_risk_info["result"]["onset_mAP"],
                                                                          maxmin_risk_info["result"]["wakeup_mean_mAP"] - maxmin_risk_info["result"]["wakeup_mAP"])
            plot_ap_curve(title=titlestr,
                          out_file=out_mean_distribution_file,
                          ap_onset_precisions=maxmin_risk_info["result"]["onset_mean_precisions"],
                          ap_onset_recalls=maxmin_risk_info["result"]["onset_mean_recalls"],
                          ap_onset_average_precisions=maxmin_risk_info["result"]["onset_mean_aps"],
                          ap_wakeup_precisions=maxmin_risk_info["result"]["wakeup_mean_precisions"],
                          ap_wakeup_recalls=maxmin_risk_info["result"]["wakeup_mean_recalls"],
                          ap_wakeup_average_precisions=maxmin_risk_info["result"]["wakeup_mean_aps"])



            # plot the noexcl distributions
            titlestr = "{} Cut {} (Onset: {}, Wakeup: {})".format(entry_name, cut, maxmin_risk_info["result_noexclude"]["onset_mAP"], maxmin_risk_info["result_noexclude"]["wakeup_mAP"])
            plot_ap_curve(title=titlestr,
                          out_file=out_noexcl_distribution_file,
                          ap_onset_precisions=maxmin_risk_info["result_noexclude"]["onset_precisions"],
                          ap_onset_recalls=maxmin_risk_info["result_noexclude"]["onset_recalls"],
                          ap_onset_average_precisions=maxmin_risk_info["result_noexclude"]["onset_aps"],
                          ap_wakeup_precisions=maxmin_risk_info["result_noexclude"]["wakeup_precisions"],
                          ap_wakeup_recalls=maxmin_risk_info["result_noexclude"]["wakeup_recalls"],
                          ap_wakeup_average_precisions=maxmin_risk_info["result_noexclude"]["wakeup_aps"])
            titlestr = "{} Cut {} Min (Onset Gap: {}, Wakeup Gap: {})".format(entry_name, cut,
                                                                 maxmin_risk_info["result_noexclude"]["onset_mAP"] - maxmin_risk_info["result_noexclude"]["onset_min_mAP"],
                                                                 maxmin_risk_info["result_noexclude"]["wakeup_mAP"] - maxmin_risk_info["result_noexclude"]["wakeup_min_mAP"])
            plot_ap_curve(title=titlestr,
                          out_file=out_noexcl_min_distribution_file,
                          ap_onset_precisions=maxmin_risk_info["result_noexclude"]["onset_min_precisions"],
                          ap_onset_recalls=maxmin_risk_info["result_noexclude"]["onset_min_recalls"],
                          ap_onset_average_precisions=maxmin_risk_info["result_noexclude"]["onset_min_aps"],
                          ap_wakeup_precisions=maxmin_risk_info["result_noexclude"]["wakeup_min_precisions"],
                          ap_wakeup_recalls=maxmin_risk_info["result_noexclude"]["wakeup_min_recalls"],
                          ap_wakeup_average_precisions=maxmin_risk_info["result_noexclude"]["wakeup_min_aps"])
            titlestr = "{} Cut {} Max (Onset Gain: {}, Wakeup Gain: {})".format(entry_name, cut,
                                                                          maxmin_risk_info["result_noexclude"]["onset_max_mAP"] - maxmin_risk_info["result_noexclude"]["onset_mAP"],
                                                                          maxmin_risk_info["result_noexclude"]["wakeup_max_mAP"] - maxmin_risk_info["result_noexclude"]["wakeup_mAP"])
            plot_ap_curve(title=titlestr,
                          out_file=out_noexcl_max_distribution_file,
                          ap_onset_precisions=maxmin_risk_info["result_noexclude"]["onset_max_precisions"],
                          ap_onset_recalls=maxmin_risk_info["result_noexclude"]["onset_max_recalls"],
                          ap_onset_average_precisions=maxmin_risk_info["result_noexclude"]["onset_max_aps"],
                          ap_wakeup_precisions=maxmin_risk_info["result_noexclude"]["wakeup_max_precisions"],
                          ap_wakeup_recalls=maxmin_risk_info["result_noexclude"]["wakeup_max_recalls"],
                          ap_wakeup_average_precisions=maxmin_risk_info["result_noexclude"]["wakeup_max_aps"])
            titlestr = "{} Cut {} Max (Onset Deviance: {}, Wakeup Deviance: {})".format(entry_name, cut,
                                                                          maxmin_risk_info["result_noexclude"]["onset_mean_mAP"] - maxmin_risk_info["result_noexclude"]["onset_mAP"],
                                                                          maxmin_risk_info["result_noexclude"]["wakeup_mean_mAP"] - maxmin_risk_info["result_noexclude"]["wakeup_mAP"])
            plot_ap_curve(title=titlestr,
                          out_file=out_noexcl_mean_distribution_file,
                          ap_onset_precisions=maxmin_risk_info["result_noexclude"]["onset_mean_precisions"],
                          ap_onset_recalls=maxmin_risk_info["result_noexclude"]["onset_mean_recalls"],
                          ap_onset_average_precisions=maxmin_risk_info["result_noexclude"]["onset_mean_aps"],
                          ap_wakeup_precisions=maxmin_risk_info["result_noexclude"]["wakeup_mean_precisions"],
                          ap_wakeup_recalls=maxmin_risk_info["result_noexclude"]["wakeup_mean_recalls"],
                          ap_wakeup_average_precisions=maxmin_risk_info["result_noexclude"]["wakeup_mean_aps"])

            mean_result_gap = (maxmin_risk_info["result"]["onset_mAP"] - maxmin_risk_info["result"]["onset_min_mAP"] +
                                maxmin_risk_info["result"]["wakeup_mAP"] - maxmin_risk_info["result"]["wakeup_min_mAP"]) / 2
            mean_result_gain = (maxmin_risk_info["result"]["onset_max_mAP"] - maxmin_risk_info["result"]["onset_mAP"] +
                                maxmin_risk_info["result"]["wakeup_max_mAP"] - maxmin_risk_info["result"]["wakeup_mAP"]) / 2
            mean_result_deviance = (maxmin_risk_info["result"]["onset_mean_mAP"] - maxmin_risk_info["result"]["onset_mAP"] +
                                    maxmin_risk_info["result"]["wakeup_mean_mAP"] - maxmin_risk_info["result"]["wakeup_mAP"]) / 2
            mean_result_noexcl_gap = (maxmin_risk_info["result_noexclude"]["onset_mAP"] - maxmin_risk_info["result_noexclude"]["onset_min_mAP"] +
                                maxmin_risk_info["result_noexclude"]["wakeup_mAP"] - maxmin_risk_info["result_noexclude"]["wakeup_min_mAP"]) / 2
            mean_result_noexcl_gain = (maxmin_risk_info["result_noexclude"]["onset_max_mAP"] - maxmin_risk_info["result_noexclude"]["onset_mAP"] +
                                maxmin_risk_info["result_noexclude"]["wakeup_max_mAP"] - maxmin_risk_info["result_noexclude"]["wakeup_mAP"]) / 2
            mean_result_noexcl_deviance = (maxmin_risk_info["result_noexclude"]["onset_mean_mAP"] - maxmin_risk_info["result_noexclude"]["onset_mAP"] +
                                            maxmin_risk_info["result_noexclude"]["wakeup_mean_mAP"] - maxmin_risk_info["result_noexclude"]["wakeup_mAP"]) / 2

            # write to matrix
            result_gaps_matrix[i, j] = mean_result_gap
            result_gains_matrix[i, j] = mean_result_gain
            result_deviance_matrix[i, j] = mean_result_deviance
            result_noexcl_gaps_matrix[i, j] = mean_result_noexcl_gap
            result_noexcl_gains_matrix[i, j] = mean_result_noexcl_gain
            result_noexcl_deviance_matrix[i, j] = mean_result_noexcl_deviance

    # save gaps gains matrix to pandas dataframe
    result_gaps_df = pd.DataFrame(result_gaps_matrix, index=list(config["densities"].keys()), columns=run_cuts)
    result_gains_df = pd.DataFrame(result_gains_matrix, index=list(config["densities"].keys()), columns=run_cuts)
    result_deviance_df = pd.DataFrame(result_deviance_matrix, index=list(config["densities"].keys()), columns=run_cuts)
    result_noexcl_gaps_df = pd.DataFrame(result_noexcl_gaps_matrix, index=list(config["densities"].keys()), columns=run_cuts)
    result_noexcl_gains_df = pd.DataFrame(result_noexcl_gains_matrix, index=list(config["densities"].keys()), columns=run_cuts)
    result_noexcl_deviance_df = pd.DataFrame(result_noexcl_deviance_matrix, index=list(config["densities"].keys()), columns=run_cuts)

    # save to csv
    result_gaps_df.to_csv(os.path.join(output_folder, "gaps.csv"))
    result_gains_df.to_csv(os.path.join(output_folder, "gains.csv"))
    result_deviance_df.to_csv(os.path.join(output_folder, "deviance.csv"))
    result_noexcl_gaps_df.to_csv(os.path.join(output_folder, "noexcl_gaps.csv"))
    result_noexcl_gains_df.to_csv(os.path.join(output_folder, "noexcl_gains.csv"))
    result_noexcl_deviance_df.to_csv(os.path.join(output_folder, "noexcl_deviance.csv"))

    print("All done!")

