import sys
import os
import json
from typing import Iterator

import numpy as np
import h5py
import pandas as pd
import tqdm
from joblib import Parallel, delayed

import postprocessing
import metrics_ap
import convert_to_seriesid_events
import model_event_density_unet
import bad_series_list


validation_AP_tolerances = [1, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30][::-1]

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
                  cutoff, augmentation, augmentation_cutoff, matrix_values_pruning,

                  regression_pruning, regression_postpruning,

                  exclude_bad_segmentations: bool=True):
    selected_series_ids = all_series_ids

    ap_onset_metrics = [metrics_ap.EventMetrics(name="", tolerance=tolerance * 12) for tolerance in validation_AP_tolerances]
    ap_wakeup_metrics = [metrics_ap.EventMetrics(name="", tolerance=tolerance * 12) for tolerance in validation_AP_tolerances]

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

        # postprune
        if regression_postpruning > 0:
            if len(preds_locs_onset) > 0:
                onset_prune_keeps = postprocessing.prune(preds_locs_onset, onset_locs_all_probas,
                                                         pruning_radius=regression_postpruning, return_idx=True)
                preds_locs_onset = preds_locs_onset[onset_prune_keeps]
                onset_locs_all_probas = onset_locs_all_probas[onset_prune_keeps]
            if len(preds_locs_wakeup) > 0:
                wakeup_prune_keeps = postprocessing.prune(preds_locs_wakeup, wakeup_locs_all_probas,
                                                          pruning_radius=regression_postpruning, return_idx=True)
                preds_locs_wakeup = preds_locs_wakeup[wakeup_prune_keeps]
                wakeup_locs_all_probas = wakeup_locs_all_probas[wakeup_prune_keeps]

        # augment and restrict the probas
        if augmentation:
            preds_locs_onset, onset_locs_all_probas = postprocessing.get_augmented_predictions_density(preds_locs_onset,
                                                            onset_kernel_vals, onset_locs_all_probas, cutoff_thresh=augmentation_cutoff)
            preds_locs_wakeup, wakeup_locs_all_probas = postprocessing.get_augmented_predictions_density(preds_locs_wakeup,
                                                            wakeup_kernel_vals, wakeup_locs_all_probas, cutoff_thresh=augmentation_cutoff)

        # prune using matrix values
        if matrix_values_pruning:
            matrix_values = np.load(os.path.join("./data_matrix_profile", "{}.npy".format(series_id)))
            preds_locs_onset_restrict = postprocessing.prune_matrix_profile(preds_locs_onset, matrix_values,
                                                                            return_idx=True)
            preds_locs_wakeup_restrict = postprocessing.prune_matrix_profile(preds_locs_wakeup, matrix_values,
                                                                             return_idx=True)

            preds_locs_onset = preds_locs_onset[preds_locs_onset_restrict]
            onset_locs_all_probas = onset_locs_all_probas[preds_locs_onset_restrict]
            preds_locs_wakeup = preds_locs_wakeup[preds_locs_wakeup_restrict]
            wakeup_locs_all_probas = wakeup_locs_all_probas[preds_locs_wakeup_restrict]

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

    # compute average precision
    ap_onset_average_precisions, ap_wakeup_average_precisions = [], []
    for ap_onset_metric, ap_wakeup_metric in zip(ap_onset_metrics, ap_wakeup_metrics):
        ap_onset_precision, ap_onset_recall, ap_onset_average_precision, ap_onset_proba = ap_onset_metric.get()
        ap_wakeup_precision, ap_wakeup_recall, ap_wakeup_average_precision, ap_wakeup_proba = ap_wakeup_metric.get()
        ap_onset_average_precisions.append(ap_onset_average_precision)
        ap_wakeup_average_precisions.append(ap_wakeup_average_precision)

    onset_mAP = np.mean(ap_onset_average_precisions)
    wakeup_mAP = np.mean(ap_wakeup_average_precisions)

    general_mAP = (onset_mAP + wakeup_mAP) / 2

    return {"onset_mAP": onset_mAP,
            "wakeup_mAP": wakeup_mAP,
            "general_mAP": general_mAP}

def run_validation_AP(config, all_series_ids, per_series_id_events):
    regression_cfg_content = config["regression_cfg_content"]
    density_cfg_content = config["density_cfg_content"]

    regression_kernels = regression_cfg_content["regression_kernels"]
    regression_pruning = regression_cfg_content["pruning"]
    regression_postpruning = regression_cfg_content["postpruning"]
    augmentation = regression_cfg_content["augmentation"]
    augmentation_cutoff = regression_cfg_content["augmentation_cutoff"]

    density_results = density_cfg_content["density_results"]

    return validation_ap(gt_events=per_series_id_events, all_series_ids=all_series_ids,
                  selected_density_folders=density_results,
                  selected_regression_folders=regression_kernels,
                  cutoff=0.0, augmentation=augmentation, augmentation_cutoff=augmentation_cutoff, matrix_values_pruning=False,

                  regression_pruning=regression_pruning, regression_postpruning=regression_postpruning)

if __name__ == "__main__":
    # load gt values and list of series ids
    all_series_ids = [filename.split(".")[0] for filename in os.listdir("./individual_train_series")]
    per_series_id_events = convert_to_seriesid_events.get_events_per_seriesid()


    # file locations
    config_file = "./inference_density_combined_statistics/inference_density_grid_search_config.json"
    out_results_file = "./inference_density_combined_statistics/inference_density_grid_search_results.csv"
    out_onset_results_file = "./inference_density_combined_statistics/inference_density_grid_search_onset_results.csv"
    out_wakeup_results_file = "./inference_density_combined_statistics/inference_density_grid_search_wakeup_results.csv"

    # load files
    with open(config_file, "r") as f:
        config = json.load(f)
    if os.path.isfile(out_results_file):
        existing_results = pd.read_csv(out_results_file, index_col=0)
        existing_onset_results = pd.read_csv(out_onset_results_file, index_col=0)
        existing_wakeup_results = pd.read_csv(out_wakeup_results_file, index_col=0)
    else:
        existing_results, existing_onset_results, existing_wakeup_results = None, None, None

    # convert config
    regression_names = []
    for entry_name in config["regressions"]:
        entry = config["regressions"][entry_name]
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

    # convert the existing results
    if existing_results is not None:
        existing_regressions = list(existing_results.index)
        existing_densities = list(existing_results.columns)
        assert set(existing_regressions).issubset(set(config["regressions"].keys()))
        assert set(existing_densities).issubset(set(config["densities"].keys()))

        write_matrix_mAP = np.full((len(config["regressions"]), len(config["densities"])), dtype=np.float32, fill_value=np.nan)
        write_matrix_mAP[:len(existing_regressions), :len(existing_densities)] = existing_results.values
        write_onset_matrix_mAP = np.full((len(config["regressions"]), len(config["densities"])), dtype=np.float32, fill_value=np.nan)
        write_onset_matrix_mAP[:len(existing_regressions), :len(existing_densities)] = existing_onset_results.values
        write_wakeup_matrix_mAP = np.full((len(config["regressions"]), len(config["densities"])), dtype=np.float32, fill_value=np.nan)
        write_wakeup_matrix_mAP[:len(existing_regressions), :len(existing_densities)] = existing_wakeup_results.values

        write_columns = existing_densities + [density_name for density_name in config["densities"] if density_name not in existing_densities] # densities
        write_rows = existing_regressions + [regression_name for regression_name in config["regressions"] if regression_name not in existing_regressions] # regressions

        assert len(write_columns) == write_matrix_mAP.shape[1], "{} != {}".format(len(write_columns), write_matrix_mAP.shape[1])
        assert len(write_rows) == write_matrix_mAP.shape[0], "{} != {}".format(len(write_rows), write_matrix_mAP.shape[0])
    else:
        write_matrix_mAP = np.full((len(config["regressions"]), len(config["densities"])), dtype=np.float32, fill_value=np.nan)
        write_onset_matrix_mAP = np.full((len(config["regressions"]), len(config["densities"])), dtype=np.float32, fill_value=np.nan)
        write_wakeup_matrix_mAP = np.full((len(config["regressions"]), len(config["densities"])), dtype=np.float32, fill_value=np.nan)

        write_columns = list(config["densities"].keys()) # densities
        write_rows = list(config["regressions"].keys()) # regressions

    # generate the values to compute
    write_entries = []
    for i, regression_name in enumerate(write_rows):
        for j, density_name in enumerate(write_columns):
            if np.isnan(write_matrix_mAP[i, j]):
                write_entries.append({
                    "coordinates": (i, j),
                    "regression_name": regression_name,
                    "density_name": density_name,
                    "regression_cfg_content": config["regressions"][regression_name],
                    "density_cfg_content": config["densities"][density_name],
                })

    # generate the values
    tasks = [delayed(run_validation_AP)(entry, all_series_ids, per_series_id_events) for entry in write_entries]
    results = Parallel(n_jobs=10, verbose=50)(tasks)

    for k, result in enumerate(results):
        i, j = write_entries[k]["coordinates"]
        write_matrix_mAP[i, j] = result["general_mAP"]
        write_onset_matrix_mAP[i, j] = result["onset_mAP"]
        write_wakeup_matrix_mAP[i, j] = result["wakeup_mAP"]

    # write the results
    write_df = pd.DataFrame(data=write_matrix_mAP, index=write_rows, columns=write_columns)
    write_df.to_csv(out_results_file)
    write_onset_df = pd.DataFrame(data=write_onset_matrix_mAP, index=write_rows, columns=write_columns)
    write_onset_df.to_csv(out_onset_results_file)
    write_wakeup_df = pd.DataFrame(data=write_wakeup_matrix_mAP, index=write_rows, columns=write_columns)
    write_wakeup_df.to_csv(out_wakeup_results_file)

    print("All done!")
