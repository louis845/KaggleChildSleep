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

def get_scores(all_series_ids,
              selected_density_folders: list[str], selected_regression_folders: list[str],

              regression_pruning):
    selected_series_ids = all_series_ids

    combined_onset_probas = []
    combined_wakeup_probas = []

    for series_id in selected_series_ids:
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

        combined_onset_probas.append(onset_locs_all_probas)
        combined_wakeup_probas.append(wakeup_locs_all_probas)

    return np.concatenate(combined_onset_probas + combined_wakeup_probas, axis=0)


def run_score_convert(config, all_series_ids):
    regression_cfg_content = config["regression_cfg_content"]
    density_cfg_content = config["density_cfg_content"]

    regression_kernels = regression_cfg_content["regression_kernels"]
    regression_pruning = regression_cfg_content["pruning"]

    density_results = density_cfg_content["density_results"]

    return get_scores(all_series_ids=all_series_ids,
                  selected_density_folders=density_results,
                  selected_regression_folders=regression_kernels,
                  regression_pruning=regression_pruning)

if __name__ == "__main__":
    # load gt values and list of series ids
    all_series_ids = [filename.split(".")[0] for filename in os.listdir("./individual_train_series")]

    # file locations
    config_file = "./time_binning_bootstrapping/scores_distribution_config.json"
    output_folder = "./time_binning_bootstrapping/scores_distribution"
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

    # get scores now
    tasks = [delayed(run_score_convert)(entry, all_series_ids) for entry in run_configs]
    results = Parallel(n_jobs=10, verbose=50)(tasks)

    # save the score to output folder
    for i, entry_name in tqdm.tqdm(enumerate(config["densities"])):
        entry_name_formatted = entry_name.replace(" ", "_")
        scores_distribution = results[i]

        out_numpy_file = os.path.join(output_folder, "probas_{}.npy".format(entry_name_formatted))
        out_distribution_file = os.path.join(output_folder, "distribution_{}.png".format(entry_name_formatted))

        np.save(out_numpy_file, scores_distribution)
        percentiles = np.linspace(0, 100, num=101)
        percentile_values = np.percentile(scores_distribution, percentiles)

        plt.figure(figsize=(16, 12))
        plt.plot(percentiles, percentile_values, marker=".", linestyle="-")
        plt.xlabel("Percentiles")
        plt.ylabel("Value")
        plt.title("Cumulative Distribution Plot")
        plt.xlim(0, 100)
        plt.savefig(out_distribution_file)

    print("All done!")
