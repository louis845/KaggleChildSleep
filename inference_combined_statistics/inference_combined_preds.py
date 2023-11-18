import os
import argparse
import json
import shutil

import numpy as np
import torch
import tqdm

import model_event_unet
import config

FOLDER = "./inference_combined_statistics/combined_predictions"

if __name__ == "__main__":
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)

    parser = argparse.ArgumentParser("Inference script to combine regression and confidence events to form predictions.")
    config.add_argparse_arguments(parser)
    args = parser.parse_args()

    # initialize gpu
    config.parse_args(args)

    union_widths = [31, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 120]
    iou_score_converters = {}
    for width in union_widths:
        iou_score_converters[width] = model_event_unet.ProbasIOUScoreConverter(intersection_width=30 * 12,
                                                                       union_width=width * 12, device=config.device)

    # get list of series ids
    series_ids = [x.split(".")[0] for x in os.listdir("individual_train_series")]

    # load options for statistics computation
    with open("./inference_combined_statistics/inference_combined_preds_options.json", "r") as f:
        options = json.load(f)

    for option in options:
        name = option["name"]
        conf_results = option["conf_results"]
        out_folder = option["out_folder"]
        if os.path.isdir(os.path.join(FOLDER, out_folder)):
            continue
        os.mkdir(os.path.join(FOLDER, out_folder))
        for width in union_widths:
            os.mkdir(os.path.join(FOLDER, out_folder, "width{}".format(width)))

        conf_results_folders = [os.path.join("./inference_confidence_statistics/confidence_labels", conf_result)
                                 for conf_result in conf_results]

        print("Combining predictions with option {}...".format(name))

        for series_id in tqdm.tqdm(series_ids):
            ensembled_onset_preds = None
            ensembled_wakeup_preds = None

            # ensemble predictions from different confidence models
            for conf_results_folder in conf_results_folders:
                onset_conf_preds = np.load(os.path.join(conf_results_folder, "{}_onset.npy".format(series_id)))
                wakeup_conf_preds = np.load(os.path.join(conf_results_folder, "{}_wakeup.npy".format(series_id)))

                if ensembled_onset_preds is None:
                    ensembled_onset_preds = onset_conf_preds
                    ensembled_wakeup_preds = wakeup_conf_preds
                else:
                    ensembled_onset_preds = ensembled_onset_preds + onset_conf_preds
                    ensembled_wakeup_preds = ensembled_wakeup_preds + wakeup_conf_preds

            # average predictions
            ensembled_onset_preds = ensembled_onset_preds / len(conf_results_folders)
            ensembled_wakeup_preds = ensembled_wakeup_preds / len(conf_results_folders)

            for width in union_widths:
                # convert to IOU score
                onset_IOU_score = iou_score_converters[width].convert(ensembled_onset_preds)
                wakeup_IOU_score = iou_score_converters[width].convert(ensembled_wakeup_preds)

                # save predictions
                np.save(os.path.join(FOLDER, out_folder, "width{}".format(width), "{}_onset.npy".format(series_id)), onset_IOU_score)
                np.save(os.path.join(FOLDER, out_folder, "width{}".format(width), "{}_wakeup.npy".format(series_id)), wakeup_IOU_score)
