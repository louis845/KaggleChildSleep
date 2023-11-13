import os
import argparse
import json

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

    iou_score_converter = model_event_unet.ProbasIOUScoreConverter(intersection_width=30 * 12,
                                                                   union_width=60 * 12, device=config.device)

    # get list of series ids
    series_ids = [x.split(".")[0] for x in os.listdir("individual_train_series")]

    # load options for statistics computation
    with open("./inference_combined_statistics/inference_combined_preds_options.json", "r") as f:
        options = json.load(f)

    for option in options:
        name = option["name"]
        conf_results = option["conf_results"]
        out_folder = option["out_folder"]

        conf_results_folders = [os.path.join("./inference_confidence_statistics/confidence_labels", conf_result)
                                 for conf_result in conf_results]

        for series_id in series_ids:
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

            # convert to IOU score
            onset_IOU_score = iou_score_converter.convert(ensembled_onset_preds)
            wakeup_IOU_score = iou_score_converter.convert(ensembled_wakeup_preds)

            # save predictions
            np.save(os.path.join(FOLDER, out_folder, "{}_onset.npy".format(series_id)), onset_IOU_score)
            np.save(os.path.join(FOLDER, out_folder, "{}_wakeup.npy".format(series_id)), wakeup_IOU_score)
