import gc
import time
import json

import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq
import tqdm

import competition_models
import kaggle_ap_detection
import os

def correct_time(secs: np.ndarray, mins: np.ndarray, hours: np.ndarray):
    daytime = secs.astype(np.int32) + mins.astype(np.int32) * 60 + hours.astype(np.int32) * 3600
    init_times = daytime - np.arange(len(daytime)) * 5 # 5 second intervals
    avg_start = np.median(init_times)
    daytime = avg_start + np.arange(len(daytime)) * 5
    secs = daytime % 60
    mins = (daytime // 60) % 60
    hours = (daytime // 3600) % 24
    return secs.astype(np.float32), mins.astype(np.float32), hours.astype(np.float32)

class CompetitionInference:

    def __init__(self, input_pq_file: str, models_callable: competition_models.CompetitionModels):
        self.input_pq_file = input_pq_file
        self.models_callable = models_callable

    def inference_all(self, out_file_path, use_matrix_profile_pruning=False,
                      log_debug=False, show_tqdm_bar=False, model_filter=None, cache=False):
        if cache:
            if not os.path.isdir("cache"):
                os.mkdir("cache")

        # Create output file
        out_file = open(out_file_path, "w")
        row_id = 0
        out_file.write("row_id,series_id,step,event,score\n")

        # Load only the "series_id" column
        series_id_df = pd.read_parquet(self.input_pq_file, columns=["series_id"])

        # Get unique "series_id" values
        series_ids = series_id_df["series_id"].unique()

        all_preprocessing_time = []
        all_inference_time = []
        all_avg_kernel_values_time = []
        all_avg_confidence_time = []
        all_matrix_profile_pruning_time = []
        all_first_postprocessing_time = []
        all_second_postprocessing_time = []

        # Iteratively read data for each "series_id"
        if show_tqdm_bar:
            series_ids = tqdm.tqdm(series_ids)
        ctime2 = time.time()
        for series_id in series_ids:
            ctime = time.time()
            if cache and os.path.isfile("cache/{}_anglez.npy".format(series_id)):
                accel_data_anglez = np.load("cache/{}_anglez.npy".format(series_id))
                accel_data_enmo = np.load("cache/{}_enmo.npy".format(series_id))
                secs_corr = np.load("cache/{}_secs.npy".format(series_id))
                mins_corr = np.load("cache/{}_mins.npy".format(series_id))
                hours_corr = np.load("cache/{}_hours.npy".format(series_id))
            else:
                df = pd.read_parquet(self.input_pq_file, columns=["step", "timestamp", "anglez", "enmo"],
                                     filters=[("series_id", "==", series_id)])

                accel_data_anglez = np.expand_dims(df["anglez"].to_numpy(dtype=np.float32) / 35.52, axis=0) # shape (1, T)
                accel_data_enmo = df["enmo"].to_numpy(dtype=np.float32) / 0.1018
                accel_data_enmo = np.expand_dims((accel_data_enmo ** 0.6 - 1) / 0.6, axis=0) # shape (1, T)
                timestamps = pd.to_datetime(df["timestamp"]).apply(lambda dt: dt.tz_localize(None))

                secs = timestamps.dt.second.to_numpy(dtype=np.float32)
                mins = timestamps.dt.minute.to_numpy(dtype=np.float32)
                hours = timestamps.dt.hour.to_numpy(dtype=np.float32)

                secs_corr, mins_corr, hours_corr = correct_time(secs, mins, hours)

                if cache:
                    np.save("cache/{}_anglez.npy".format(series_id), accel_data_anglez)
                    np.save("cache/{}_enmo.npy".format(series_id), accel_data_enmo)
                    np.save("cache/{}_secs.npy".format(series_id), secs_corr)
                    np.save("cache/{}_mins.npy".format(series_id), mins_corr)
                    np.save("cache/{}_hours.npy".format(series_id), hours_corr)
            preprocessing_time = time.time() - ctime

            # Call the series inference callable
            ctime = time.time()
            filtered_subset = None
            if model_filter is not None:
                filtered_subset = model_filter(series_id)
            onset_locs, onset_IOU_probas, wakeup_locs, wakeup_IOU_probas, time_elapsed_performance_metrics =\
                self.models_callable.run_inference(series_id, accel_data_anglez, accel_data_enmo,
                                                   secs_corr, mins_corr, hours_corr,
                                                   models_subset=filtered_subset, use_matrix_profile_pruning=use_matrix_profile_pruning)
            inference_time = time.time() - ctime

            # Save the results
            if len(onset_locs) > 0:
                for i in range(len(onset_locs)):
                    out_file.write("{},{},{},{},{}\n".format(row_id, series_id, onset_locs[i], "onset", onset_IOU_probas[i]))
                    row_id += 1
            out_file.flush()
            if len(wakeup_locs) > 0:
                for i in range(len(wakeup_locs)):
                    out_file.write("{},{},{},{},{}\n".format(row_id, series_id, wakeup_locs[i], "wakeup", wakeup_IOU_probas[i]))
                    row_id += 1
            out_file.flush()

            gc.collect()

            if log_debug:
                print("---------- Completed inference for series_id: {} ----------".format(series_id))
                print("Preprocessing Time: {}".format(preprocessing_time))
                print("Inference Time: {}".format(inference_time))
                print("Kernel Values Time: {}".format(time_elapsed_performance_metrics["kernel_values_computation_time"]))
                print("First postprocessing Time: {}".format(time_elapsed_performance_metrics["first_postprocessing_time"]))
                print("Matrix profile pruning Time: {}".format(time_elapsed_performance_metrics["matrix_profile_pruning_time"]))
                print("Confidence Time: {}".format(time_elapsed_performance_metrics["confidence_computation_time"]))
                print("Second postprocessing Time: {}".format(time_elapsed_performance_metrics["second_postprocessing_time"]))
                print("Avg Kernel Values Time: {}".format(time_elapsed_performance_metrics["avg_kernel_values_time"]))
                print("Avg Confidence Time: {}".format(time_elapsed_performance_metrics["avg_confidence_time"]))

                all_preprocessing_time.append(preprocessing_time)
                all_inference_time.append(inference_time)
                all_avg_kernel_values_time.append(time_elapsed_performance_metrics["avg_kernel_values_time"])
                all_avg_confidence_time.append(time_elapsed_performance_metrics["avg_confidence_time"])
                all_matrix_profile_pruning_time.append(time_elapsed_performance_metrics["matrix_profile_pruning_time"])
                all_first_postprocessing_time.append(time_elapsed_performance_metrics["first_postprocessing_time"])
                all_second_postprocessing_time.append(time_elapsed_performance_metrics["second_postprocessing_time"])

        print("---------------------------------------- All completed ----------------------------------------")
        print("Total Time: {}".format(time.time() - ctime2))
        print("Avg Preprocessing Time: {}".format(np.mean(all_preprocessing_time)))
        print("Avg Inference Time: {}".format(np.mean(all_inference_time)))
        print("Avg Kernel Values Time: {}".format(np.mean(all_avg_kernel_values_time)))
        print("Avg Confidence Time: {}".format(np.mean(all_avg_confidence_time)))
        print("Avg Matrix profile pruning Time: {}".format(np.mean(all_matrix_profile_pruning_time)))
        print("Avg First postprocessing Time: {}".format(np.mean(all_first_postprocessing_time)))
        print("Avg Second postprocessing Time: {}".format(np.mean(all_second_postprocessing_time)))

        out_file.close()

if __name__ == "__main__":
    input_pq_file = "../data/train_series.parquet"
    input_models_root_dir = "../final_models"
    input_model_cfg_file = "competition_models_cfg.json"
    out_file_path = "submission.csv"

    # For 5-fold like effect
    def load_fold(fold):
        assert 1 <= fold <= 5, "Fold must be between 1 and 5"
        fold_json_file = "../folds/fold_{}_val_5cv.json".format(fold)
        with open(fold_json_file, "r") as f:
            fold_json = json.load(f)
        return fold_json["dataset"]
    folds_map = {}
    for k in range(1, 6):
        for series_id in load_fold(k):
            folds_map[series_id] = k

    with open(input_model_cfg_file, "r") as f:
        model_cfg = json.load(f)
    all_models = [x["model_name"] for x in model_cfg["regression_models"]] + [x["model_name"] for x in model_cfg["confidence_models"]]
    print("All models: {}".format(all_models))
    def model_filter_func(series_id):
        belonging_fold = folds_map[series_id]
        return [model_name for model_name in all_models if "fold{}".format(str(belonging_fold)) in model_name]

    # Load the config and models
    models_callable = competition_models.CompetitionModels(model_config_file=input_model_cfg_file,
                                                           models_root_dir=input_models_root_dir,
                                                           device=torch.device("cuda:0"))
    models_callable.load_models()
    models_callable.set_parameters(regression_cutoff=0.01,
                                   regression_pruning=72,
                                   confidence_cutoff=0.01,
                                   confidence_aug_cutoff=1.0,
                                   matrix_profile_downsampling_rate=12)

    # Load the data and run inference
    competition_inference = CompetitionInference(input_pq_file=input_pq_file,
                                                    models_callable=models_callable)
    competition_inference.inference_all(out_file_path=out_file_path, log_debug=True, show_tqdm_bar=True, model_filter=model_filter_func,
                                        cache=True, use_matrix_profile_pruning=False)

    # Done. Garbage collect
    gc.collect()

    # Run the evaluation script
    tolerances = [1, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30]
    tolerances = [tolerance * 12 for tolerance in tolerances]
    solution = pd.read_csv("../data/train_events.csv")
    solution = solution.dropna()
    submission = pd.read_csv(out_file_path)
    """ctime = time.time()
    score = kaggle_ap_detection.score(solution, submission,
                                      tolerances={"onset": tolerances,
                                                  "wakeup": tolerances},
                                      series_id_column_name="series_id",
                                      time_column_name="step",
                                      event_column_name="event",
                                      score_column_name="score",
                                      use_scoring_intervals=False)
    print("Evaluation Time: {}".format(time.time() - ctime))

    print("Score: {}".format(score)) # this has data leakage since there is no train/test split. Just for bug catching"""


    """
    Output:
    Evaluation Time: 1083.2370390892029
    Score: 0.8465210245523218
    """


    # for debugging and comparison with the kaggle script
    """Onset Average Precision: 0.8202315361864594
Wakeup Average Precision: 0.8159815777928676
Combined Average Precision: 0.8181065569896635"""

    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import metrics_ap
    import convert_to_seriesid_events
    gt_events = convert_to_seriesid_events.get_events_per_seriesid("../data/train_events.csv", "../individual_train_series")

    predicted_events = {}
    all_series_ids = [filename.split(".")[0] for filename in os.listdir("../individual_train_series")]
    for series_id in all_series_ids:
        series_submission = submission.loc[submission["series_id"] == series_id]
        series_onset = series_submission.loc[series_submission["event"] == "onset"]
        series_wakeup = series_submission.loc[series_submission["event"] == "wakeup"]
        predicted_events[series_id] = {
            "onset": series_onset["step"].to_numpy(dtype=np.int32),
            "onset_proba": series_onset["score"].to_numpy(dtype=np.float32),
            "wakeup": series_wakeup["step"].to_numpy(dtype=np.int32),
            "wakeup_proba": series_wakeup["score"].to_numpy(dtype=np.float32)
        }


    ap_onset_metrics = [metrics_ap.EventMetrics(name="", tolerance=tolerance) for tolerance in tolerances]
    ap_wakeup_metrics = [metrics_ap.EventMetrics(name="", tolerance=tolerance) for tolerance in tolerances]

    for series_id in all_series_ids:
        # get the ground truth
        gt_onset_locs = gt_events[series_id]["onset"]
        gt_wakeup_locs = gt_events[series_id]["wakeup"]

        # get the predictions
        preds_onset = predicted_events[series_id]["onset"]
        preds_wakeup = predicted_events[series_id]["wakeup"]
        onset_IOU_probas = predicted_events[series_id]["onset_proba"]
        wakeup_IOU_probas = predicted_events[series_id]["wakeup_proba"]

        # add info
        for ap_onset_metric, ap_wakeup_metric in zip(ap_onset_metrics, ap_wakeup_metrics):
            ap_onset_metric.add(pred_locs=preds_onset, pred_probas=onset_IOU_probas, gt_locs=gt_onset_locs)
            ap_wakeup_metric.add(pred_locs=preds_wakeup, pred_probas=wakeup_IOU_probas, gt_locs=gt_wakeup_locs)

    # compute average precision
    ap_onset_precisions, ap_onset_recalls, ap_onset_average_precisions, ap_onset_probas = [], [], [], []
    ap_wakeup_precisions, ap_wakeup_recalls, ap_wakeup_average_precisions, ap_wakeup_probas = [], [], [], []
    for ap_onset_metric, ap_wakeup_metric in zip(ap_onset_metrics, ap_wakeup_metrics):
        ap_onset_precision, ap_onset_recall, ap_onset_average_precision, ap_onset_proba = ap_onset_metric.get()
        ap_wakeup_precision, ap_wakeup_recall, ap_wakeup_average_precision, ap_wakeup_proba = ap_wakeup_metric.get()
        ap_onset_precisions.append(ap_onset_precision)
        ap_onset_recalls.append(ap_onset_recall)
        ap_onset_average_precisions.append(ap_onset_average_precision)
        ap_onset_probas.append(ap_onset_proba)
        ap_wakeup_precisions.append(ap_wakeup_precision)
        ap_wakeup_recalls.append(ap_wakeup_recall)
        ap_wakeup_average_precisions.append(ap_wakeup_average_precision)
        ap_wakeup_probas.append(ap_wakeup_proba)

    # print the mean of average precisions
    print("Onset Average Precision: {}".format(np.mean(ap_onset_average_precisions)))
    print("Wakeup Average Precision: {}".format(np.mean(ap_wakeup_average_precisions)))
    print("Combined Average Precision: {}".format((np.mean(ap_onset_average_precisions) + np.mean(ap_wakeup_average_precisions)) / 2))