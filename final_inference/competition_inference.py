import gc
import time

import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq
import tqdm

import competition_models
import kaggle_ap_detection

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

    def inference_all(self, out_file_path, log_debug=False, show_tqdm_bar=False):
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
        all_first_postprocessing_time = []
        all_second_postprocessing_time = []

        # Iteratively read data for each "series_id"
        if show_tqdm_bar:
            series_ids = tqdm.tqdm(series_ids)
        for series_id in series_ids:
            ctime = time.time()
            df = pd.read_parquet(self.input_pq_file, columns=["step", "timestamp", "anglez"],
                                 filters=[("series_id", "==", series_id)])

            accel_data = np.expand_dims(df["anglez"].to_numpy(dtype=np.float32) / 35.52, axis=0) # shape (1, T)
            timestamps = pd.to_datetime(df["timestamp"]).apply(lambda dt: dt.tz_localize(None))

            secs = timestamps.dt.second.to_numpy(dtype=np.float32)
            mins = timestamps.dt.minute.to_numpy(dtype=np.float32)
            hours = timestamps.dt.hour.to_numpy(dtype=np.float32)

            secs_corr, mins_corr, hours_corr = correct_time(secs, mins, hours)
            preprocessing_time = time.time() - ctime

            # Call the series inference callable
            ctime = time.time()
            onset_locs, onset_IOU_probas, wakeup_locs, wakeup_IOU_probas, time_elapsed_performance_metrics =\
                self.models_callable.run_inference(series_id, accel_data, secs_corr, mins_corr, hours_corr)
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
                print("Confidence Time: {}".format(time_elapsed_performance_metrics["confidence_computation_time"]))
                print("Second postprocessing Time: {}".format(time_elapsed_performance_metrics["second_postprocessing_time"]))
                print("Avg Kernel Values Time: {}".format(time_elapsed_performance_metrics["avg_kernel_values_time"]))
                print("Avg Confidence Time: {}".format(time_elapsed_performance_metrics["avg_confidence_time"]))

                all_preprocessing_time.append(preprocessing_time)
                all_inference_time.append(inference_time)
                all_avg_kernel_values_time.append(time_elapsed_performance_metrics["avg_kernel_values_time"])
                all_avg_confidence_time.append(time_elapsed_performance_metrics["avg_confidence_time"])
                all_first_postprocessing_time.append(time_elapsed_performance_metrics["first_postprocessing_time"])
                all_second_postprocessing_time.append(time_elapsed_performance_metrics["second_postprocessing_time"])

        print("---------------------------------------- All completed ----------------------------------------")
        print("Avg Preprocessing Time: {}".format(np.mean(all_preprocessing_time)))
        print("Avg Inference Time: {}".format(np.mean(all_inference_time)))
        print("Avg Kernel Values Time: {}".format(np.mean(all_avg_kernel_values_time)))
        print("Avg Confidence Time: {}".format(np.mean(all_avg_confidence_time)))
        print("Avg First postprocessing Time: {}".format(np.mean(all_first_postprocessing_time)))
        print("Avg Second postprocessing Time: {}".format(np.mean(all_second_postprocessing_time)))

        out_file.close()

if __name__ == "__main__":
    input_pq_file = "../data/train_series.parquet"
    input_models_root_dir = "../final_models"
    input_model_cfg_file = "competition_models_cfg.json"
    out_file_path = "submission.csv"

    # Load the config and models
    models_callable = competition_models.CompetitionModels(model_config_file=input_model_cfg_file,
                                                           models_root_dir=input_models_root_dir,
                                                           device=torch.device("cuda:0"))
    models_callable.load_models()

    # Load the data and run inference
    competition_inference = CompetitionInference(input_pq_file=input_pq_file,
                                                    models_callable=models_callable)
    competition_inference.inference_all(out_file_path=out_file_path, log_debug=True, show_tqdm_bar=True)

    # Done. Garbage collect
    gc.collect()

    # Run the evaluation script
    solution = pd.read_csv("../data/train_events.csv")
    submission = pd.read_csv(out_file_path)
    score = kaggle_ap_detection.score(solution, submission,
                                      tolerances={"onset": [1, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30],
                                                  "wakeup": [1, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30]},
                                      series_id_column_name="series_id",
                                      time_column_name="step",
                                      event_column_name="event",
                                      score_column_name="score",
                                      use_scoring_intervals=False)

    print("Score: {}".format(score)) # this has data leakage since there is no train/test split. Just for bug catching
