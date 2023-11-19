import gc

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import competition_models

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

    def inference_all(self, out_file_path):
        # Create output file
        out_file = open(out_file_path, "w")
        row_id = 0
        out_file.write("row_id,series_id,step,event,score\n")

        # Load only the "series_id" column
        series_id_df = pd.read_parquet(self.input_pq_file, columns=["series_id"])

        # Get unique "series_id" values
        series_ids = series_id_df["series_id"].unique()

        # Iteratively read data for each "series_id"
        for series_id in series_ids:
            df = pd.read_parquet(self.input_pq_file, columns=["step", "timestamp", "anglez"],
                                 filters=[("series_id", "==", series_id)])

            accel_data = np.expand_dims(df["anglez"].to_numpy(dtype=np.float32) / 35.52, axis=0) # shape (1, T)
            timestamps = pd.to_datetime(df["timestamp"]).apply(lambda dt: dt.tz_localize(None))

            secs = timestamps.dt.second.to_numpy(dtype=np.float32)
            mins = timestamps.dt.minute.to_numpy(dtype=np.float32)
            hours = timestamps.dt.hour.to_numpy(dtype=np.float32)

            secs_corr, mins_corr, hours_corr = correct_time(secs, mins, hours)

            # Call the series inference callable
            onset_locs, onset_IOU_probas, wakeup_locs, wakeup_IOU_probas = self.models_callable.run_inference(series_id, accel_data, secs_corr, mins_corr, hours_corr)

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

        out_file.close()