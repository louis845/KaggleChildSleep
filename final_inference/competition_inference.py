import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import typing

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

    def __init__(self, input_pq_file: str, series_inference_callable: typing.Callable):
        self.input_pq_file = input_pq_file
        self.series_inference_callable = series_inference_callable

    def inference_all(self):
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
            self.series_inference_callable(series_id, accel_data, secs_corr, mins_corr, hours_corr)
