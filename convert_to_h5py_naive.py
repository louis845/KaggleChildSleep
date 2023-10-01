# you should run extract_individual_series.py before this
import os

import tqdm
import h5py
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

FOLDER = "data_naive"

if __name__ == "__main__":
    # Ensure the input directory exists
    assert os.path.exists("individual_train_series"), "You should run extract_individual_series.py before this"

    # Load the events
    events = pd.read_csv("data/train_events.csv")
    assert set(list(events["event"].unique())) == {"onset", "wakeup"}


    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)
    with h5py.File(os.path.join(FOLDER, "series_data.h5"), "w") as f:
        for file in tqdm.tqdm(os.listdir("individual_train_series")):
            series_id = file.split(".")[0]
            group = f.create_group(series_id)

            data_frame = pd.read_parquet(os.path.join("individual_train_series", file))
            accel_data = np.stack([data_frame["anglez"].to_numpy(dtype=np.float32) / 75.0,
                      data_frame["enmo"].to_numpy(dtype=np.float32)], axis=1)
            group.create_dataset("accel", data=accel_data, dtype=np.float32, compression="gzip", compression_opts=0)

            # Load the events
            series_events = events.loc[events["series_id"] == series_id]
            sleeping_timesteps = np.zeros((accel_data.shape[0],), dtype=np.uint8)
            for k in range(0, len(series_events), 2):
                single_event_onset = series_events.iloc[k]
                single_event_wakeup = series_events.iloc[k + 1]
                has_event = not (pd.isna(single_event_onset["step"]) or pd.isna(single_event_wakeup["step"]))
                if has_event:
                    start = int(single_event_onset["step"])
                    end = int(single_event_wakeup["step"])
                    sleeping_timesteps[start:(end + 1)] = 1
            group.create_dataset("sleeping_timesteps", data=sleeping_timesteps, dtype=np.uint8, compression="gzip", compression_opts=0)


