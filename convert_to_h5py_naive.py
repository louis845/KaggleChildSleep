# you should run extract_individual_series.py before this
import os

import tqdm
import h5py
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

FOLDER = "data_naive"

def load_all_data_into_dict():
    all_data = {}
    for series_id in os.listdir(FOLDER):
        series_folder = os.path.join(FOLDER, series_id)
        accel = np.load(os.path.join(series_folder, "accel.npy"))
        sleeping_timesteps = np.load(os.path.join(series_folder, "sleeping_timesteps.npy"))
        all_data[series_id] = {}
        all_data[series_id]["accel"] = accel
        all_data[series_id]["sleeping_timesteps"] = sleeping_timesteps
    return all_data

if __name__ == "__main__":
    # Ensure the input directory exists
    assert os.path.exists("individual_train_series"), "You should run extract_individual_series.py before this"

    # Load the events
    events = pd.read_csv("data/train_events.csv")
    assert set(list(events["event"].unique())) == {"onset", "wakeup"}

    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)
    for file in tqdm.tqdm(os.listdir("individual_train_series")):
        series_id = file.split(".")[0]
        series_folder = os.path.join(FOLDER, series_id)
        if not os.path.isdir(series_folder):
            os.mkdir(series_folder)

        data_frame = pd.read_parquet(os.path.join("individual_train_series", file))
        accel_data = np.stack([data_frame["anglez"].to_numpy(dtype=np.float32) / 35.52,
                  data_frame["enmo"].to_numpy(dtype=np.float32) / 0.1018], axis=0) # shape (2, T)

        np.save(os.path.join(series_folder, "accel.npy"), accel_data)

        # Load the events
        series_events = events.loc[events["series_id"] == series_id]
        sleeping_timesteps = np.zeros((accel_data.shape[1],), dtype=np.uint8)
        for k in range(0, len(series_events), 2):
            single_event_onset = series_events.iloc[k]
            single_event_wakeup = series_events.iloc[k + 1]
            has_event = not (pd.isna(single_event_onset["step"]) or pd.isna(single_event_wakeup["step"]))
            if has_event:
                start = int(single_event_onset["step"])
                end = int(single_event_wakeup["step"])
                sleeping_timesteps[start:(end + 1)] = 1
        np.save(os.path.join(series_folder, "sleeping_timesteps.npy"), sleeping_timesteps)

        # Save the time stamps
        timestamps = pd.to_datetime(data_frame["timestamp"])
        timestamps = timestamps.dt.tz_localize(None) # localize time
        secs = timestamps.dt.second.to_numpy(dtype=np.float32)
        mins = timestamps.dt.minute.to_numpy(dtype=np.float32)
        hours = timestamps.dt.hour.to_numpy(dtype=np.float32)

        np.save(os.path.join(series_folder, "secs.npy"), secs)
        np.save(os.path.join(series_folder, "mins.npy"), mins)
        np.save(os.path.join(series_folder, "hours.npy"), hours)

