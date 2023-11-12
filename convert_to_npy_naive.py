# "Naive" raw preprocessing+conversion into npy files for faster loading. You should run convert_to_individual_series.py before this
import os

import tqdm
import pandas as pd
import numpy as np

FOLDER = "data_naive"

def load_all_data_into_dict():
    all_data = {}
    for series_id in os.listdir(FOLDER):
        series_folder = os.path.join(FOLDER, series_id)
        accel = np.load(os.path.join(series_folder, "accel.npy"))
        sleeping_timesteps = np.load(os.path.join(series_folder, "sleeping_timesteps.npy"))
        secs = np.load(os.path.join(series_folder, "secs.npy"))
        mins = np.load(os.path.join(series_folder, "mins.npy"))
        hours = np.load(os.path.join(series_folder, "hours.npy"))
        all_data[series_id] = {}
        all_data[series_id]["accel"] = accel
        all_data[series_id]["sleeping_timesteps"] = sleeping_timesteps
        all_data[series_id]["secs"] = secs
        all_data[series_id]["mins"] = mins
        all_data[series_id]["hours"] = hours
    return all_data

def correct_time(secs: np.ndarray, mins: np.ndarray, hours: np.ndarray):
    daytime = secs.astype(np.int32) + mins.astype(np.int32) * 60 + hours.astype(np.int32) * 3600
    init_times = daytime - np.arange(len(daytime)) * 5 # 5 second intervals
    avg_start = np.median(init_times)
    daytime = avg_start + np.arange(len(daytime)) * 5
    secs = daytime % 60
    mins = (daytime // 60) % 60
    hours = (daytime // 3600) % 24
    return secs.astype(np.float32), mins.astype(np.float32), hours.astype(np.float32)


if __name__ == "__main__":
    # Ensure the input directory exists
    assert os.path.exists("individual_train_series"), "You should run convert_to_individual_series.py before this"

    # Load the events
    events = pd.read_csv("data/train_events.csv")
    assert set(list(events["event"].unique())) == {"onset", "wakeup"}

    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)
    num_discrepancy = 0
    num_time = 0
    discrepancy_series = []
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
        timestamps = pd.to_datetime(data_frame["timestamp"]).apply(lambda dt: dt.tz_localize(None)) # localize time also
        secs = timestamps.dt.second.to_numpy(dtype=np.float32)
        mins = timestamps.dt.minute.to_numpy(dtype=np.float32)
        hours = timestamps.dt.hour.to_numpy(dtype=np.float32)

        secs_corr, mins_corr, hours_corr = correct_time(secs, mins, hours)
        discrepancies = np.sum((secs_corr != secs) | (mins_corr != mins) | (hours_corr != hours))
        num_discrepancy += discrepancies
        num_time += len(secs)
        if discrepancies > 0:
            discrepancy_series.append(series_id)

        np.save(os.path.join(series_folder, "secs.npy"), secs_corr)
        np.save(os.path.join(series_folder, "mins.npy"), mins_corr)
        np.save(os.path.join(series_folder, "hours.npy"), hours_corr)

    print(f"Total number of discrepancies: {num_discrepancy}/{num_time}")
    print(f"Percentage of discrepancies: {num_discrepancy / num_time * 100:.2f}%")
    print(f"Total number of series with discrepancies: {len(discrepancy_series)}")
    print(f"Series with discrepancies: {discrepancy_series}")

    # ['08db4255286f', '10469f6765bf', '1087d7b0ff2e', '12d01911d509', '16fe2798ed0f', '188d4b7cd28b', '18a0ca03431d', '1d4569cbac0f', '2654a87be968', '292a75c0b94e', '2b8d87addea9', '2f7504d0f426', '3c336d6ba566', '416354edd92a', '44d8c02b369e', '4a31811f3558', '51fdcc8d9fe7', '5e816f11f5c3', '655f19eabf1e', '67f5fc60e494', '7476c0bd18d2', '7df249527c63', '808652a666c6', '844f54dcab89', '8a306e0890c0', '971207c6a525', 'a596ad0b82aa', 'a88088855de5', 'a9a2f7fac455', 'aa81faa78747', 'b4b75225b224', 'b737f8c78ec5', 'b84960841a75', 'bccf2f2819f8', 'bfe41e96d12f', 'c3072a759efb', 'c5d08fc3e040', 'c6788e579967', 'c75b4b207bea', 'd5e47b94477e', 'df33ae359fb5', 'dfc3ccebfdc9', 'e30cb792a2bc', 'e4500e7e19e1', 'eec197a4bdca', 'f56824b503a0']