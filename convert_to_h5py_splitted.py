# you should run extract_individual_series.py before this
import os

import tqdm
import h5py
import pandas as pd
import numpy as np

FOLDER = "data_splitted"

def load_all_data_into_dict():
    all_data = {}
    for series_id in os.listdir(FOLDER):
        if series_id == "summary.txt":
            continue
        events_file = os.path.join(FOLDER, series_id, "event.csv")
        non_events_file = os.path.join(FOLDER, series_id, "non_event.csv")

        all_data[series_id] = {}

        # load events
        all_data[series_id]["event"] = []
        try:
            intervals = pd.read_csv(events_file, header=None).to_numpy(dtype=np.int32)
            for k in range(intervals.shape[0]):
                start, end = intervals[k, :]
                all_data[series_id]["event"].append((start, end))
        except pd.errors.EmptyDataError:
            pass

        # load non-events
        all_data[series_id]["non_event"] = []
        try:
            intervals = pd.read_csv(non_events_file, header=None).to_numpy(dtype=np.int32)
            for k in range(intervals.shape[0]):
                start, end = intervals[k, :]
                all_data[series_id]["non_event"].append((start, end))
        except pd.errors.EmptyDataError:
            pass
    return all_data

if __name__ == "__main__":
    # Ensure the input directory exists
    assert os.path.exists("individual_train_series"), "You should run extract_individual_series.py before this"

    # Load the events
    events = pd.read_csv("data/train_events.csv")
    assert set(list(events["event"].unique())) == {"onset", "wakeup"}

    lengths_history = []
    non_event_lengths_history = []
    event_lengths_history = []
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)
    for file in tqdm.tqdm(os.listdir("individual_train_series")):
        series_id = file.split(".")[0]
        series_output_folder = os.path.join(FOLDER, series_id)
        series_non_event_file = os.path.join(series_output_folder, "non_event.csv")
        series_event_file = os.path.join(series_output_folder, "event.csv")
        if not os.path.isdir(series_output_folder):
            os.mkdir(series_output_folder)

        data_frame = pd.read_parquet(os.path.join("individual_train_series", file))
        anglez = data_frame["anglez"].to_numpy(dtype=np.float32) / 35.52
        enmo = data_frame["enmo"].to_numpy(dtype=np.float32) / 0.1018

        # Load the events
        events_list = []
        series_events = events.loc[events["series_id"] == series_id]
        for k in range(0, len(series_events), 2):
            single_event_onset = series_events.iloc[k]
            single_event_wakeup = series_events.iloc[k + 1]
            has_event = not (pd.isna(single_event_onset["step"]) or pd.isna(single_event_wakeup["step"]))
            if has_event:
                start = int(single_event_onset["step"])
                end = int(single_event_wakeup["step"])
                events_list.append((start, end))

        prev_end = 0
        non_events_lowhigh = []
        events_lowhigh = []
        for k in range(len(events_list)):
            start, end = events_list[k]
            assert start > prev_end, "start: {}, prev_end: {}".format(start, prev_end)
            assert end > start, "end: {}, start: {}".format(end, start)
            non_events_lowhigh.append((prev_end, start))
            lengths_history.append(start - prev_end)
            non_event_lengths_history.append(start - prev_end)

            events_lowhigh.append((start, end + 1))
            lengths_history.append(end + 1 - start)
            event_lengths_history.append(end + 1 - start)

            prev_end = end + 1

        if prev_end < len(anglez):
            non_events_lowhigh.append((prev_end, len(anglez)))
            lengths_history.append(len(anglez) - prev_end)
            non_event_lengths_history.append(len(anglez) - prev_end)

        # Save the non-events
        non_events_lowhigh = np.array(non_events_lowhigh, dtype=np.int32)
        pd.DataFrame(non_events_lowhigh).to_csv(series_non_event_file, index=False, header=False)

        # Save the events
        events_lowhigh = np.array(events_lowhigh, dtype=np.int32)
        pd.DataFrame(events_lowhigh).to_csv(series_event_file, index=False, header=False)


    print("Max length: {}".format(np.max(lengths_history)))
    print("Min length: {}".format(np.min(lengths_history)))
    print("Median length: {}".format(np.median(lengths_history)))
    print("Max event length: {}".format(np.max(event_lengths_history)))
    print("Min event length: {}".format(np.min(event_lengths_history)))
    print("Median event length: {}".format(np.median(event_lengths_history)))
    print("Max non-event length: {}".format(np.max(non_event_lengths_history)))
    print("Min non-event length: {}".format(np.min(non_event_lengths_history)))
    print("Median non-event length: {}".format(np.median(non_event_lengths_history)))

    with open(os.path.join(FOLDER, "summary.txt"), "w") as f:
        f.write("Max length: {}\n".format(np.max(lengths_history)))
        f.write("Min length: {}\n".format(np.min(lengths_history)))
        f.write("Median length: {}\n".format(np.median(lengths_history)))
        f.write("Max event length: {}\n".format(np.max(event_lengths_history)))
        f.write("Min event length: {}\n".format(np.min(event_lengths_history)))
        f.write("Median event length: {}\n".format(np.median(event_lengths_history)))
        f.write("Max non-event length: {}\n".format(np.max(non_event_lengths_history)))
        f.write("Min non-event length: {}\n".format(np.min(non_event_lengths_history)))
        f.write("Median non-event length: {}\n".format(np.median(non_event_lengths_history)))
