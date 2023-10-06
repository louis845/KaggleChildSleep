# extracts good events from the events file

# you should run extract_individual_series.py before this
import os

import tqdm
import h5py
import pandas as pd
import numpy as np

FOLDER = "data_good_events"

class GoodEvents:
    def __init__(self, data_dict):


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
    min_event_series_id, max_event_series_id = None, None
    min_non_event_series_id, max_non_event_series_id = None, None
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)

    bad_segmentations_tail = ["31011ade7c0a", "a596ad0b82aa", "13b4d6a01d27", "10469f6765bf",
                              "05e1944c3818", "a9a2f7fac455", "ccdee561ee5d"] # non completed segmentations ccdee561ee5d
    bad_segmentations_tail = bad_segmentations_tail + [
        "b7188813d58a", "60d31b0bec3b", "3318a0e3ed6f", "599ca4ed791b", "0f572d690310",
        "0cfc06c129cc", "55a47ff9dc8a", "a4e48102f402"
    ] # probably really take off watch
    bad_segmentations_tail = bad_segmentations_tail + [
        "4feda0596965", "df33ae359fb5", "72ba4a8afff4"
    ] # partial, with some non-completed, some take off watch
    bad_segmentations_head = []

    for file in tqdm.tqdm(os.listdir("individual_train_series")):
        series_id = file.split(".")[0]
        series_output_folder = os.path.join(FOLDER, series_id)
        series_non_event_file = os.path.join(series_output_folder, "non_event.csv")
        series_event_file = os.path.join(series_output_folder, "event.csv")
        if not os.path.isdir(series_output_folder):
            os.mkdir(series_output_folder)

        data_frame = pd.read_parquet(os.path.join("individual_train_series", file))
        anglez = data_frame["anglez"].to_numpy(dtype=np.float32) / 35.52
        total_series_length = len(anglez)

        # Load the events
        non_events_lowhigh = []
        events_lowhigh = []

        series_events = events.loc[events["series_id"] == series_id]
        prev_has_event = True
        prev_event_step = 0
        for k in range(0, len(series_events) + 1):
            if k == len(series_events):
                if series_id in bad_segmentations_tail:
                    break
                if prev_has_event:
                    low = prev_event_step
                    high = total_series_length
                    cur_event_length = high - low
                    if k % 2 == 0:
                        if min_non_event_series_id is None:
                            min_non_event_series_id = series_id
                            max_non_event_series_id = series_id
                        else:
                            if cur_event_length < np.min(non_event_lengths_history):
                                min_non_event_series_id = series_id
                            if cur_event_length > np.max(non_event_lengths_history):
                                max_non_event_series_id = series_id
                        non_events_lowhigh.append((low, high))
                        non_event_lengths_history.append(high - low)
                    else:
                        if min_event_series_id is None:
                            min_event_series_id = series_id
                            max_event_series_id = series_id
                        else:
                            if cur_event_length < np.min(event_lengths_history):
                                min_event_series_id = series_id
                            if cur_event_length > np.max(event_lengths_history):
                                max_event_series_id = series_id
                        events_lowhigh.append((low, high))
                        event_lengths_history.append(high - low)
                    lengths_history.append(high - low)
            else:
                event = series_events.iloc[k]
                cur_has_event = not (pd.isna(event["step"]))
                if cur_has_event and prev_has_event:
                    if (k > 0) or ((series_id not in bad_segmentations_head)
                                    and (int(event["night"]) == 1)):
                        low = prev_event_step
                        high = int(event["step"])
                        cur_event_length = high - low
                        if k % 2 == 0:
                            if min_non_event_series_id is None:
                                min_non_event_series_id = series_id
                                max_non_event_series_id = series_id
                            else:
                                if cur_event_length < np.min(non_event_lengths_history):
                                    min_non_event_series_id = series_id
                                if cur_event_length > np.max(non_event_lengths_history):
                                    max_non_event_series_id = series_id
                            non_events_lowhigh.append((low, high))
                            non_event_lengths_history.append(cur_event_length)
                        else:
                            if min_event_series_id is None:
                                min_event_series_id = series_id
                                max_event_series_id = series_id
                            else:
                                if cur_event_length < np.min(event_lengths_history):
                                    min_event_series_id = series_id
                                if cur_event_length > np.max(event_lengths_history):
                                    max_event_series_id = series_id
                            events_lowhigh.append((low, high))
                            event_lengths_history.append(cur_event_length)
                        lengths_history.append(cur_event_length)
                if cur_has_event:
                    prev_event_step = int(event["step"])
                prev_has_event = cur_has_event

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
    print()
    print("Min event series id: {}".format(min_event_series_id))
    print("Max event series id: {}".format(max_event_series_id))
    print("Min non-event series id: {}".format(min_non_event_series_id))
    print("Max non-event series id: {}".format(max_non_event_series_id))

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
        f.write("\n")
        f.write("Min event series id: {}\n".format(min_event_series_id))
        f.write("Max event series id: {}\n".format(max_event_series_id))
        f.write("Min non-event series id: {}\n".format(min_non_event_series_id))
        f.write("Max non-event series id: {}\n".format(max_non_event_series_id))

