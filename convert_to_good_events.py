# extracts good events from the events file

# you should run extract_individual_series.py before this
import os

import tqdm
import h5py
import pandas as pd
import numpy as np

FOLDER = "data_good_events"

class GoodEvents:
    def __init__(self, data_dict, entries_sublist: list[str], is_train=True,
                 load_tail_entries=False, load_head_entries=False, min_headtail_length=1440):
        self.is_train = is_train
        if is_train:
            self.relevant_data = []
            for series_id in entries_sublist:
                if series_id not in data_dict:
                    raise ValueError("Series {} not found in data_dict".format(series_id))
                for low, high, pos in data_dict[series_id]["event"]:
                    if (not load_head_entries) and (pos == "head"):
                        continue
                    if (not load_tail_entries) and (pos == "tail"):
                        continue
                    if (high - low < min_headtail_length) and (pos != "middle"):
                        continue

                    self.relevant_data.append((low, high + 1, "event", series_id))
                for low, high, pos in data_dict[series_id]["non_event"]:
                    if (not load_head_entries) and (pos == "head"):
                        continue
                    if (not load_tail_entries) and (pos == "tail"):
                        continue
                    if (high - low < min_headtail_length) and (pos != "middle"):
                        continue

                    if low == 0:
                        self.relevant_data.append((0, high, "non_event", series_id))
                    else:
                        self.relevant_data.append((low + 1, high, "non_event", series_id))
            self.current_shuffle = None
            self.current_index = 0
        else:
            self.relevant_data = {}
            for series_id in entries_sublist:
                if series_id not in data_dict:
                    raise ValueError("Series {} not found in data_dict".format(series_id))
                self.relevant_data[series_id] = []
                for low, high, pos in data_dict[series_id]["event"]:
                    if (not load_head_entries) and (pos == "head"):
                        continue
                    if (not load_tail_entries) and (pos == "tail"):
                        continue
                    if (high - low < min_headtail_length) and (pos != "middle"):
                        continue

                    self.relevant_data[series_id].append((low, high + 1))
                for low, high, pos in data_dict[series_id]["non_event"]:
                    if (not load_head_entries) and (pos == "head"):
                        continue
                    if (not load_tail_entries) and (pos == "tail"):
                        continue
                    if (high - low < min_headtail_length) and (pos != "middle"):
                        continue

                    if low == 0:
                        self.relevant_data[series_id].append((0, high))
                    else:
                        self.relevant_data[series_id].append((low + 1, high))

    def shuffle(self):
        assert self.is_train, "You can only shuffle the training set"
        self.current_shuffle = np.random.permutation(len(self.relevant_data))
        self.current_index = 0

    def get_next(self, target_length: int, all_series_data: dict) -> tuple[np.ndarray, np.ndarray, int]:
        assert self.is_train, "You can only get_next on the training set"

        accel_data_batch_list = []
        labels_batch_list = []
        current_length = 0

        original_current_index = self.current_index
        while current_length < target_length:
            if self.current_index >= len(self.relevant_data):
                # randomly pick
                low, high, event_type, series_id = self.relevant_data[np.random.randint(0, len(self.relevant_data))]
            else:
                low, high, event_type, series_id = self.relevant_data[self.current_index]
                self.current_index += 1
            max_diff = min(int(high - low) // 4, 1440)
            low += np.random.randint(0, max_diff)
            high -= np.random.randint(0, max_diff)

            accel_data_batch_list.append(all_series_data[series_id]["accel"][:, low:high])
            labels_batch_list.append(all_series_data[series_id]["sleeping_timesteps"][low:high])

            current_length += accel_data_batch_list[-1].shape[-1]

        accel_data_batch = np.concatenate(accel_data_batch_list, axis=-1)
        labels_batch = np.concatenate(labels_batch_list, axis=-1)

        left_erosion = np.random.randint(0, accel_data_batch.shape[-1] - target_length + 1)
        accel_data_batch = accel_data_batch[:, left_erosion:(left_erosion + target_length)]
        labels_batch = labels_batch[left_erosion:(left_erosion + target_length)]

        return accel_data_batch, labels_batch, self.current_index - original_current_index

    def series_remaining(self) -> int:
        assert self.is_train, "You can only get_next on the training set"
        return len(self.relevant_data) - self.current_index

    def get_series_mask(self, series_id: str, all_series_data: dict) -> np.ndarray:
        assert not self.is_train, "You can only get_series_mask on the validation set"
        gt = all_series_data[series_id]["sleeping_timesteps"]
        mask = np.zeros(gt.shape, dtype=np.float32)
        for low, high in self.relevant_data[series_id]:
            mask[low:high] = 1.0
        return mask


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
            intervals = pd.read_csv(events_file, header=None).to_numpy(dtype="object")
            for k in range(intervals.shape[0]):
                start, end, position = intervals[k, :]
                all_data[series_id]["event"].append((int(start), int(end), position))
        except pd.errors.EmptyDataError:
            pass

        # load non-events
        all_data[series_id]["non_event"] = []
        try:
            intervals = pd.read_csv(non_events_file, header=None).to_numpy(dtype="object")
            for k in range(intervals.shape[0]):
                start, end, position = intervals[k, :]
                all_data[series_id]["non_event"].append((int(start), int(end), position))
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
                        non_events_lowhigh.append((low, high, "tail"))
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
                        events_lowhigh.append((low, high, "tail"))
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
                            if k == 0:
                                non_events_lowhigh.append((low, high, "head"))
                            else:
                                non_events_lowhigh.append((low, high, "middle"))
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
                            events_lowhigh.append((low, high, "middle"))
                            event_lengths_history.append(cur_event_length)
                        lengths_history.append(cur_event_length)
                if cur_has_event:
                    prev_event_step = int(event["step"])
                prev_has_event = cur_has_event

        # Save the non-events
        non_events_lowhigh = np.array(non_events_lowhigh, dtype="object")
        pd.DataFrame(non_events_lowhigh).to_csv(series_non_event_file, index=False, header=False)

        # Save the events
        events_lowhigh = np.array(events_lowhigh, dtype="object")
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

