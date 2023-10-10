import numpy as np
import pandas as pd
import tqdm

import os
import shutil
import convert_to_h5py_naive

OUT_FOLDER = "ensembled_pseudo_labels"
PROBAS_FOLDER = os.path.join(OUT_FOLDER, "probas")
LABELS_FOLDER = os.path.join(OUT_FOLDER, "labels")
EVENTS_FOLDER = os.path.join(OUT_FOLDER, "events")

class PseudoEvents:
    def __init__(self, data_dict, entries_sublist: list[str], is_train=True):
        self.is_train = is_train
        if is_train:
            self.relevant_data = []
            for series_id in entries_sublist:
                if series_id not in data_dict:
                    raise ValueError("Series {} not found in data_dict".format(series_id))
                for low, high in data_dict[series_id]["event"]:
                    self.relevant_data.append((low, high + 1, "event", series_id))
                for low, high in data_dict[series_id]["non_event"]:
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
                for low, high in data_dict[series_id]["event"]:
                    self.relevant_data[series_id].append((low, high + 1))
                for low, high in data_dict[series_id]["non_event"]:
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
    for series_id in os.listdir(convert_to_h5py_naive.FOLDER):
        series_folder = os.path.join(convert_to_h5py_naive.FOLDER, series_id)
        accel = np.load(os.path.join(series_folder, "accel.npy"))
        sleeping_timesteps = np.load(os.path.join(LABELS_FOLDER, series_id + ".npy"))
        all_data[series_id] = {}
        all_data[series_id]["accel"] = accel
        all_data[series_id]["sleeping_timesteps"] = sleeping_timesteps
    return all_data

def load_all_events_into_dict():
    all_data = {}
    for series_id in os.listdir(EVENTS_FOLDER):
        events_file = os.path.join(EVENTS_FOLDER, series_id, "event.csv")
        non_events_file = os.path.join(EVENTS_FOLDER, series_id, "non_event.csv")

        all_data[series_id] = {}

        # load events
        all_data[series_id]["event"] = []
        try:
            intervals = pd.read_csv(events_file, header=None).to_numpy(dtype="object")
            for k in range(intervals.shape[0]):
                start, end = intervals[k, :]
                all_data[series_id]["event"].append((int(start), int(end)))
        except pd.errors.EmptyDataError:
            pass

        # load non-events
        all_data[series_id]["non_event"] = []
        try:
            intervals = pd.read_csv(non_events_file, header=None).to_numpy(dtype="object")
            for k in range(intervals.shape[0]):
                start, end = intervals[k, :]
                all_data[series_id]["non_event"].append((int(start), int(end)))
        except pd.errors.EmptyDataError:
            pass
    return all_data

if __name__ == "__main__":
    if os.path.isdir(OUT_FOLDER):
        shutil.rmtree(OUT_FOLDER)
    os.mkdir(OUT_FOLDER)
    os.mkdir(PROBAS_FOLDER)
    os.mkdir(LABELS_FOLDER)
    os.mkdir(EVENTS_FOLDER)

    series_ids = os.listdir("data_naive")
    series_ids.sort()

    # load collection of generated labels
    generated_labels = os.listdir("pseudo_labels")
    generated_labels_collections = {}
    for generated_label_collection in generated_labels:
        generated_labels_collections[generated_label_collection] = \
            [x[:-4] for x in os.listdir(os.path.join("pseudo_labels", generated_label_collection))]

    # average (ensemble) the predictions for each series
    for series_id in tqdm.tqdm(series_ids):
        collections_containing_series = [generated_label_collection for generated_label_collection in generated_labels
                                                if series_id in generated_labels_collections[generated_label_collection]]
        assert len(collections_containing_series) == 4, "Each series should be in exactly 6 collections"

        # load predictions
        predictions = []

        for generated_label_collection in collections_containing_series:
            predictions.append(np.load(os.path.join("pseudo_labels", generated_label_collection, series_id + ".npy")))

        # average
        predictions = np.stack(predictions, axis=0)
        predictions = np.mean(predictions, axis=0)

        # save
        np.save(os.path.join(PROBAS_FOLDER, series_id + ".npy"), predictions)

        # convert to labels
        LENGTH_THRESHOLD = 120
        MISSING_TOLERANCE = 60

        labels = predictions > 0.6

        found_min = -1
        k = 0

        found_events = []
        while (k < len(labels)):
            if found_min == -1:
                if labels[k]:
                    found_min = k
            else:
                if not labels[k]:
                    seek_end = min(k + MISSING_TOLERANCE, len(labels))
                    if np.all(np.logical_not(labels[k:seek_end])):
                        if k - found_min >= LENGTH_THRESHOLD:
                            if np.sum(labels[found_min:k]) > int(0.6 * (k - found_min)):
                                labels[found_min:k] = True
                                found_events.append((found_min, k))
                            else:
                                labels[found_min:k] = False
                        else:
                            labels[found_min:k] = False
                        found_min = -1
                    else:
                        for j in range(k, seek_end):
                            if labels[j]:
                                k = j
            k += 1

        if found_min != -1:
            if k - found_min >= LENGTH_THRESHOLD:
                labels[found_min:k] = True
            else:
                labels[found_min:k] = False
        np.save(os.path.join(LABELS_FOLDER, series_id + ".npy"), labels.astype(np.float32))

        if len(found_events) == 0:
            found_nonevents = [(0, len(labels))]
        else:
            found_nonevents = []
            if found_events[0][0] > 0:
                found_nonevents.append((0, found_events[0][0]))
            for k in range(len(found_events)):
                if k == len(found_events) - 1:
                    if found_events[k][1] < len(labels):
                        found_nonevents.append((found_events[k][1], len(labels)))
                else:
                    found_nonevents.append((found_events[k][1], found_events[k + 1][0]))

        # Create the series folder
        series_folder = os.path.join(EVENTS_FOLDER, series_id)
        os.mkdir(series_folder)
        series_event_file = os.path.join(series_folder, "events.csv")
        series_non_event_file = os.path.join(series_folder, "non_events.csv")

        # Save the non-events
        non_events_lowhigh = np.array(found_nonevents, dtype="object")
        pd.DataFrame(non_events_lowhigh).to_csv(series_non_event_file, index=False, header=False)

        # Save the events
        events_lowhigh = np.array(found_events, dtype="object")
        pd.DataFrame(events_lowhigh).to_csv(series_event_file, index=False, header=False)

