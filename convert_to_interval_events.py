import os
import json

import numpy as np
import pandas as pd
import tqdm

import convert_to_h5py_naive
import convert_to_good_events

class IntervalEventsSampler:
    def __init__(self, series_ids: list[str], naive_all_data: dict, all_segmentations: dict,
                 train_or_test="train"):
        self.series_ids = series_ids
        self.naive_all_data = naive_all_data

        self.all_segmentations = {}
        self.all_segmentations_list = []
        for series_id in series_ids:
            self.all_segmentations[series_id] = all_segmentations[series_id]
            self.all_segmentations_list.extend(all_segmentations[series_id])

        self.shuffle_indices = None
        self.sample_low = 0

        self.train_or_test = train_or_test

        events = pd.read_csv("data/train_events.csv")
        self.events = events.dropna()

    def shuffle(self):
        if self.train_or_test == "train":
            self.shuffle_indices = np.random.permutation(len(self.all_segmentations_list))
        else:
            if self.shuffle_indices is None:
                self.shuffle_indices = np.arange(len(self.all_segmentations_list))
        self.sample_low = 0

    def sample_single(self, index: int, random_shift: int=0):
        # index denotes the index in self.all_segmentations_list
        # returns (accel_data, event_segmentations), where event_segmentations[0, :] is onset, and event_segmentations[1, :] is wakeup

        night_info = self.all_segmentations_list[index]
        series_id = night_info["series_id"]
        start = night_info["start"]
        end = night_info["end"]
        total_length = int(self.naive_all_data[series_id]["accel"].shape[1])

        # apply random shift
        shift = 0
        if random_shift > 0:
            shift = np.random.randint(-random_shift, random_shift + 1)
            shift = max(min(shift, total_length - end), -start)
        start, end = start + shift, end + shift

        # load events
        series_events = self.events.loc[self.events["series_id"] == series_id]
        events_contained = series_events.loc[
            ((series_events["step"] >= start) & (series_events["step"] < end))]
        grouped_events = []
        if len(events_contained) > 0:
            events_contained = events_contained.sort_values(by=["step"])
            for k in events_contained["night"].unique():
                evts = events_contained.loc[events_contained["night"] == k]
                if len(evts) == 2:
                    grouped_events.append({
                        "onset": int(evts.iloc[0]["step"]),
                        "wakeup": int(evts.iloc[1]["step"])
                    })

        accel_data = self.naive_all_data[series_id]["accel"][:, start:end]
        event_segmentations = np.zeros((2, accel_data.shape[1] // 12), dtype=np.float32)
        event_tolerance_width = 30
        for event in grouped_events:
            onset = int(np.round((event["onset"] - start) / 12.0))
            wakeup = int(np.round((event["wakeup"] - start) / 12.0))
            event_segmentations[0, max(0, onset - event_tolerance_width):min(event_segmentations.shape[1], onset + event_tolerance_width + 1)] = 1.0
            event_segmentations[1, max(0, wakeup - event_tolerance_width):min(event_segmentations.shape[1], wakeup + event_tolerance_width + 1)] = 1.0

        return accel_data, event_segmentations

    def sample(self, batch_size: int, random_shift: int=0):
        assert self.shuffle_indices is not None, "shuffle_indices is None, call shuffle() first"

        accel_datas = []
        event_segmentations = []

        increment = min(batch_size, len(self.all_segmentations_list) - self.sample_low)

        for k in range(self.sample_low, self.sample_low + increment):
            accel_data, event_segmentation = self.sample_single(self.shuffle_indices[k], random_shift=random_shift)
            accel_datas.append(accel_data)
            event_segmentations.append(event_segmentation)

        self.sample_low += increment
        accel_datas = np.stack(accel_datas, axis=0)
        event_segmentations = np.stack(event_segmentations, axis=0)

        return accel_datas, event_segmentations, increment

    def __len__(self):
        return len(self.all_segmentations_list)

    def entries_remaining(self):
        return len(self.all_segmentations_list) - self.sample_low

class SemiSyntheticIntervalEventsSampler:
    # augments the data with cutmix
    def __init__(self, series_ids: list[str], naive_all_data: dict, all_segmentations: dict,
                 spliced_good_events: convert_to_good_events.GoodEventsSplicedSampler, cutmix_skip=720):
        self.series_ids = series_ids
        self.naive_all_data = naive_all_data

        self.all_segmentations = {}
        self.all_segmentations_list = []
        for series_id in series_ids:
            self.all_segmentations[series_id] = all_segmentations[series_id]
            self.all_segmentations_list.extend(all_segmentations[series_id])

        self.shuffle_indices = None
        self.sample_low = 0

        events = pd.read_csv("data/train_events.csv")
        self.events = events.dropna()

        self.cutmix_skip = cutmix_skip
        self.cutmix_length = spliced_good_events.expected_length
        self.spliced_good_events = spliced_good_events

    def shuffle(self):
        if self.shuffle_indices is None:
            self.shuffle_indices = np.random.permutation(len(self.all_segmentations_list))
        self.sample_low = 0

    def cutmix_segment(self, accel_data: np.ndarray, start: int, end: int, type: str):
        if end - start < 720: # 1 hr
            return

        start += 360 # contract by 30 minutes on both ends
        end -= 360

        # determine which segments to cut and replace
        cut_segments = []
        mix_method = np.random.randint(0, 3)
        if mix_method == 0:
            return
        elif mix_method == 1:
            start += np.random.randint(0, self.cutmix_skip)
            while start + self.cutmix_length <= end:
                cut_segments.append((start, start + self.cutmix_length))
                start = start + self.cutmix_length + np.random.randint(0, self.cutmix_skip)
        else:
            end -= np.random.randint(0, self.cutmix_skip)
            while end - self.cutmix_length >= start:
                cut_segments.append((end - self.cutmix_length, end))
                end = end - self.cutmix_length - np.random.randint(0, self.cutmix_skip)

        # cut and replace here
        for seg_start, seg_end in cut_segments:
            if type == "event":
                series_id, low, high = self.spliced_good_events.sample_event()
            else:
                series_id, low, high = self.spliced_good_events.sample_non_event()
            assert low >= 0 and high < self.naive_all_data[series_id]["accel"].shape[1], "low: {}, high: {}, accel.shape: {}".format(low, high, self.naive_all_data[series_id]["accel"].shape)

            accel_data[:, seg_start:seg_end] = self.naive_all_data[series_id]["accel"][:, low:high]



    def sample_single(self, index: int, random_shift: int=0):
        # index denotes the index in self.all_segmentations_list
        # returns (accel_data, event_segmentations), where event_segmentations[0, :] is onset, and event_segmentations[1, :] is wakeup

        night_info = self.all_segmentations_list[index]
        series_id = night_info["series_id"]
        start = night_info["start"]
        end = night_info["end"]
        total_length = int(self.naive_all_data[series_id]["accel"].shape[1])

        # apply random shift
        shift = 0
        if random_shift > 0:
            shift = np.random.randint(-random_shift, random_shift + 1)
            shift = max(min(shift, total_length - end), -start)
        start, end = start + shift, end + shift

        # load events
        series_events = self.events.loc[self.events["series_id"] == series_id]
        events_contained = series_events.loc[
            ((series_events["step"] >= start) & (series_events["step"] < end))]
        grouped_events = []
        if len(events_contained) > 0:
            events_contained = events_contained.sort_values(by=["step"])
            for k in events_contained["night"].unique():
                evts = events_contained.loc[events_contained["night"] == k]
                if len(evts) == 2:
                    grouped_events.append({
                        "onset": int(evts.iloc[0]["step"]),
                        "wakeup": int(evts.iloc[1]["step"])
                    })

        accel_data = self.naive_all_data[series_id]["accel"][:, start:end]
        event_segmentations = np.zeros((2, accel_data.shape[1] // 12), dtype=np.float32)
        event_tolerance_width = 30
        for event in grouped_events:
            onset = int(np.round((event["onset"] - start) / 12.0))
            wakeup = int(np.round((event["wakeup"] - start) / 12.0))
            event_segmentations[0, max(0, onset - event_tolerance_width):min(event_segmentations.shape[1], onset + event_tolerance_width + 1)] = 1.0
            event_segmentations[1, max(0, wakeup - event_tolerance_width):min(event_segmentations.shape[1], wakeup + event_tolerance_width + 1)] = 1.0

        # cutmix
        if len(grouped_events) > 0:
            prev_end = 0
            for k in range(len(grouped_events)):
                #self.cutmix_segment(accel_data, prev_end, grouped_events[k]["onset"] - start, type="non_event")
                self.cutmix_segment(accel_data, grouped_events[k]["onset"] - start, grouped_events[k]["wakeup"] - start, type="event")
                prev_end = grouped_events[k]["wakeup"] - start

            #self.cutmix_segment(accel_data, prev_end, accel_data.shape[1], type="non_event")


        return accel_data, event_segmentations

    def sample(self, batch_size: int, random_shift: int=0):
        assert self.shuffle_indices is not None, "shuffle_indices is None, call shuffle() first"

        accel_datas = []
        event_segmentations = []

        increment = min(batch_size, len(self.all_segmentations_list) - self.sample_low)

        for k in range(self.sample_low, self.sample_low + increment):
            accel_data, event_segmentation = self.sample_single(self.shuffle_indices[k], random_shift=random_shift)
            accel_datas.append(accel_data)
            event_segmentations.append(event_segmentation)

        self.sample_low += increment
        accel_datas = np.stack(accel_datas, axis=0)
        event_segmentations = np.stack(event_segmentations, axis=0)

        return accel_datas, event_segmentations, increment

    def __len__(self):
        return len(self.all_segmentations_list)

    def entries_remaining(self):
        return len(self.all_segmentations_list) - self.sample_low


def load_all_segmentations(truncated=True) -> dict:
    if truncated:
        with open(os.path.join(FOLDER, "series_segmentations_trunc.json"), "r") as f:
            return json.load(f)
    else:
        with open(os.path.join(FOLDER, "series_segmentations.json"), "r") as f:
            return json.load(f)

FOLDER = "data_interval_events"

if __name__ == "__main__":
    naive_all_data = convert_to_h5py_naive.load_all_data_into_dict()
    series_ids = list(naive_all_data.keys())

    events = pd.read_csv("data/train_events.csv")
    events = events.dropna()

    series_segmentations = {}

    for series_id in tqdm.tqdm(series_ids):
        series_events = events.loc[events["series_id"] == series_id]
        series_segmentations[series_id] = []

        assert series_id in naive_all_data, "series_id {} not in naive_all_data".format(series_id)
        hours = naive_all_data[series_id]["hours"]
        mins = naive_all_data[series_id]["mins"]
        secs = naive_all_data[series_id]["secs"]

        night_onset = (hours == 17) & (mins == 0) & (secs == 0)
        first_night = np.argwhere(night_onset).flatten()[0]
        first_night_end = first_night + 17280

        while first_night_end < len(hours):
            night_info_dict = {
                "series_id": series_id,
                "start": int(first_night),
                "end": int(first_night_end),
                "events": []
            }
            events_contained = series_events.loc[
                ((series_events["step"] >= first_night) & (series_events["step"] < first_night_end))]
            if len(events_contained) > 0:
                events_contained = events_contained.sort_values(by=["step"])
                for k in events_contained["night"].unique():
                    evts = events_contained.loc[events_contained["night"] == k]
                    if len(evts) == 2:
                        assert (int(evts.iloc[0]["step"]) - int(first_night)) % 12 == 0, "All events should be in terms of nearest minute"
                        assert (int(evts.iloc[1]["step"]) - int(first_night)) % 12 == 0, "All events should be in terms of nearest minute"
                        night_info_dict["events"].append({
                            "onset": int(evts.iloc[0]["step"]),
                            "wakeup": int(evts.iloc[1]["step"])
                        })

            series_segmentations[series_id].append(night_info_dict)
            first_night += 4320
            first_night_end = first_night + 17280


    series_segmentations_trunc = {}
    num_bad = 0
    for series_id in tqdm.tqdm(series_ids):
        series_events = events.loc[events["series_id"] == series_id]
        series_segmentations_trunc[series_id] = []

        assert series_id in naive_all_data, "series_id {} not in naive_all_data".format(series_id)
        hours = naive_all_data[series_id]["hours"]
        mins = naive_all_data[series_id]["mins"]
        secs = naive_all_data[series_id]["secs"]

        night_onset = (hours == 17) & (mins == 0) & (secs == 0)
        first_night = np.argwhere(night_onset).flatten()[0]
        first_night_end = first_night + 17280

        series_max = len(hours)
        if series_id in convert_to_good_events.bad_segmentations_tail:
            num_bad += 1
            series_max = min(series_events.iloc[-1]["step"] + 8640, series_max)
        while first_night_end < series_max:
            night_info_dict = {
                "series_id": series_id,
                "start": int(first_night),
                "end": int(first_night_end),
                "events": []
            }
            events_contained = series_events.loc[
                ((series_events["step"] >= first_night) & (series_events["step"] < first_night_end))]
            if len(events_contained) > 0:
                events_contained = events_contained.sort_values(by=["step"])
                for k in events_contained["night"].unique():
                    evts = events_contained.loc[events_contained["night"] == k]
                    if len(evts) == 2:
                        assert (int(evts.iloc[0]["step"]) - int(
                            first_night)) % 12 == 0, "All events should be in terms of nearest minute"
                        assert (int(evts.iloc[1]["step"]) - int(
                            first_night)) % 12 == 0, "All events should be in terms of nearest minute"
                        night_info_dict["events"].append({
                            "onset": int(evts.iloc[0]["step"]),
                            "wakeup": int(evts.iloc[1]["step"])
                        })

            series_segmentations_trunc[series_id].append(night_info_dict)
            first_night += 4320
            first_night_end = first_night + 17280
    print("num_bad: {}".format(num_bad))

    # save series_segmentations to disk
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)

    with open(os.path.join(FOLDER, "series_segmentations.json"), "w") as f:
        json.dump(series_segmentations, f, indent=4)

    with open(os.path.join(FOLDER, "series_segmentations_trunc.json"), "w") as f:
        json.dump(series_segmentations_trunc, f, indent=4)
