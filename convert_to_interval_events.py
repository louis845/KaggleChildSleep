import os
import json

import numpy as np
import pandas as pd
import tqdm

import convert_to_h5py_naive
import convert_to_good_events

def create_regression_range(mask: np.ndarray, values: np.ndarray, event_type: int, location: int, window_radius: int):
    assert mask.shape == values.shape, "mask and values must have the same shape"
    assert event_type in [0, 1], "event_type must be 0 or 1"
    assert mask.shape[0] == 2

    window_max = location + window_radius + 1
    window_min = location - window_radius
    window_max = min(window_max, mask.shape[1])
    window_min = max(window_min, 0)

    mask[event_type, window_min:window_max] = 1.0
    min_val = window_min - location
    max_val = window_max - location
    values[event_type, window_min:window_max] = np.arange(min_val, max_val)

def get_first_day_step(naive_all_data, series_id):
    hours = naive_all_data[series_id]["hours"]
    mins = naive_all_data[series_id]["mins"]
    secs = naive_all_data[series_id]["secs"]

    night_onset = (hours == 17) & (mins == 0) & (secs == 0) # 1700
    first_day_step = np.argwhere(night_onset).flatten()[0]
    return first_day_step

def get_truncated_series_length(naive_all_data, series_id, events):
    series_max = len(naive_all_data[series_id]["hours"])
    if series_id in convert_to_good_events.bad_segmentations_tail:
        series_events = events.loc[events["series_id"] == series_id]
        series_max = min(series_events.iloc[-1]["step"] + 8640, series_max)
    return series_max

class IntervalEventsSampler:
    def __init__(self, series_ids: list[str], naive_all_data: dict, train_or_test="train",
                 prediction_length=17280, # 24 hours
                 prediction_stride=4320 # 6 hours
                 ):
        self.series_ids = series_ids
        self.naive_all_data = naive_all_data

        events = pd.read_csv("data/train_events.csv")
        self.events = events.dropna()

        self.all_segmentations = {}
        self.all_segmentations_list = []
        for series_id in series_ids:
            self.all_segmentations[series_id] = []

            interval_min = get_first_day_step(naive_all_data, series_id)
            interval_max = get_truncated_series_length(naive_all_data, series_id, self.events)

            while interval_min + prediction_length < interval_max:
                self.all_segmentations_list.append({
                    "series_id": series_id,
                    "start": interval_min,
                    "end": interval_min + prediction_length
                })
                self.all_segmentations[series_id].append({
                    "start": interval_min,
                    "end": interval_min + prediction_length
                })
                interval_min += prediction_stride

        self.shuffle_indices = None
        self.sample_low = 0

        self.train_or_test = train_or_test


    def shuffle(self):
        if self.train_or_test == "train":
            self.shuffle_indices = np.random.permutation(len(self.all_segmentations_list))
        else:
            if self.shuffle_indices is None:
                self.shuffle_indices = np.arange(len(self.all_segmentations_list))
        self.sample_low = 0

    def sample_single(self, index: int, random_shift: int=0, flip: bool=False, expand: int=0,
                      elastic_deformation=False, include_all_events=False):
        # index denotes the index in self.all_segmentations_list
        # returns (accel_data, event_segmentations), where event_segmentations[0, :] is onset, and event_segmentations[1, :] is wakeup

        assert expand % 12 == 0, "expand must be a multiple of 12"

        interval_info = self.all_segmentations_list[index]
        series_id = interval_info["series_id"]
        start = interval_info["start"]
        end = interval_info["end"]
        total_length = int(self.naive_all_data[series_id]["accel"].shape[1])

        # shift if expansion makes it out of boundaries
        if expand > 0:
            if start - expand < 0:
                overhead = expand - start
                start += overhead
                end += overhead
            if end + expand > total_length:
                overhead = end + expand - total_length
                end -= overhead
                start -= overhead

        # apply random shift
        shift = 0
        if random_shift > 0:
            shift = np.random.randint(-random_shift, random_shift + 1)
            shift = max(min(shift, total_length - end - expand), -start + expand)
        start, end = start + shift, end + shift

        assert start - expand >= 0 and end + expand <= total_length, "start: {}, end: {}, total_length: {}".format(start, end, total_length)

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
                elif include_all_events:
                    evt_type = evts.iloc[0]["event"]
                    assert len(evts) == 1
                    assert evt_type in ["onset", "wakeup"]
                    if evt_type == "onset":
                        grouped_events.append({
                            "onset": int(evts.iloc[0]["step"]),
                            "wakeup": None
                        })
                    else:
                        grouped_events.append({
                            "onset": None,
                            "wakeup": int(evts.iloc[0]["step"])
                        })

        # Load acceleration data and event segmentations
        accel_data = self.naive_all_data[series_id]["accel"][:, (start - expand):(end + expand)]
        event_segmentations = np.zeros((2, end - start + 2 * expand), dtype=np.float32)
        event_tolerance_width = 30 * 12 - 1
        for event in grouped_events:
            onset = (int(event["onset"] - start + expand)) if event["onset"] is not None else None
            wakeup = (int(event["wakeup"] - start + expand)) if event["wakeup"] is not None else None
            if flip:
                onset, wakeup = wakeup, onset
            if onset is not None:
                event_segmentations[0, max(0, onset - event_tolerance_width):min(event_segmentations.shape[1],
                                                                                 onset + event_tolerance_width + 1)] = 1.0
            if wakeup is not None:
                event_segmentations[1, max(0, wakeup - event_tolerance_width):min(event_segmentations.shape[1],
                                                                                  wakeup + event_tolerance_width + 1)] = 1.0

        if flip:
            accel_data = np.flip(accel_data, axis=1)
            event_segmentations = np.flip(event_segmentations, axis=1)

        return accel_data, event_segmentations

    def sample(self, batch_size: int, random_shift: int=0, random_flip: bool=False, always_flip: bool=False, expand: int=0, include_all_events=False):
        assert self.shuffle_indices is not None, "shuffle_indices is None, call shuffle() first"

        accel_datas = []
        event_segmentations = []

        increment = min(batch_size, len(self.all_segmentations_list) - self.sample_low)

        for k in range(self.sample_low, self.sample_low + increment):
            flip = False
            if random_flip:
                flip = np.random.randint(0, 2) == 1
            if always_flip:
                flip = True
            accel_data, event_segmentation = self.sample_single(self.shuffle_indices[k], random_shift=random_shift, flip=flip, expand=expand, include_all_events=include_all_events)
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

class RecurrentIntervalEventsSampler:
    pass


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
        first_night = get_first_day_step(naive_all_data, series_id)
        first_night_end = first_night + 17280

        while first_night_end < len(naive_all_data[series_id]["hours"]):
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
        first_night = get_first_day_step(naive_all_data, series_id)
        first_night_end = first_night + 17280

        series_max = len(naive_all_data[series_id]["hours"])
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
