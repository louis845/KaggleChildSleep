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

def create_regression_kernel(mask: np.ndarray, values: np.ndarray, event_type: int, location: int, window_radius: int,
                             kernel_radius: int):
    assert mask.shape == values.shape, "mask and values must have the same shape"
    assert event_type in [0, 1], "event_type must be 0 or 1"
    assert mask.shape[0] == 2

    window_max = location + window_radius + 1
    window_min = location - window_radius
    window_max = min(window_max, mask.shape[1])
    window_min = max(window_min, 0)

    mask[event_type, window_min:window_max] = 1.0

    kernel_max = location + kernel_radius + 1
    kernel_min = location - kernel_radius
    kernel_max = min(kernel_max, mask.shape[1])
    kernel_min = max(kernel_min, 0)

    min_val = kernel_min - location
    max_val = kernel_max - location
    kernel_values = (1 - np.abs(np.arange(min_val, max_val)).astype(np.float32) / kernel_radius) ** 2
    assert np.all(kernel_values >= 0.0) and np.all(kernel_values <= 1.0), "kernel_values must be between 0 and 1"
    values[event_type, kernel_min:kernel_max] = kernel_values


class IntervalRegressionSampler:
    def __init__(self, series_ids: list[str], naive_all_data: dict, event_regressions: list[int],
                 train_or_test: str="train", use_kernel: int=None):
        if use_kernel is not None:
            assert isinstance(use_kernel, int), "use_kernel must be an integer"
            assert use_kernel >= 12, "use_kernel must be a positive integer >= 12"
        assert isinstance(event_regressions, list), "event_regressions must be a list of integers"
        for k in event_regressions:
            assert isinstance(k, int), "event_regressions must be a list of integers"
            assert k > 0, "event_regressions must be a list of positive integers"

        self.series_ids = series_ids
        self.naive_all_data = naive_all_data

        events = pd.read_csv("data/train_events.csv")
        events = events.dropna()
        self.events = events
        self.event_regressions = event_regressions


        self.train_or_test = train_or_test
        if train_or_test == "train":
            self.event_regression_samples = []
            self.shuffle_indices = None
            self.sample_low = 0
            max_length = 0
            for series_id in series_ids:
                series_events = events.loc[events["series_id"] == series_id]
                for night in series_events["night"].unique():
                    night_events = series_events.loc[series_events["night"] == night]
                    if len(night_events) == 2:
                        assert night_events.iloc[0]["event"] == "onset"
                        assert night_events.iloc[1]["event"] == "wakeup"
                        onset = int(night_events.iloc[0]["step"])
                        wakeup = int(night_events.iloc[1]["step"])
                        self.event_regression_samples.append({
                            "series_id": series_id,
                            "onset": onset,
                            "wakeup": wakeup
                        })
                        max_length = max(max_length, (wakeup - onset) + 2 * max(event_regressions))
            self.max_length = max_length
        else:
            self.max_length = None
            self.event_regression_samples = {}
            self.shuffle_indices = None
            series_ids_with_events = []
            for series_id in series_ids:
                series_events = events.loc[events["series_id"] == series_id]
                if len(series_events) == 0:
                    continue

                series_ids_with_events.append(series_id)
                self.event_regression_samples[series_id] = []

                for night in series_events["night"].unique():
                    night_events = series_events.loc[series_events["night"] == night]
                    if len(night_events) == 2:
                        assert night_events.iloc[0]["event"] == "onset"
                        assert night_events.iloc[1]["event"] == "wakeup"
                        onset = int(night_events.iloc[0]["step"])
                        wakeup = int(night_events.iloc[1]["step"])
                        self.event_regression_samples[series_id].append({
                            "onset": onset,
                            "wakeup": wakeup
                        })
            self.series_ids = series_ids_with_events

        self.use_kernel = use_kernel


    def shuffle(self):
        if self.train_or_test == "train":
            self.shuffle_indices = np.random.permutation(len(self.event_regression_samples))
        else:
            if self.shuffle_indices is None:
                self.shuffle_indices = np.arange(len(self.series_ids))
        self.sample_low = 0

    def sample_single(self, index: int, target_length: int):
        assert target_length >= self.max_length, "target_length must be at least {}".format(self.max_length)

        event_info = self.event_regression_samples[index]
        series_id = event_info["series_id"]
        onset = event_info["onset"]
        wakeup = event_info["wakeup"]

        # Expand randomly to match target length
        series_length = self.naive_all_data[series_id]["accel"].shape[1]
        start = onset - max(self.event_regressions)
        end = wakeup + max(self.event_regressions)
        start, end = max(0, start), min(series_length, end)

        expand_max, expand_min = min(start, target_length - (end - start)), max(target_length - (end - start) - (series_length - end), 0)
        if expand_max == expand_min:
            left_expand = expand_max
        else:
            left_expand = np.random.randint(expand_min, expand_max + 1)
        right_expand = target_length - (end - start) - left_expand
        start -= left_expand
        end += right_expand
        assert end - start == target_length, "end - start must be equal to target_length"
        assert start >= 0 and end <= series_length, "start and end must be within series_length"
        assert start <= onset and end >= wakeup, "start must be less than onset and end must be greater than wakeup"

        # Shift onset and wakeup
        onset, wakeup = onset - start, wakeup - start

        # Load acceleration data and event segmentations
        accel_data = self.naive_all_data[series_id]["accel"][:, start:end]

        # Load event regression data
        event_regression_values = [np.zeros((2, end - start), dtype=np.float32) for _ in self.event_regressions]
        event_regression_mask = [np.zeros((2, end - start), dtype=np.float32) for _ in self.event_regressions]

        for k in range(len(self.event_regressions)):
            regression_range = self.event_regressions[k]
            if self.use_kernel is not None:
                create_regression_kernel(event_regression_mask[k], event_regression_values[k], event_type=0,
                                        location=onset, window_radius=regression_range, kernel_radius=self.use_kernel)
                create_regression_kernel(event_regression_mask[k], event_regression_values[k], event_type=1,
                                        location=wakeup, window_radius=regression_range, kernel_radius=self.use_kernel)
            else:
                create_regression_range(event_regression_mask[k], event_regression_values[k], event_type=0, location=onset, window_radius=regression_range)
                create_regression_range(event_regression_mask[k], event_regression_values[k], event_type=1, location=wakeup, window_radius=regression_range)

        return accel_data, event_regression_values, event_regression_mask

    def sample(self, batch_size: int, target_length: int):
        assert self.train_or_test == "train", "sample() can only be called on train data"
        assert self.shuffle_indices is not None, "shuffle_indices is None, call shuffle() first"

        accel_datas = []
        event_regression_values = [[] for _ in self.event_regressions]
        event_regression_masks = [[] for _ in self.event_regressions]

        increment = min(batch_size, len(self.event_regression_samples) - self.sample_low)

        for k in range(self.sample_low, self.sample_low + increment):
            accel_data, event_regression_value, event_regression_mask =\
                self.sample_single(self.shuffle_indices[k], target_length)
            accel_datas.append(accel_data)
            for k in range(len(self.event_regressions)):
                event_regression_values[k].append(event_regression_value[k])
                event_regression_masks[k].append(event_regression_mask[k])

        self.sample_low += increment
        accel_datas = np.stack(accel_datas, axis=0)
        for k in range(len(self.event_regressions)):
            event_regression_values[k] = np.stack(event_regression_values[k], axis=0)
            event_regression_masks[k] = np.stack(event_regression_masks[k], axis=0)

        return accel_datas, event_regression_values, event_regression_masks, increment

    def sample_series(self, target_multiple: int):
        assert self.train_or_test == "test", "sample_series() can only be called on test data"
        assert self.shuffle_indices is not None, "shuffle_indices is None, call shuffle() first"

        series_id = self.series_ids[self.shuffle_indices[self.sample_low]]
        accel_data = self.naive_all_data[series_id]["accel"]

        # Load event regression data
        event_regression_values = [np.zeros((2, accel_data.shape[1]), dtype=np.float32) for _ in self.event_regressions]
        event_regression_mask = [np.zeros((2, accel_data.shape[1]), dtype=np.float32) for _ in self.event_regressions]

        for k in range(len(self.event_regressions)):
            regression_range = self.event_regressions[k]
            for event_info in self.event_regression_samples[series_id]:
                onset = event_info["onset"]
                wakeup = event_info["wakeup"]
                if self.use_kernel is not None:
                    create_regression_kernel(event_regression_mask[k], event_regression_values[k], event_type=0,
                                             location=onset, window_radius=regression_range,
                                             kernel_radius=self.use_kernel)
                    create_regression_kernel(event_regression_mask[k], event_regression_values[k], event_type=1,
                                             location=wakeup, window_radius=regression_range,
                                             kernel_radius=self.use_kernel)
                else:
                    create_regression_range(event_regression_mask[k], event_regression_values[k], event_type=0, location=onset,
                                            window_radius=regression_range)
                    create_regression_range(event_regression_mask[k], event_regression_values[k], event_type=1, location=wakeup,
                                            window_radius=regression_range)

        # Contract equally on both sides so that the length is a multiple of target_multiple
        series_length = accel_data.shape[1]
        target_length = (series_length // target_multiple) * target_multiple
        start = (series_length - target_length) // 2
        end = start + target_length
        accel_data = accel_data[:, start:end]

        for k in range(len(self.event_regressions)):
            event_regression_values[k] = event_regression_values[k][:, start:end]
            event_regression_mask[k] = event_regression_mask[k][:, start:end]
            event_regression_values[k] = np.expand_dims(event_regression_values[k], axis=0)
            event_regression_mask[k] = np.expand_dims(event_regression_mask[k], axis=0)

        self.sample_low += 1
        shifted_samples = []
        for event in self.event_regression_samples[series_id]:
            shifted_samples.append({"onset": event["onset"] - start, "wakeup": event["wakeup"] - start})

        return np.expand_dims(accel_data, axis=0), event_regression_values, event_regression_mask, shifted_samples

    def __len__(self):
        if self.train_or_test == "train":
            return len(self.event_regression_samples)
        else:
            return len(self.series_ids)

    def entries_remaining(self):
        if self.train_or_test == "train":
            return len(self.event_regression_samples) - self.sample_low
        else:
            return len(self.series_ids) - self.sample_low
