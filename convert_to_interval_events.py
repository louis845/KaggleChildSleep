import numpy as np
import pandas as pd

import bad_series_list
import transform_elastic_deformation

def set_kernel_range(arr, idx, loc, kernel_shape, kernel_radius):
    assert kernel_shape in ["gaussian", "laplace"]

    if kernel_shape == "gaussian":
        replace_radius = 3 * kernel_radius
    elif kernel_shape == "laplace":
        replace_radius = 5 * kernel_radius

    replace_start = max(0, loc - replace_radius)
    replace_end = min(arr.shape[1], loc + replace_radius + 1)

    if kernel_shape == "gaussian":
        kernel = np.exp(-np.square(np.arange(replace_start - loc, replace_end - loc, dtype=np.float32) / kernel_radius))
    elif kernel_shape == "laplace":
        kernel = np.exp(-np.abs(np.arange(replace_start - loc, replace_end - loc, dtype=np.float32) / kernel_radius))

    arr[idx, replace_start:replace_end] = kernel


def get_first_day_step(naive_all_data, series_id):
    hours = naive_all_data[series_id]["hours"]
    mins = naive_all_data[series_id]["mins"]
    secs = naive_all_data[series_id]["secs"]

    night_onset = (hours == 17) & (mins == 0) & (secs == 0) # 1700
    first_day_step = np.argwhere(night_onset).flatten()[0]
    return first_day_step

def get_truncated_series_length(naive_all_data, series_id, events):
    series_max = len(naive_all_data[series_id]["hours"])
    if series_id in bad_series_list.bad_segmentations_tail:
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

    def sample_single(self, index: int, random_shift: int=0, flip: bool=False, vflip=False, expand: int=0,
                      elastic_deformation=False, v_elastic_deformation=False, randomly_augment_time=False,
                      randomly_dropout_expanded_parts=False,

                      kernel_mode="constant", event_tolerance_width = 30 * 12):
        # index denotes the index in self.all_segmentations_list
        # vflip and v_elastic_deformation is applied to anglez only, not to enmo
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

        # find the events that need to be included, and generate elastic deformation if necessary
        events_start, events_end = start - expand, end + expand
        if elastic_deformation:
            deformation_indices = transform_elastic_deformation.generate_deformation_indices(length=end - start + expand * 2)
            events_start, events_end = start - expand + deformation_indices[0], start - expand + deformation_indices[-1]

        # generate expanded parts dropout if necessary
        dropout_expanded_parts = False
        if randomly_dropout_expanded_parts:
            if np.random.rand() < 0.1:
                dropout_expanded_parts = True
                left_dropout, right_dropout = 0, 0
                if np.random.rand() < 0.5:
                    left_dropout = np.random.randint(low=0, high=expand)
                else:
                    right_dropout = np.random.randint(low=0, high=expand)

        # load events
        series_events = self.events.loc[self.events["series_id"] == series_id]
        events_contained = series_events.loc[
            ((series_events["step"] >= events_start) & (series_events["step"] < events_end))]
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
                else:
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
        if (vflip or v_elastic_deformation) and (not elastic_deformation):
            accel_data = accel_data.copy()
        event_segmentations = np.zeros((2, end - start + 2 * expand), dtype=np.float32)
        for event in grouped_events:
            onset = (int(event["onset"] - start + expand)) if event["onset"] is not None else None
            wakeup = (int(event["wakeup"] - start + expand)) if event["wakeup"] is not None else None
            if elastic_deformation:
                # compute position after elastic deformation
                onset = transform_elastic_deformation.find_closest_index(deformation_indices, onset) if onset is not None else None
                wakeup = transform_elastic_deformation.find_closest_index(deformation_indices, wakeup) if wakeup is not None else None

            if flip:
                onset, wakeup = wakeup, onset
            if onset is not None:
                if (not dropout_expanded_parts) or (onset >= left_dropout and onset <= (end - start + 2 * expand - right_dropout)):
                    if kernel_mode == "constant":
                        event_segmentations[0, max(0, onset - event_tolerance_width):min(event_segmentations.shape[1],
                                                                                         onset + event_tolerance_width + 1)] = 1.0
                    else:
                        set_kernel_range(event_segmentations, 0, onset, kernel_mode, kernel_radius=event_tolerance_width)
            if wakeup is not None:
                if (not dropout_expanded_parts) or (wakeup >= left_dropout and wakeup <= (end - start + 2 * expand - right_dropout)):
                    if kernel_mode == "constant":
                        event_segmentations[1, max(0, wakeup - event_tolerance_width):min(event_segmentations.shape[1],
                                                                                          wakeup + event_tolerance_width + 1)] = 1.0
                    else:
                        set_kernel_range(event_segmentations, 1, wakeup, kernel_mode, kernel_radius=event_tolerance_width)
        if elastic_deformation:
            accel_data = transform_elastic_deformation.deform_time_series(accel_data, deformation_indices)
        if vflip:
            # flip anglez only
            accel_data[0, :] = -accel_data[0, :]
        if v_elastic_deformation:
            accel_data[0, :] = transform_elastic_deformation.deform_v_time_series(accel_data[0, :])
        if dropout_expanded_parts:
            if left_dropout > 0:
                accel_data[:, :left_dropout] = 0
            elif right_dropout > 0:
                accel_data[:, -right_dropout:] = 0
        if flip:
            accel_data = np.flip(accel_data, axis=1)
            event_segmentations = np.flip(event_segmentations, axis=1)

        hour = self.naive_all_data[series_id]["hours"][start - expand]
        minute = self.naive_all_data[series_id]["mins"][start - expand]
        second = self.naive_all_data[series_id]["secs"][start - expand]
        time = (hour * 3600 + minute * 60 + second) // 5
        if randomly_augment_time:
            time += np.random.randint(-360, 361)
            if time < 0:
                time += 17280
            time = time % 17280

        return accel_data, event_segmentations, time

    def sample(self, batch_size: int, random_shift: int=0, random_flip: bool=False, always_flip: bool=False, random_vflip=False, expand: int=0,
               elastic_deformation=False, v_elastic_deformation=False, randomly_dropout_expanded_parts=False, kernel_mode="constant", event_tolerance_width = 30 * 12):
        assert kernel_mode in ["constant", "gaussian", "laplace"]

        assert self.shuffle_indices is not None, "shuffle_indices is None, call shuffle() first"

        accel_datas = []
        event_segmentations = []
        times = []

        increment = min(batch_size, len(self.all_segmentations_list) - self.sample_low)

        for k in range(self.sample_low, self.sample_low + increment):
            flip, vflip = False, False
            if random_flip:
                flip = np.random.randint(0, 2) == 1
            if always_flip:
                flip = True
            if random_vflip:
                vflip = np.random.randint(0, 2) == 1

            accel_data, event_segmentation, time = self.sample_single(self.shuffle_indices[k], random_shift=random_shift, flip=flip, vflip=vflip, expand=expand, elastic_deformation=elastic_deformation,
                                                                      v_elastic_deformation=v_elastic_deformation, randomly_augment_time=self.train_or_test == "train",
                                                                      randomly_dropout_expanded_parts=randomly_dropout_expanded_parts, kernel_mode=kernel_mode, event_tolerance_width=event_tolerance_width)
            accel_datas.append(accel_data)
            event_segmentations.append(event_segmentation)
            times.append(time)

        self.sample_low += increment
        accel_datas = np.stack(accel_datas, axis=0)
        event_segmentations = np.stack(event_segmentations, axis=0)
        times = np.array(times, dtype=np.int32)

        return accel_datas, event_segmentations, times, increment

    def __len__(self):
        return len(self.all_segmentations_list)

    def entries_remaining(self):
        return len(self.all_segmentations_list) - self.sample_low
