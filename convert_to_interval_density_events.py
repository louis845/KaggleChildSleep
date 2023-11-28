import numpy as np
import pandas as pd

import bad_series_list
import convert_to_interval_events
import transform_elastic_deformation

class IntervalDensityEventsSampler:
    def __init__(self, series_ids: list[str], naive_all_data: dict,
                 input_length_multiple: int,
                 train_or_test="train",
                 prediction_length=17280, # 24 hours
                 prediction_stride=4320, # 6 hours
                 is_enmo_only=False,
                 donot_exclude_bad_series_from_training=False
                 ):
        assert prediction_length % input_length_multiple == 0, "prediction_length must be a multiple of input_length_multiple"
        assert prediction_stride % input_length_multiple == 0, "prediction_stride must be a multiple of input_length_multiple"

        self.series_ids = series_ids
        self.naive_all_data = naive_all_data

        events = pd.read_csv("data/train_events.csv")
        self.events = events.dropna()

        self.all_segmentations = {}
        self.all_segmentations_list = []
        for series_id in series_ids:
            self.all_segmentations[series_id] = []

            interval_min = convert_to_interval_events.get_first_day_step(naive_all_data, series_id)
            if donot_exclude_bad_series_from_training:
                interval_max = naive_all_data[series_id]["accel"].shape[1]
            else:
                interval_max = convert_to_interval_events.get_truncated_series_length(naive_all_data, series_id, self.events)

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
        self.is_enmo_only = is_enmo_only
        self.input_length_multiple = input_length_multiple


    def shuffle(self):
        if self.train_or_test == "train":
            self.shuffle_indices = np.random.permutation(len(self.all_segmentations_list))
        else:
            if self.shuffle_indices is None:
                self.shuffle_indices = np.arange(len(self.all_segmentations_list))
        self.sample_low = 0

    def sample_single(self, index: int, random_shift: int=0, flip: bool=False, vflip=False, expand: int=0,
                      elastic_deformation=False, v_elastic_deformation=False, randomly_augment_time=False,

                      return_mode="interval_density"):
        event_tolerance_width = 5 * 12

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
        event_segmentations_downscaled = np.zeros((2, (end - start + 2 * expand) // self.input_length_multiple), dtype=np.float32)

        has_onset, has_wakeup = False, False
        has_onset_in_center, has_wakeup_in_center = False, False
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
                onset_downscaled = onset // self.input_length_multiple
                event_segmentations_downscaled[0, max(onset_downscaled - 2, 0):min(onset_downscaled + 3, event_segmentations_downscaled.shape[-1])] = 1.0
                convert_to_interval_events.set_kernel_range(event_segmentations, idx=0, loc=onset,
                                                            kernel_shape="laplace", kernel_radius=30, replace_radius=event_tolerance_width)
                if 0 <= onset < event_segmentations.shape[1]:
                    has_onset = True
                if expand <= onset < event_segmentations.shape[1] - expand:
                    has_onset_in_center = True
            if wakeup is not None:
                wakeup_downscaled = wakeup // self.input_length_multiple
                event_segmentations_downscaled[1, max(wakeup_downscaled - 2, 0):min(wakeup_downscaled + 3, event_segmentations_downscaled.shape[-1])] = 1.0
                convert_to_interval_events.set_kernel_range(event_segmentations, idx=1, loc=wakeup,
                                                            kernel_shape="laplace", kernel_radius=30, replace_radius=event_tolerance_width)
                if 0 <= wakeup < event_segmentations.shape[1]:
                    has_wakeup = True
                if expand <= wakeup < event_segmentations.shape[1] - expand:
                    has_wakeup_in_center = True
        if elastic_deformation:
            accel_data = transform_elastic_deformation.deform_time_series(accel_data, deformation_indices)
        if vflip:
            # flip anglez only
            accel_data[0, :] = -accel_data[0, :]
        if v_elastic_deformation:
            if self.is_enmo_only:
                accel_data[1, :] = transform_elastic_deformation.deform_v_time_series_enmo(accel_data[1, :])
            else:
                accel_data[0, :] = transform_elastic_deformation.deform_v_time_series(accel_data[0, :])

        if flip:
            accel_data = np.flip(accel_data, axis=1)
            event_segmentations = np.flip(event_segmentations, axis=1)
            event_segmentations_downscaled = np.flip(event_segmentations_downscaled, axis=1)

        hour = self.naive_all_data[series_id]["hours"][start - expand]
        minute = self.naive_all_data[series_id]["mins"][start - expand]
        second = self.naive_all_data[series_id]["secs"][start - expand]
        time = (hour * 3600 + minute * 60 + second) // 5
        if randomly_augment_time:
            time += np.random.randint(-360, 361)
            if time < 0:
                time += 17280
            time = time % 17280

        if return_mode == "expanded_interval_density":
            if has_onset:
                onset_density = event_segmentations[0, :] / np.sum(event_segmentations[0, :])
            else:
                onset_density = np.full_like(event_segmentations[0, :], 1.0 / event_segmentations.shape[1])
            if has_wakeup:
                wakeup_density = event_segmentations[1, :] / np.sum(event_segmentations[1, :])
            else:
                wakeup_density = np.full_like(event_segmentations[1, :], 1.0 / event_segmentations.shape[1])

            event_info = {"density": np.stack([onset_density, wakeup_density], axis=0),
                          "occurrence": np.array([has_onset_in_center, has_wakeup_in_center], dtype=np.float32)}
        else:
            if has_onset_in_center:
                onset_density = event_segmentations[0, expand:-expand] / np.sum(event_segmentations[0, expand:-expand])
            else:
                onset_density = np.full_like(event_segmentations[0, expand:-expand], 1.0 / (end - start))
                assert event_segmentations.shape[1] == end - start + 2 * expand
            if has_wakeup_in_center:
                wakeup_density = event_segmentations[1, expand:-expand] / np.sum(event_segmentations[1, expand:-expand])
            else:
                wakeup_density = np.full_like(event_segmentations[1, expand:-expand], 1.0 / (end - start))
                assert event_segmentations.shape[1] == end - start + 2 * expand
            event_info = {"density": np.stack([onset_density, wakeup_density], axis=0),
                            "occurrence": np.array([has_onset_in_center, has_wakeup_in_center], dtype=np.float32),
                            "segmentation": event_segmentations_downscaled}

        return accel_data, event_info, time

    def sample(self, batch_size: int, random_shift: int=0, random_flip: bool=False, always_flip: bool=False, random_vflip=False, expand: int=0,
               elastic_deformation=False, v_elastic_deformation=False,

               return_mode="expanded_interval_density"):
        assert self.shuffle_indices is not None, "shuffle_indices is None, call shuffle() first"
        assert isinstance(return_mode, str), "return_mode must be a string"
        assert return_mode in ["expanded_interval_density", "interval_density_and_expanded_events"]

        accel_datas = []
        event_infos = {"density": [], "occurrence": []}
        times = []

        if return_mode == "interval_density_and_expanded_events":
            event_infos["segmentation"] = []

        increment = min(batch_size, len(self.all_segmentations_list) - self.sample_low)

        for k in range(self.sample_low, self.sample_low + increment):
            flip, vflip = False, False
            if random_flip:
                flip = np.random.randint(0, 2) == 1
            if always_flip:
                flip = True
            if random_vflip:
                vflip = np.random.randint(0, 2) == 1

            accel_data, event_info, time = self.sample_single(self.shuffle_indices[k], random_shift=random_shift, flip=flip, vflip=vflip, expand=expand, elastic_deformation=elastic_deformation,
                                                                      v_elastic_deformation=v_elastic_deformation, randomly_augment_time=self.train_or_test == "train",

                                                                      return_mode=return_mode)
            accel_datas.append(accel_data)
            for key in event_infos:
                event_infos[key].append(event_info[key])
            times.append(time)

        self.sample_low += increment
        accel_datas = np.stack(accel_datas, axis=0)
        for key in event_infos:
            event_infos[key] = np.stack(event_infos[key], axis=0)
        times = np.array(times, dtype=np.int32)

        return accel_datas, event_infos, times, increment

    def __len__(self):
        return len(self.all_segmentations_list)

    def entries_remaining(self):
        return len(self.all_segmentations_list) - self.sample_low
