# You should run convert_to_npy_naive.py first.
import os
import time

import tqdm
import pandas as pd
import numpy as np
import stumpy
from numba import cuda

import convert_to_npy_naive
import bad_series_list

FOLDER = "data_matrix_profile_detailed"

if __name__ == "__main__":
    interested_series_ids = bad_series_list.noisy_bad_segmentations + bad_series_list.bad_segmentations_tail
    all_naive_data = convert_to_npy_naive.load_all_data_into_dict()

    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)

    all_gpu_devices = [device.id for device in cuda.list_devices()]

    lengths = []
    all_series = []
    for series_id in tqdm.tqdm(interested_series_ids):
        accel = all_naive_data[series_id]["accel"]
        anglez = accel[0, :]

        lengths.append(len(anglez))
        all_series.append(anglez)

    # compute matrix profile
    all_series = np.concatenate(all_series, axis=0)
    matrix_profile = stumpy.gpu_stump(all_series.astype(np.float64), m=4320, device_id=all_gpu_devices[0])[:, 0].astype(np.float32)
    matrix_profile = np.pad(matrix_profile, (0, 4320 - 1), mode="constant", constant_values=matrix_profile[-1])  # subsequence stride right

    # upsample and revert
    assert matrix_profile.dtype == np.float32

    # save matrix profile
    start = 0
    for i, series_id in tqdm.tqdm(enumerate(interested_series_ids)):
        np.save(os.path.join(FOLDER, series_id + ".npy"), matrix_profile[start:start + lengths[i]])
        start += lengths[i]

