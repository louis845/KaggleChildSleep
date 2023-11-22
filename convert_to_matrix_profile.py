# You should run convert_to_npy_naive.py first.
import os
import time

import tqdm
import pandas as pd
import numpy as np
import stumpy
from numba import cuda

import convert_to_npy_naive

FOLDER = "data_matrix_profile"

def load_all_matrix_profiles() -> dict[str, np.ndarray]:
    all_series_ids = os.listdir("data_naive")
    all_matrix_profiles = {}
    for series_id in tqdm.tqdm(all_series_ids):
        matrix_profile = np.load(os.path.join(FOLDER, series_id + ".npy"))
        all_matrix_profiles[series_id] = matrix_profile
    return all_matrix_profiles

if __name__ == "__main__":
    rep_bad_series = []

    downsampling_rate = 12

    all_series_ids = os.listdir("data_naive")
    all_naive_data = convert_to_npy_naive.load_all_data_into_dict()

    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)

    log_file = open(os.path.join(FOLDER, "log.txt"), "w")
    all_times = []

    all_gpu_devices = [device.id for device in cuda.list_devices()]
    for series_id in tqdm.tqdm(all_series_ids):
        ctime = time.time()
        accel = all_naive_data[series_id]["accel"]
        anglez = accel[0, :]

        # compute left pad and downsample by pooling
        original_length = len(anglez)
        left_pad = original_length % downsampling_rate
        if left_pad != 0:
            left_pad = downsampling_rate - left_pad
        if left_pad > 0:
            anglez = np.pad(anglez, (left_pad, 0))
        anglez = anglez.reshape(-1, downsampling_rate).mean(axis=1)

        # compute matrix profile
        matrix_profile = stumpy.gpu_stump(anglez.astype(np.float64), m=4320 // downsampling_rate, device_id=all_gpu_devices[0])[:, 0].astype(np.float32)
        matrix_profile = np.pad(matrix_profile, (0, 4320 // downsampling_rate - 1), mode="constant",
                                constant_values=matrix_profile[-1]) # subsequence stride right

        # upsample and revert
        matrix_profile = np.repeat(matrix_profile, downsampling_rate)
        if left_pad > 0:
            matrix_profile = matrix_profile[left_pad:]
        assert len(matrix_profile) == original_length
        assert matrix_profile.dtype == np.float32

        # save matrix profile
        np.save(os.path.join(FOLDER, series_id + ".npy"), matrix_profile)
        
        time_elapsed = time.time() - ctime

        log_file.write("{} Time Elapsed: {:.2f}s\n".format(series_id, time_elapsed))
        log_file.flush()
        all_times.append(time_elapsed)

        # log bad series
        if np.any(matrix_profile < 1.0):
            rep_bad_series.append(series_id)

    log_file.write("Average Time Elapsed: {:.2f}s\n".format(np.mean(all_times)))
    log_file.write("Median Time Elapsed: {:.2f}s\n".format(np.median(all_times)))
    log_file.write("\n")
    log_file.write("Bad Series ({}):\n".format(len(rep_bad_series)))
    log_file.write(" ".join(rep_bad_series))
    log_file.close()

    print(rep_bad_series)

    """all_matrix_profiles = load_all_matrix_profiles()
    for series_id, matrix_profile in all_matrix_profiles.items():
        if np.any(matrix_profile < 0.000001):
            rep_bad_series.append(series_id)
    print(len(rep_bad_series))
    print(rep_bad_series)"""