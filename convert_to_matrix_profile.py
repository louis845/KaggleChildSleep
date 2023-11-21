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

if __name__ == "__main__":
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

        # compute matrix profile
        matrix_profile = stumpy.gpu_stump(anglez.astype(np.float64), m=4320, device_id=all_gpu_devices[0])[:, 0]

        # save matrix profile
        np.save(os.path.join(FOLDER, series_id + ".npy"), matrix_profile)
        
        time_elapsed = time.time() - ctime

        log_file.write("{} Time Elapsed: {:.2f}s\n".format(series_id, time_elapsed))
        log_file.flush()
        all_times.append(time_elapsed)

        time.sleep(180)

    log_file.write("Average Time Elapsed: {:.2f}s\n".format(np.mean(all_times)))
    log_file.write("Median Time Elapsed: {:.2f}s\n".format(np.median(all_times)))
    log_file.close()

    with open(os.path.join(FOLDER, "finished.txt")) as f:
        f.write("finished")
