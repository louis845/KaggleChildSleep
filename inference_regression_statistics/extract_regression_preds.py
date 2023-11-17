import os
import shutil
import sys

import numpy as np
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add root folder to sys.path
import postprocessing

out_folder = "regression_preds"
if os.path.isdir(out_folder):
    shutil.rmtree(out_folder)
os.mkdir(out_folder)

PREDS = "Standard_5CV"
kernel_shape = "laplace"
kernel_size = 6
cutoff = 5.0
#kernel_shape = "gaussian"
#kernel_size = 9
#cutoff = 4.5
pruning = 60
alignment = True

preds_folder = os.path.join("regression_labels", PREDS, "{}_kernel{}".format(kernel_shape, kernel_size))

"""cutoff = 2.0
pruning = 60
alignment = True
preds_folder = os.path.join("regression_labels", "Standard_5CV_Sigmas_VElastic", "gaussian_kernel")"""
series_ids = [x.split(".")[0] for x in os.listdir("../individual_train_series")]
for series_id in tqdm.tqdm(series_ids):
    onset_locmax_file = os.path.join(preds_folder, "{}_onset_locmax.npy".format(series_id))
    wakeup_locmax_file = os.path.join(preds_folder, "{}_wakeup_locmax.npy".format(series_id))

    onset_locmax = np.load(onset_locmax_file)
    wakeup_locmax = np.load(wakeup_locmax_file)

    onset_locs = onset_locmax[0, :]
    onset_values = onset_locmax[1, :]
    wakeup_locs = wakeup_locmax[0, :]
    wakeup_values = wakeup_locmax[1, :]

    onset_locs = onset_locs[onset_values > cutoff].astype(np.int32)
    wakeup_locs = wakeup_locs[wakeup_values > cutoff].astype(np.int32)

    if pruning > 0:
        if len(onset_locs) > 0:
            onset_locs = postprocessing.prune(onset_locs, onset_values[onset_values > cutoff], pruning)
        if len(wakeup_locs) > 0:
            wakeup_locs = postprocessing.prune(wakeup_locs, wakeup_values[wakeup_values > cutoff], pruning)

    if alignment:
        seconds_values = np.load("../data_naive/{}/secs.npy".format(series_id))
        first_zero = postprocessing.compute_first_zero(seconds_values)
        if len(onset_locs) > 0:
            onset_kernel_values = np.load(os.path.join(preds_folder, "{}_onset.npy".format(series_id)))
            onset_locs = postprocessing.align_predictions(onset_locs, onset_kernel_values, first_zero=first_zero)
        if len(wakeup_locs) > 0:
            wakeup_kernel_values = np.load(os.path.join(preds_folder, "{}_wakeup.npy".format(series_id)))
            wakeup_locs = postprocessing.align_predictions(wakeup_locs, wakeup_kernel_values, first_zero=first_zero)

    np.save(os.path.join(out_folder, "{}_onset_locs.npy".format(series_id)), onset_locs)
    np.save(os.path.join(out_folder, "{}_wakeup_locs.npy".format(series_id)), wakeup_locs)
