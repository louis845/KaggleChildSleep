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

PREDS = ["Standard_5CV"]
kernel_shape = ["laplace"]
kernel_size = [6]
cutoff = 5.0
#PREDS = ["Standard_5CV", "Standard_5CV_Mid", "Standard_5CV_Wide"]
#kernel_shape = ["gaussian", "gaussian", "gaussian"]
#kernel_size = [9, 9, 9]
#cutoff = 4.5
pruning = 60
alignment = True

preds_folder = [os.path.join("regression_labels", PREDS[k], "{}_kernel{}".format(kernel_shape[k], kernel_size[k])) for k in range(len(PREDS))]

"""cutoff = 2.0
pruning = 60
alignment = True
preds_folder = os.path.join("regression_labels", "Standard_5CV_Sigmas_VElastic", "gaussian_kernel")"""
series_ids = [x.split(".")[0] for x in os.listdir("../individual_train_series")]
for series_id in tqdm.tqdm(series_ids):
    onset_kernel_values, wakeup_kernel_values = None, None

    for k in range(len(PREDS)):
        onset_kernel_file = os.path.join(preds_folder[k], "{}_onset.npy".format(series_id))
        wakeup_kernel_file = os.path.join(preds_folder[k], "{}_wakeup.npy".format(series_id))

        if k == 0:
            onset_kernel_values = np.load(os.path.join(preds_folder[k], "{}_onset.npy".format(series_id)))
            wakeup_kernel_values = np.load(os.path.join(preds_folder[k], "{}_wakeup.npy".format(series_id)))
        else:
            onset_kernel_values = onset_kernel_values + np.load(os.path.join(preds_folder[k], "{}_onset.npy".format(series_id)))
            wakeup_kernel_values = wakeup_kernel_values + np.load(os.path.join(preds_folder[k], "{}_wakeup.npy".format(series_id)))
    if len(PREDS) > 1:
        onset_kernel_values = onset_kernel_values / len(PREDS)
        wakeup_kernel_values = wakeup_kernel_values / len(PREDS)

    onset_locs = (onset_kernel_values[1:-1] > onset_kernel_values[0:-2]) & (onset_kernel_values[1:-1] > onset_kernel_values[2:])
    onset_locs = np.argwhere(onset_locs).flatten() + 1
    wakeup_locs = (wakeup_kernel_values[1:-1] > wakeup_kernel_values[0:-2]) & (wakeup_kernel_values[1:-1] > wakeup_kernel_values[2:])
    wakeup_locs = np.argwhere(wakeup_locs).flatten() + 1

    onset_values = onset_kernel_values[onset_locs]
    wakeup_values = wakeup_kernel_values[wakeup_locs]

    onset_locs = onset_locs[onset_values > cutoff]
    wakeup_locs = wakeup_locs[wakeup_values > cutoff]

    if pruning > 0:
        if len(onset_locs) > 0:
            onset_locs = postprocessing.prune(onset_locs, onset_values[onset_values > cutoff], pruning)
        if len(wakeup_locs) > 0:
            wakeup_locs = postprocessing.prune(wakeup_locs, wakeup_values[wakeup_values > cutoff], pruning)

    if alignment:
        seconds_values = np.load("../data_naive/{}/secs.npy".format(series_id))
        first_zero = postprocessing.compute_first_zero(seconds_values)
        if len(onset_locs) > 0:
            onset_locs = postprocessing.align_predictions(onset_locs, onset_kernel_values, first_zero=first_zero)
        if len(wakeup_locs) > 0:
            wakeup_locs = postprocessing.align_predictions(wakeup_locs, wakeup_kernel_values, first_zero=first_zero)

    np.save(os.path.join(out_folder, "{}_onset_locs.npy".format(series_id)), onset_locs)
    np.save(os.path.join(out_folder, "{}_wakeup_locs.npy".format(series_id)), wakeup_locs)
