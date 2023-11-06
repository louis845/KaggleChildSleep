import os
import shutil

import numpy as np
import tqdm

def prune(event_locs, event_vals, pruning_radius):
    descending_order = np.argsort(event_vals)[::-1]
    keeps = np.ones(len(event_locs), dtype=bool)

    for k in range(len(event_locs)):
        if keeps[descending_order[k]]:
            loc = event_locs[descending_order[k]]
            left_idx, right_idx = np.searchsorted(event_locs, [loc - pruning_radius + 1, loc + pruning_radius], side="left")
            if right_idx - left_idx > 1:
                keeps[left_idx:right_idx] = False
                keeps[descending_order[k]] = True
    return event_locs[keeps]

out_folder = "regression_preds"
if os.path.isdir(out_folder):
    shutil.rmtree(out_folder)
os.mkdir(out_folder)

PREDS = "Standard_5CV"
kernel_shape = "huber"
kernel_size = 6
cutoff = 5.0
pruning = 60

preds_folder = os.path.join("regression_labels", PREDS, "{}_kernel{}".format(kernel_shape, kernel_size))
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
            onset_locs = prune(onset_locs, onset_values[onset_values > cutoff], pruning)
        if len(wakeup_locs) > 0:
            wakeup_locs = prune(wakeup_locs, wakeup_values[wakeup_values > cutoff], pruning)

    np.save(os.path.join(out_folder, "{}_onset_locs.npy".format(series_id)), onset_locs)
    np.save(os.path.join(out_folder, "{}_wakeup_locs.npy".format(series_id)), wakeup_locs)