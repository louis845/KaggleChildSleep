import numpy as np

def align_and_augment_predictions(preds_locs, preds_local_kernel, preds_probas, series_secs):
    total_length = len(series_secs)

    first_zero = np.argwhere(series_secs == 0).flatten()[0]
    align_mod6 = (first_zero + 3) % 6

    # align predictions
    preds_locs1 = preds_locs + np.mod(align_mod6 - preds_locs, 6)
    preds_locs2 = preds_locs1 - 6
    out_of_bounds = preds_locs1 >= total_length
    preds_locs1[out_of_bounds] = preds_locs2[out_of_bounds]
    out_of_bounds = preds_locs2 < 0
    preds_locs2[out_of_bounds] = preds_locs1[out_of_bounds]
    preds_locs_new = np.where(preds_local_kernel[preds_locs1] > preds_local_kernel[preds_locs2], preds_locs1, preds_locs2)

    preds_locs = np.where(np.mod(preds_locs, 6) == align_mod6, preds_locs, preds_locs_new)

    # augment predictions
    left_shift_locs = preds_locs - 24
    right_shift_locs = preds_locs + 24

