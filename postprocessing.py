import numpy as np

def index_out_of_bounds(arr, indices):
    # assumes that the indices are sorted
    start = np.searchsorted(indices, 0, side="left")
    end = np.searchsorted(indices, len(arr), side="left")
    return np.pad(arr[indices[start:end]], (start, len(indices) - end), mode="constant", constant_values=-1)

def compute_first_zero(series_secs):
    return np.argwhere(series_secs == 0).flatten()[0]

def align_predictions(preds_locs, first_zero):
    align_mod6 = (first_zero + 3) % 6

    # align predictions
    moddiff = np.mod(align_mod6 - preds_locs, 6)
    choice_locs = moddiff == 0
    not_choice_locs = np.logical_not(choice_locs)

    preds_locs1 = preds_locs[not_choice_locs] + moddiff[not_choice_locs]
    preds_locs2 = preds_locs1 - 6
    out_of_bounds = preds_locs1 >= len()


    return preds_locs

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

