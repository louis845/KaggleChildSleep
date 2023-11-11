import numpy as np

def index_out_of_bounds(arr, indices):
    # assumes that the indices are sorted
    start = np.searchsorted(indices, 0, side="left")
    end = np.searchsorted(indices, len(arr), side="left")
    return np.pad(arr[indices[start:end]], (start, len(indices) - end), mode="constant", constant_values=-1)

def compute_first_zero(series_secs):
    return np.argwhere(series_secs == 0).flatten()[0]

def align_predictions(preds_locs, preds_local_kernel, first_zero):
    align_mod6 = (first_zero + 3) % 6

    # align predictions
    moddiff = np.mod(align_mod6 - preds_locs, 6)
    choice_locs = moddiff == 3
    increase_locs = (1 <= moddiff) & (moddiff <= 2)
    decrease_locs = (4 <= moddiff) & (moddiff <= 5)
    increase_preds_locs = preds_locs + moddiff
    decrease_preds_locs = increase_preds_locs - 6

    # get kernel values at aligned locations
    increase_preds_kernel_val = index_out_of_bounds(preds_local_kernel, increase_preds_locs)
    decrease_preds_kernel_val = index_out_of_bounds(preds_local_kernel, decrease_preds_locs)

    if np.any(choice_locs):
        # choose the one with the higher kernel value. this also handles out of bounds
        preds_locs[choice_locs] = np.where(increase_preds_kernel_val[choice_locs] > decrease_preds_kernel_val[choice_locs],
                                           increase_preds_kernel_val[choice_locs], decrease_preds_kernel_val[choice_locs])
    if np.any(increase_locs) or np.any(decrease_locs):
        increase_oob = increase_preds_kernel_val == -1
        decrease_oob = decrease_preds_kernel_val == -1
        increase_locs2 = (increase_locs | decrease_oob) & (~increase_oob)
        decrease_locs2 = (decrease_locs | increase_oob) & (~decrease_oob)

        preds_locs[increase_locs2] = increase_preds_locs[increase_locs2]
        preds_locs[decrease_locs2] = decrease_preds_locs[decrease_locs2]

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

