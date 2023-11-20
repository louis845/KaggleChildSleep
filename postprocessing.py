import numpy as np

def prune(event_locs, event_vals, pruning_radius):
    # assumes the indices (event_locs) are sorted
    # event locs same length as event vals, prunes event locs according to radius
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

def index_out_of_bounds(arr, indices):
    # assumes that the indices are sorted
    start = np.searchsorted(indices, 0, side="left")
    end = np.searchsorted(indices, len(arr), side="left")
    return np.pad(arr[indices[start:end]], (start, len(indices) - end), mode="constant", constant_values=-1)

def edges_detect(preds):
    assert preds.dtype == bool, "preds must be a boolean array"
    assert len(preds.shape) == 1, "preds must be a 1D array"

    start_edges = (~preds[:-1]) & preds[1:]
    end_edges = preds[:-1] & (~preds[1:])

    start_indices = np.argwhere(start_edges).flatten() + 1
    end_indices = np.argwhere(end_edges).flatten() + 1

    if preds[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if preds[-1]:
        end_indices = np.append(end_indices, len(preds))

    return start_indices, end_indices

def remove_short_gaps(start_indices, end_indices, gap_threshold=10):
    assert len(start_indices.shape) == 1, "start_indices must be a 1D array"
    assert len(end_indices.shape) == 1, "end_indices must be a 1D array"
    assert len(start_indices) == len(end_indices), "start_indices and end_indices must have the same length"

    if len(start_indices) <= 1:
        return start_indices, end_indices

    d = start_indices[1:] - end_indices[:-1]
    good = d >= gap_threshold

    start_indices = start_indices[np.concatenate([[True], good])]
    end_indices = end_indices[np.concatenate([good, [True]])]
    return start_indices, end_indices

def index_probas_distance_based(locs, pred_probas, proba_threshold: float, dropoff_factor: float,
                                preds_above_threshold: np.ndarray=None):
    if preds_above_threshold is None:
        preds_above_threshold = pred_probas >= proba_threshold



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
                                           increase_preds_locs[choice_locs], decrease_preds_locs[choice_locs])
    if np.any(increase_locs) or np.any(decrease_locs):
        increase_oob = increase_preds_kernel_val == -1
        decrease_oob = decrease_preds_kernel_val == -1
        increase_locs2 = (increase_locs | decrease_oob) & (~increase_oob)
        decrease_locs2 = (decrease_locs | increase_oob) & (~decrease_oob)

        preds_locs[increase_locs2] = increase_preds_locs[increase_locs2]
        preds_locs[decrease_locs2] = decrease_preds_locs[decrease_locs2]

    return np.unique(preds_locs)

def augment_left_right(preds_locs, preds_local_kernel):
    preds_left = preds_locs - 12
    preds_right = preds_locs + 12

    left_kernel_val = index_out_of_bounds(preds_local_kernel, preds_left)
    right_kernel_val = index_out_of_bounds(preds_local_kernel, preds_right)

    augmented_locs1 = np.where(left_kernel_val > right_kernel_val, preds_left, preds_right)
    augmented_locs2 = np.where(left_kernel_val > right_kernel_val, preds_right, preds_left)[(left_kernel_val != -1) & (right_kernel_val != -1)]

    return augmented_locs1, augmented_locs2

def augment_and_cat(preds_locs, preds_local_kernel, preds_probas, divisor: float, shifts: list[float]):
    augmented_locs1, augmented_locs2 = augment_left_right(preds_locs, preds_local_kernel)
    preds_locs_probas = preds_probas[preds_locs]
    augmented_locs_probas1 = preds_probas[augmented_locs1]
    augmented_locs_probas2 = preds_probas[augmented_locs2]

    all_preds_locs = np.concatenate([preds_locs, augmented_locs1, augmented_locs2])
    all_preds_probas = np.concatenate([preds_locs_probas / divisor + shifts[0],
                                       augmented_locs_probas1 / divisor + shifts[1],
                                       augmented_locs_probas2 / divisor + shifts[2]])
    return all_preds_locs, all_preds_probas

def get_augmented_predictions(preds_locs, preds_local_kernel, preds_probas, cutoff_thresh: float):
    if len(preds_locs) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    if cutoff_thresh == 0.0:
        all_preds_locs, all_preds_probas = augment_and_cat(preds_locs, preds_local_kernel, preds_probas,
                                                           3.0, [2.0 / 3, 1.0 / 3, 0.0])
    else:
        preds_locs_below = preds_locs[preds_probas[preds_locs] < cutoff_thresh]
        preds_locs_above = preds_locs[preds_probas[preds_locs] >= cutoff_thresh]

        if len(preds_locs_above) == 0: # all below
            all_preds_locs, all_preds_probas = augment_and_cat(preds_locs, preds_local_kernel, preds_probas,
                                                               6.0, [2.0 / 6, 1.0 / 6, 0.0])
        elif len(preds_locs_below) == 0: # all above
            all_preds_locs, all_preds_probas = augment_and_cat(preds_locs, preds_local_kernel, preds_probas,
                                                               6.0, [5.0 / 6, 4.0 / 6, 3.0 / 6])
        else:
            all_above_preds_locs, all_above_preds_probas = augment_and_cat(preds_locs_above, preds_local_kernel, preds_probas,
                                                               6.0, [5.0 / 6, 4.0 / 6, 3.0 / 6])
            all_below_preds_locs, all_below_preds_probas = augment_and_cat(preds_locs_below, preds_local_kernel, preds_probas,
                                                               6.0, [2.0 / 6, 1.0 / 6, 0.0])
            all_preds_locs = np.concatenate([all_above_preds_locs, all_below_preds_locs])
            all_preds_probas = np.concatenate([all_above_preds_probas, all_below_preds_probas])
    sort_index = np.argsort(all_preds_locs)
    all_preds_locs, all_preds_probas = all_preds_locs[sort_index], all_preds_probas[sort_index]
    return all_preds_locs, all_preds_probas

def align_and_augment_predictions(preds_locs, preds_local_kernel, preds_probas, series_secs, cutoff_thresh: float):
    if len(preds_locs) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    first_zero = compute_first_zero(series_secs)
    preds_locs = align_predictions(preds_locs, preds_local_kernel, first_zero)

    return get_augmented_predictions(preds_locs, preds_local_kernel, preds_probas, cutoff_thresh)
