import numpy as np

def find_closest(Z: np.ndarray, x: np.ndarray):
    """Given Z and x, find the indices i(j) such that Z[i(j)] is the closest element to x[j]"""
    # find the indices i such that Z[i] is the first element greater than or equal to x
    i = np.searchsorted(Z, x)
    # Handle cases where x[j] is greater than all elements in Z
    i[i == len(Z)] = len(Z) - 1
    # For all locations (i > 0), if x[j] is closer to Z[i - 1] than Z[i], decrement i
    # Note that when i = 0, then Z[i - 1] would be the last element, which doesn't make sense.
    # But it will be masked out by the mask anyway.
    mask = (i > 0) & ((np.abs(Z[i - 1] - x) <= np.abs(Z[i] - x)))
    i[mask] -= 1
    return i

def prune(event_locs, event_vals, pruning_radius, return_idx=False): # if return_idx, return a boolean array with the same length as event_locs
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

    if return_idx:
        return keeps
    return event_locs[keeps]

def prune_relative(reference_event_locs, event_locs, pruning_radius=60):
    prune_low = reference_event_locs - pruning_radius + 1
    prune_high = reference_event_locs + pruning_radius
    d = prune_low[1:] - prune_high[:-1]
    prune_low = prune_low[np.concatenate([[True], d > 0], axis=0)]
    prune_high = prune_high[np.concatenate([d > 0, [True]], axis=0)]

    low_idx = np.searchsorted(prune_low, event_locs, side="left")
    high_idx = np.searchsorted(prune_high, event_locs, side="right")

    return event_locs[low_idx != high_idx]

def align_index_probas(event_locs, probas_array, align_radius=60):
    event_probas = np.zeros(len(event_locs), dtype=np.float32)
    for k in range(len(event_locs)):
        low = max(event_locs[k] - align_radius + 1, 0)
        high = min(event_locs[k] + align_radius, len(probas_array))
        probas_of_interest = probas_array[low:high]
        idxmax = np.argmax(probas_of_interest)
        if 0 < idxmax < len(probas_of_interest) - 1:
            event_probas[k] = probas_of_interest[idxmax]
        else:
            event_probas[k] = probas_array[event_locs[k]]

    return event_probas

def prune_ROI_possible_probas(keeps, event_probas, left_idx, right_idx, interest_idx, cutoff):
    if right_idx - left_idx > 1:
        inradius_probas = event_probas[left_idx:right_idx]
        prunes = inradius_probas < cutoff
        if np.any(prunes):
            prunes_idx = np.argwhere(prunes).flatten() + left_idx
            keeps[prunes_idx] = False
            keeps[interest_idx] = True

def prune_ROI(event_locs, event_probas, pruning_radius, pruning_inner_radius, pruning_dropoff_factor):
    # assumes the indices (event_locs) are sorted
    # event locs same length as event probas, prunes event locs according to radius
    descending_order = np.argsort(event_probas)[::-1]
    keeps = np.ones(len(event_locs), dtype=bool)

    for k in range(len(event_locs)):
        interest_idx = descending_order[k]
        if keeps[interest_idx] and event_probas[interest_idx] > 0.5:
            loc = event_locs[interest_idx]
            if pruning_inner_radius == 0:
                left_idx, right_idx = np.searchsorted(event_locs, [loc - pruning_radius + 1, loc + pruning_radius], side="left")
                prune_ROI_possible_probas(keeps, event_probas, left_idx, right_idx, interest_idx, event_probas[interest_idx] * pruning_dropoff_factor)
            else:
                left_idx, right_idx = np.searchsorted(event_locs, [loc - pruning_radius + 1, loc - pruning_inner_radius], side="left")
                prune_ROI_possible_probas(keeps, event_probas, left_idx, right_idx, interest_idx, event_probas[interest_idx] * pruning_dropoff_factor)
                left_idx, right_idx = np.searchsorted(event_locs, [loc + 1 + pruning_inner_radius, loc + pruning_radius], side="left")
                prune_ROI_possible_probas(keeps, event_probas, left_idx, right_idx, interest_idx, event_probas[interest_idx] * pruning_dropoff_factor)

    return event_locs[keeps]


def prune_matrix_profile(event_locs, matrix_profile_vals, matrix_profile_thresh=0.01, matrix_profile_stride=4320,
                         return_idx=False): # if return idx, return a boolean array with same length as event_locs
    bad_locations = matrix_profile_vals < matrix_profile_thresh
    if not np.any(bad_locations):
        if return_idx:
            return np.full(len(event_locs), True, dtype=bool)
        else:
            return event_locs
    bad_locs_start, bad_locs_end = edges_detect(bad_locations)

    locs_idx_start = np.searchsorted(event_locs, bad_locs_start[0], side="left")
    locs_idx_end = np.searchsorted(event_locs, bad_locs_end[-1] + matrix_profile_stride, side="right")
    if locs_idx_end - locs_idx_start == 0:
        if return_idx:
            return np.full(len(event_locs), True, dtype=bool)
        else:
            return event_locs

    event_locs_of_interest = event_locs[locs_idx_start:locs_idx_end]

    idxs = np.searchsorted(bad_locs_start, event_locs_of_interest, side="right") - 1
    inside_bad_locs = (bad_locs_start[idxs] <= event_locs_of_interest) & (event_locs_of_interest < bad_locs_end[idxs] + matrix_profile_stride)
    good_locs = np.pad(~inside_bad_locs, (locs_idx_start, len(event_locs) - locs_idx_end), mode="constant", constant_values=True)

    if return_idx:
        return good_locs
    else:
        return event_locs[good_locs]

def prune_wakeup_at_heads(wakeup_event_locs, onset_probas, cutoff=0.5):
    assert np.all(wakeup_event_locs[:-1] <= wakeup_event_locs[1:]), "wakeup_event_locs must be sorted"

    onset_happen_locs = onset_probas >= cutoff
    if not np.any(onset_happen_locs):
        return np.array([], dtype=wakeup_event_locs.dtype)

    onset_first = np.argwhere(onset_happen_locs).flatten()[0]
    cut_idx = np.searchsorted(wakeup_event_locs, onset_first, side="left")
    if cut_idx == 0:
        return wakeup_event_locs
    return wakeup_event_locs[cut_idx:]

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

def compute_distances(probas_array, cutoff, locs_array):
    # The "good" locations are probas_array > cutoff.
    # For each element in locs_array, find the closest "good" location, and compute the distance
    good_locations = probas_array > cutoff
    good_locs_edge_start, good_locs_edge_end = edges_detect(good_locations)

    locs_is_good = good_locations[locs_array]
    locs_start_diff = np.abs(locs_array - good_locs_edge_start[find_closest(good_locs_edge_start, locs_array)])
    locs_end_diff = np.abs(locs_array - good_locs_edge_end[find_closest(good_locs_edge_end, locs_array)] + 1)
    locs_diff = np.minimum(locs_start_diff, locs_end_diff)

    return np.where(locs_is_good, 0, locs_diff)


def index_probas_distance_based(locs, pred_probas, proba_threshold: float, dropoff_factor: float,
                                prev_run_info: tuple=None):
    # index pred_probas with locs, but also make it distance aware
    if prev_run_info is None:
        preds_above_threshold = pred_probas > proba_threshold
        preds_above_edge_start, preds_above_edge_end = edges_detect(preds_above_threshold)
    else:
        preds_above_threshold, preds_above_edge_start, preds_above_edge_end = prev_run_info

    locs_is_above_threshold = preds_above_threshold[locs]
    locs_start_diff = np.abs(locs - preds_above_edge_start[find_closest(preds_above_edge_start, locs)])
    locs_end_diff = np.abs(locs - preds_above_edge_end[find_closest(preds_above_edge_end, locs)] + 1)
    locs_diff = np.minimum(locs_start_diff, locs_end_diff)

    original_index_probas = pred_probas[locs]
    locs_dropoff_probas = original_index_probas * np.exp(-locs_diff / dropoff_factor)

    return np.where(locs_is_above_threshold, original_index_probas, locs_dropoff_probas)

def compute_first_zero(series_secs):
    return np.argwhere(series_secs == 0).flatten()[0]

def align_predictions(preds_locs, preds_local_kernel, first_zero, return_sorted=True):
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

    if return_sorted:
        return np.unique(preds_locs)
    else:
        return preds_locs

def augment_left_right(preds_locs, preds_probas, preds_local_kernel):
    preds_left = preds_locs - 12
    preds_right = preds_locs + 12

    left_kernel_val = index_out_of_bounds(preds_local_kernel, preds_left)
    right_kernel_val = index_out_of_bounds(preds_local_kernel, preds_right)

    both_in_bounds = (left_kernel_val != -1) & (right_kernel_val != -1)

    augmented_locs1 = np.where(left_kernel_val > right_kernel_val, preds_left, preds_right)
    augmented_locs2 = np.where(left_kernel_val > right_kernel_val, preds_right, preds_left)[both_in_bounds]

    augmented_locs_probas1 = preds_probas.copy()
    augmented_locs_probas2 = preds_probas[both_in_bounds]

    return augmented_locs1, augmented_locs2, augmented_locs_probas1, augmented_locs_probas2

def augment_and_cat(preds_locs, preds_local_kernel, preds_probas, divisor: float, shifts: list[float]):
    augmented_locs1, augmented_locs2, augmented_locs_probas1, augmented_locs_probas2 =\
        augment_left_right(preds_locs, preds_probas, preds_local_kernel)
    preds_locs_probas = preds_probas

    all_preds_locs = np.concatenate([preds_locs, augmented_locs1, augmented_locs2])
    all_preds_probas = np.concatenate([preds_locs_probas / divisor + shifts[0],
                                       augmented_locs_probas1 / divisor + shifts[1],
                                       augmented_locs_probas2 / divisor + shifts[2]])
    return all_preds_locs, all_preds_probas

def get_augmented_predictions(preds_locs, preds_local_kernel, preds_probas, cutoff_thresh: float):
    assert len(preds_probas) == len(preds_locs), "preds_probas and preds_locs must have the same length"

    if len(preds_locs) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    if cutoff_thresh == 0.0:
        all_preds_locs, all_preds_probas = augment_and_cat(preds_locs, preds_local_kernel, preds_probas,
                                                           3.0, [2.0 / 3, 1.0 / 3, 0.0])
    else:
        preds_locs_below = preds_locs[preds_probas < cutoff_thresh]
        preds_locs_above = preds_locs[preds_probas >= cutoff_thresh]

        if len(preds_locs_above) == 0: # all below
            all_preds_locs, all_preds_probas = augment_and_cat(preds_locs, preds_local_kernel, preds_probas,
                                                               6.0, [2.0 / 6, 1.0 / 6, 0.0])
        elif len(preds_locs_below) == 0: # all above
            all_preds_locs, all_preds_probas = augment_and_cat(preds_locs, preds_local_kernel, preds_probas,
                                                               6.0, [5.0 / 6, 4.0 / 6, 3.0 / 6])
        else:
            all_above_preds_locs, all_above_preds_probas = augment_and_cat(preds_locs_above, preds_local_kernel, preds_probas[preds_probas >= cutoff_thresh],
                                                               6.0, [5.0 / 6, 4.0 / 6, 3.0 / 6])
            all_below_preds_locs, all_below_preds_probas = augment_and_cat(preds_locs_below, preds_local_kernel, preds_probas[preds_probas < cutoff_thresh],
                                                               6.0, [2.0 / 6, 1.0 / 6, 0.0])
            all_preds_locs = np.concatenate([all_above_preds_locs, all_below_preds_locs])
            all_preds_probas = np.concatenate([all_above_preds_probas, all_below_preds_probas])
    sort_index = np.argsort(all_preds_locs)
    all_preds_locs, all_preds_probas = all_preds_locs[sort_index], all_preds_probas[sort_index]
    return all_preds_locs, all_preds_probas

def augment_and_cat_density(preds_locs, preds_local_kernel, preds_probas, shifts: list[float]):
    augmented_locs1, augmented_locs2, augmented_locs_probas1, augmented_locs_probas2 =\
        augment_left_right(preds_locs, preds_probas, preds_local_kernel)
    preds_locs_probas = preds_probas

    all_preds_locs = np.concatenate([preds_locs, augmented_locs1, augmented_locs2])
    all_preds_probas = np.concatenate([preds_locs_probas + shifts[0],
                                       augmented_locs_probas1 + shifts[1],
                                       augmented_locs_probas2 + shifts[2]])
    return all_preds_locs, all_preds_probas

def get_augmented_predictions_density(preds_locs, preds_local_kernel, preds_score, cutoff_thresh: float):
    assert len(preds_score) == len(preds_locs), "preds_score and preds_locs must have the same length"

    if len(preds_locs) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    if cutoff_thresh == 0.0:
        all_preds_locs, all_preds_probas = augment_and_cat_density(preds_locs, preds_local_kernel, preds_score,
                                                           [122.0, 61.0, 0.0])
    else:
        preds_locs_below = preds_locs[preds_score < cutoff_thresh]
        preds_locs_above = preds_locs[preds_score >= cutoff_thresh]

        if len(preds_locs_above) == 0: # all below
            all_preds_locs, all_preds_probas = augment_and_cat_density(preds_locs, preds_local_kernel, preds_score,
                                                               [122.0, 61.0, 0.0])
        elif len(preds_locs_below) == 0: # all above
            all_preds_locs, all_preds_probas = augment_and_cat_density(preds_locs, preds_local_kernel, preds_score,
                                                               [305.0, 244.0, 183.0])
        else:
            all_above_preds_locs, all_above_preds_probas = augment_and_cat_density(preds_locs_above, preds_local_kernel, preds_score[preds_score >= cutoff_thresh],
                                                               [305.0, 244.0, 183.0])
            all_below_preds_locs, all_below_preds_probas = augment_and_cat_density(preds_locs_below, preds_local_kernel, preds_score[preds_score < cutoff_thresh],
                                                               [122.0, 61.0, 0.0])
            all_preds_locs = np.concatenate([all_above_preds_locs, all_below_preds_locs])
            all_preds_probas = np.concatenate([all_above_preds_probas, all_below_preds_probas])
    sort_index = np.argsort(all_preds_locs)
    all_preds_locs, all_preds_probas = all_preds_locs[sort_index], all_preds_probas[sort_index]
    return all_preds_locs, all_preds_probas

DAY_STEPS = 17280
MAX_DAY_MULTIPLE = 50

def get60cut(cut):
    x = np.arange(0, 61, cut, dtype=np.float32)
    return x[:np.searchsorted(x, 60.0, side="right")]

def get_time_binned_probas(preds_locs, preds_score, score_cut: float, day_multiple: int):
    time_bins = MAX_DAY_MULTIPLE - (preds_locs.astype(np.int32) // (DAY_STEPS * day_multiple))
    time_bins = np.maximum(time_bins, 0)

    time_proba_bins = time_bins
    cutoffs = get60cut(score_cut)
    for i in range(len(cutoffs) - 1):
        cutoff_low = cutoffs[i]
        cutoff_high = cutoffs[i + 1]
        mask = np.logical_and(cutoff_low < preds_score, preds_score <= cutoff_high)

        if np.any(mask):
            time_proba_bins[mask] += (i * (MAX_DAY_MULTIPLE + 1))

    return preds_score + time_proba_bins * 61.0

def chris_postprocessing(preds_locs: np.ndarray, preds_score: np.ndarray, series_length: int):
    idxsort = np.argsort(preds_score)[::-1]
    num_expected_events = int(np.ceil(series_length / DAY_STEPS))

    processed = 0
    exponent = 0
    while processed < len(preds_locs):
        processed_end = min(processed + num_expected_events, len(preds_locs))
        preds_score[idxsort[processed:processed_end]] = preds_score[idxsort[processed:processed_end]] * (2.0 ** exponent)
        processed = processed_end
        exponent -= 1

    return preds_score