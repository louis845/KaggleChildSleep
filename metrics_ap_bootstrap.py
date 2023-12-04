import numpy as np

import metrics_ap

def compute_precision_recall_curve(num_positive, matches, probas): # should be sorted
    # compute precision and recall curve (using Kaggle code)
    distinct_value_indices = np.where(np.diff(probas))[0]
    threshold_idxs = np.r_[distinct_value_indices, matches.size - 1]
    probas = probas[threshold_idxs]

    # Matches become TPs and non-matches FPs as confidence threshold decreases
    tps = np.cumsum(matches)[threshold_idxs]
    fps = np.cumsum(~matches)[threshold_idxs]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / num_positive  # total number of ground truths might be different than total number of matches

    # Stop when full recall attained and reverse the outputs so recall is non-increasing.
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    # Final precision is 1 and final recall is 0 and final proba is 1
    precision, recall, probas = np.r_[precision[sl], 1], np.r_[recall[sl], 0], np.r_[probas[sl], 1]

    # compute average precision
    average_precision = -np.sum(np.diff(recall) * np.array(precision)[:-1])

    return precision, recall, average_precision, probas

def get_identity_results(matches, probas, cutoffs):
    assert np.all(cutoffs[1:] > cutoffs[:-1]), "cutoffs must be strictly increasing"

    matches2 = matches[::-1].copy() # increasing order by probas
    probas2 = probas[::-1].copy()
    for k in range(len(cutoffs) - 1):
        cutlow, cuthigh = cutoffs[k], cutoffs[k + 1]
        low_idx, high_idx = np.searchsorted(probas2, [cutlow, cuthigh], side="left") # find the indices between [cutlow, cuthigh)
        if low_idx + 1 < high_idx:
            low_proba_val = probas2[low_idx]
            high_proba_val = probas2[high_idx - 1]
            proba_diff = high_proba_val - low_proba_val
            low_proba_val += proba_diff * 0.1
            high_proba_val -= proba_diff * 0.1
            probas2[low_idx:high_idx] = np.linspace(low_proba_val, high_proba_val, high_idx - low_idx)
    return matches2[::-1], probas2[::-1]

def get_lower_bound(matches, probas, cutoffs):
    assert np.all(cutoffs[1:] > cutoffs[:-1]), "cutoffs must be strictly increasing"

    matches2 = matches[::-1].copy()  # increasing order by probas
    probas2 = probas[::-1].copy()
    for k in range(len(cutoffs) - 1):
        cutlow, cuthigh = cutoffs[k], cutoffs[k + 1]
        low_idx, high_idx = np.searchsorted(probas2, [cutlow, cuthigh],
                                            side="left")  # find the indices between [cutlow, cuthigh)
        if low_idx + 1 < high_idx:
            low_proba_val = probas2[low_idx]
            high_proba_val = probas2[high_idx - 1]
            proba_diff = high_proba_val - low_proba_val
            low_proba_val += proba_diff * 0.1
            high_proba_val -= proba_diff * 0.1
            num_matches = np.sum(matches2[low_idx:high_idx])

            probas2[low_idx:high_idx] = np.linspace(low_proba_val, high_proba_val, high_idx - low_idx)
            matches2[low_idx:high_idx] = False
            matches2[low_idx:low_idx + num_matches] = True
    return matches2[::-1], probas2[::-1]

def get_upper_bound(matches, probas, cutoffs):
    assert np.all(cutoffs[1:] > cutoffs[:-1]), "cutoffs must be strictly increasing"

    matches2 = matches[::-1].copy()  # increasing order by probas
    probas2 = probas[::-1].copy()
    for k in range(len(cutoffs) - 1):
        cutlow, cuthigh = cutoffs[k], cutoffs[k + 1]
        low_idx, high_idx = np.searchsorted(probas2, [cutlow, cuthigh],
                                            side="left")  # find the indices between [cutlow, cuthigh)
        if low_idx + 1 < high_idx:
            low_proba_val = probas2[low_idx]
            high_proba_val = probas2[high_idx - 1]
            proba_diff = high_proba_val - low_proba_val
            low_proba_val += proba_diff * 0.1
            high_proba_val -= proba_diff * 0.1
            num_matches = np.sum(matches2[low_idx:high_idx])

            probas2[low_idx:high_idx] = np.linspace(low_proba_val, high_proba_val, high_idx - low_idx)
            matches2[low_idx:high_idx] = False
            matches2[high_idx - num_matches:high_idx] = True
    return matches2[::-1], probas2[::-1]

def get_single_bootstrap(series_ids: np.ndarray, series_locs: np.ndarray, series_probas: np.ndarray, cutoffs: np.ndarray):
    idxsort = np.argsort(series_probas)
    series_ids = series_ids[idxsort]
    series_locs = series_locs[idxsort]
    series_probas = series_probas[idxsort]

    for k in range(len(cutoffs) - 1):
        cutlow, cuthigh = cutoffs[k], cutoffs[k + 1]
        low_idx, high_idx = np.searchsorted(series_probas, [cutlow, cuthigh], side="left")
        if low_idx + 1 < high_idx:
            sub_series_ids = series_ids[low_idx:high_idx]
            sub_series_locs = series_locs[low_idx:high_idx]

            low_proba_val = series_probas[low_idx]
            high_proba_val = series_probas[high_idx - 1]
            proba_diff = high_proba_val - low_proba_val
            low_proba_val += proba_diff * 0.1
            high_proba_val -= proba_diff * 0.1

            rng_idx = np.random.permutation(high_idx - low_idx)
            sub_series_ids = sub_series_ids[rng_idx]
            sub_series_locs = sub_series_locs[rng_idx]

            series_ids[low_idx:high_idx] = sub_series_ids
            series_locs[low_idx:high_idx] = sub_series_locs
            series_probas[low_idx:high_idx] = np.linspace(low_proba_val, high_proba_val, high_idx - low_idx)

    return series_ids, series_locs, series_probas

def get_bootstrapped_stream(series_ids: np.ndarray, series_locs: np.ndarray, series_probas: np.ndarray,
                            num_bootstraps: int, cutoffs: np.ndarray):
    assert np.all(cutoffs[1:] > cutoffs[:-1]), "cutoffs must be strictly increasing"
    assert len(series_ids) == len(series_locs) == len(series_probas), "series_ids, series_locs, series_probas must have the same length"

    for j in range(num_bootstraps):
        yield get_single_bootstrap(series_ids, series_locs, series_probas, cutoffs)

class EventMetricsBootstrap:
    def __init__(self, name:str, tolerance=12 * 30):
        self.name = name
        self.tolerance = tolerance

        self.matches = []
        self.probas = []
        self.num_positive = 0

    def add(self, pred_locs: np.ndarray, pred_probas: np.ndarray, gt_locs):
        matches = metrics_ap.match_series(pred_locs, pred_probas, gt_locs, tolerance=self.tolerance)
        self.matches.append(matches)
        self.probas.append(pred_probas)
        self.num_positive += len(gt_locs)

    def get_precision_recall(self):
        matches, probas = self.get_sorted_matches_probas()

        return compute_precision_recall_curve(self.num_positive, matches, probas)

    def get_sorted_matches_probas(self):
        matches = np.concatenate(self.matches, axis=0)
        probas = np.concatenate(self.probas, axis=0)

        # sort by probas in descending order
        idxs = np.argsort(probas)[::-1]
        matches = matches[idxs]
        probas = probas[idxs]

        return matches, probas

    def reset(self):
        self.matches.clear()
        self.probas.clear()
        self.num_positive = 0