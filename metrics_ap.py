import numpy as np

def match_series(pred_locs: np.ndarray, pred_probas: np.ndarray, gt_locs, tolerance=12 * 30):
    """
    Probably faster algorithm for matching, since the gt are disjoint (within tolerance)

    pred_locs: predicted locations of events, assume sorted in ascending order
    pred_probas: predicted probabilities of events
    gt_locs: ground truth locations of events (either list[int] or np.ndarray or int32 type)
    """
    assert pred_locs.shape == pred_probas.shape, "pred_locs {} and pred_probas {} must have the same shape".format(pred_locs.shape, pred_probas.shape)
    assert len(pred_locs.shape) == 1, "pred_locs {} and pred_probas {} must be 1D".format(pred_locs.shape, pred_probas.shape)
    matches = np.zeros_like(pred_locs, dtype=bool)

    if isinstance(gt_locs, list):
        gt_locs = np.array(gt_locs, dtype=np.int32)
    else:
        assert isinstance(gt_locs, np.ndarray), "gt_locs must be list or np.ndarray"
        assert gt_locs.dtype == np.int32, "gt_locs must be int32 type"

    # lie within (event_loc - tolerance, event_loc + tolerance), where event_loc in gt_locs
    idx_lows = np.searchsorted(pred_locs, gt_locs - tolerance + 1)
    idx_highs = np.searchsorted(pred_locs, gt_locs + tolerance)
    for k in range(len(gt_locs)):
        idx_low, idx_high = idx_lows[k], idx_highs[k]
        if idx_low == idx_high:
            continue
        # find argmax within range
        max_location = idx_low + np.argmax(pred_probas[idx_low:idx_high])
        matches[max_location] = True
    return matches

class EventMetrics:
    def __init__(self, name:str, tolerance=12 * 30):
        self.name = name
        self.tolerance = tolerance

        self.matches = []
        self.probas = []
        self.num_positive = 0

    def add(self, pred_locs: np.ndarray, pred_probas: np.ndarray, gt_locs):
        matches = match_series(pred_locs, pred_probas, gt_locs, tolerance=self.tolerance)
        self.matches.append(matches)
        self.probas.append(pred_probas)
        self.num_positive += len(gt_locs)

    def get(self):
        matches = np.concatenate(self.matches, axis=0)
        probas = np.concatenate(self.probas, axis=0)

        # sort by probas in descending order
        idxs = np.argsort(probas)[::-1]
        matches = matches[idxs]
        probas = probas[idxs]

        # compute precision and recall curve
        precision = np.cumsum(matches) / np.arange(1, len(matches) + 1)
        recall = np.cumsum(matches) / self.num_positive
        precision = np.concatenate([[1], precision, [0]])
        recall = np.concatenate([[0], recall, [1]])

        # compute average precision
        average_precision = np.trapz(precision, recall)

        return precision, recall, average_precision

    def write_to_dict(self, x: dict):
        x[self.name] = self.get()

    def reset(self):
        self.matches.clear()
        self.probas.clear()
        self.num_true = 0