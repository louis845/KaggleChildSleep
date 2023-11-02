import numpy as np
import torch

def compute_iou_metrics(gt: torch.Tensor, pred: torch.Tensor, iou_threshold=0.5):
    assert isinstance(gt, torch.Tensor), "gt must be a torch.Tensor"
    assert isinstance(pred, torch.Tensor), "pred must be a torch.Tensor"
    assert gt.shape == pred.shape, "gt and pred must have the same shape"
    assert gt.dtype == torch.long, "gt must be of type torch.long"
    assert pred.dtype == torch.long, "pred must be of type torch.long"

    # per sample information
    intersection_per_batch = torch.sum(gt * pred, dim=(1, 2))
    union_per_batch = torch.sum(gt + pred, dim=(1, 2)) - intersection_per_batch
    positive_preds_batch = torch.sum(pred, dim=(1, 2)) > 0
    negative_preds_batch = torch.sum(pred, dim=(1, 2)) == 0
    positive_gt_batch = torch.sum(gt, dim=(1, 2)) > 0
    negative_gt_batch = torch.sum(gt, dim=(1, 2)) == 0

    # compute metrics here
    true_positives = torch.sum(torch.logical_and((intersection_per_batch.to(torch.float32) / union_per_batch.to(torch.float32)) > iou_threshold,
                                        positive_gt_batch)).item() # a true positive only if IOU is high enough and there is a ground truth

    false_positives = torch.sum(torch.logical_and((intersection_per_batch.to(torch.float32) / union_per_batch.to(torch.float32)) <= iou_threshold,
                                            positive_preds_batch)).item() # out of the positive predictions, if the IOU is too low, it is a false positive.
    # if there is no ground truth, it is a false positive. note that no ground truth implies IOU low.

    true_negatives = torch.sum(torch.logical_and(negative_preds_batch, negative_gt_batch)).item()
    # if there is no prediction and no ground truth, it is a true negative.

    false_negatives = torch.sum(torch.logical_and((intersection_per_batch.to(torch.float32) / union_per_batch.to(torch.float32)) <= iou_threshold,
                                                  positive_gt_batch)).item()
    # if there is a ground truth, but the IOU is too low, it is a false negative.

    return true_positives, false_positives, true_negatives, false_negatives

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

def find_segment(preds, loc, start_indices=None, end_indices=None):
    assert preds[loc] == True, "Location must be in a True segment"

    if start_indices is None:
        start_indices, end_indices = edges_detect(preds)

    i = start_indices[np.searchsorted(start_indices, loc, side="right") - 1]
    j = end_indices[np.searchsorted(end_indices, loc, side="right")]

    return i, j

def find_start(preds, loc, start_indices=None, end_indices=None) -> int:
    assert preds[loc] == True, "Location must be in a True segment"

    if start_indices is None:
        start_indices, end_indices = edges_detect(preds)

    i = start_indices[np.searchsorted(start_indices, loc, side="right") - 1]

    return int(i)

def find_end(preds, loc, start_indices=None, end_indices=None) -> int:
    assert preds[loc] == True, "Location must be in a True segment"

    if start_indices is None:
        start_indices, end_indices = edges_detect(preds)

    j = end_indices[np.searchsorted(end_indices, loc, side="right")]

    return int(j)

def compute_iou_metrics_events(preds: np.ndarray, events: list[int], proba_threshold=0.5, iou_threshold=0.5,
                               event_tolerance_width=30 * 12 - 1, gap_threshold=0):
    assert len(preds.shape) == 1, "preds must be a 1D array"
    assert isinstance(events, list), "events must be a list"

    preds_thresh = preds > proba_threshold

    nopreds = np.all(~preds_thresh)

    start_indices, end_indices = edges_detect(preds_thresh)

    # remove short gaps
    if (not nopreds) and (gap_threshold > 0):
        start_indices, end_indices = remove_short_gaps(start_indices, end_indices, gap_threshold=gap_threshold)
        preds_thresh[:] = False
        for i, j in zip(start_indices, end_indices):
            preds_thresh[i:j] = True

    # predict now
    tp_preds = np.zeros_like(preds, dtype=bool)
    tp = 0
    events_size_count = 0
    for event in events:
        event_start = max(0, event - event_tolerance_width)
        event_end = min(len(preds), event + event_tolerance_width + 1)
        events_size_count += (event_end - event_start)

        event_seg = preds_thresh[event_start:event_end]
        if nopreds or np.all(~event_seg):
            continue

        # find min and max of True values inside the event
        preds_locs = np.argwhere(event_seg).flatten()
        interval_min, interval_max = event_start + preds_locs[0], event_start + preds_locs[-1]

        # extend the min and max to find maximum length interval
        if interval_min == event_start:
            interval_min = find_start(preds_thresh, interval_min, start_indices, end_indices)
        if interval_max == event_end - 1:
            interval_max = find_end(preds_thresh, interval_max, start_indices, end_indices)
        else:
            interval_max += 1

        # compute intersection and union
        union_max, union_min = max(interval_max, event_end), min(interval_min, event_start)
        intersection = np.sum(preds_thresh[interval_min:interval_max])
        union = union_max - union_min

        if intersection / union > iou_threshold:
            tp_preds[union_min:union_max] = True
            tp += 1

    fp_preds = preds_thresh & (~tp_preds)
    fp = np.sum(fp_preds)
    false_count = len(preds) - events_size_count

    return tp, fp, false_count