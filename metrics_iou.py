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