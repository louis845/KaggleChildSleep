import numpy as np
import torch
import io

import abc

class Metrics(abc.ABC):
    @abc.abstractmethod
    def add(self, *args):
        pass

    @abc.abstractmethod
    def get(self):
        pass

    @abc.abstractmethod
    def write_to_dict(self, x: dict):
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class NumericalMetric(Metrics):
    def __init__(self, name: str):
        self.name = name
        self.values = []
        self.reset()

    def add(self, value, batch_size: int):
        self.sum += value
        self.count += batch_size
        self.values.append(value)

    def get(self):
        if self.count == 0:
            return -1.0
        return self.sum / self.count

    def write_to_dict(self, x: dict):
        x[self.name] = self.get()
        if self.count == 0:
            x[self.name + "_median"] = -1.0
        else:
            x[self.name + "_median"] = np.median(self.values)

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.values.clear()

class BinaryMetrics(Metrics):
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def add(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor = None):
        """
        y_pred and y_true must be tensors of the same shape. They contain binary values (0 or 1).
        """
        assert y_pred.shape == y_true.shape, "y_pred and y_true must have the same shape"
        assert y_pred.dtype == torch.long and y_true.dtype == torch.long, "y_pred and y_true must be long tensors"
        if mask is not None:
            assert mask.shape == y_pred.shape, "mask must have the same shape as y_pred and y_true"
            assert mask.dtype == torch.long, "mask must be a long tensor"

        if mask is None:
            self.tp += torch.sum(y_pred * y_true).cpu().item()
            self.tn += torch.sum((1 - y_pred) * (1 - y_true)).cpu().item()
            self.fp += torch.sum(y_pred * (1 - y_true)).cpu().item()
            self.fn += torch.sum((1 - y_pred) * y_true).cpu().item()
        else:
            self.tp += torch.sum(y_pred * y_true * mask).cpu().item()
            self.tn += torch.sum((1 - y_pred) * (1 - y_true) * mask).cpu().item()
            self.fp += torch.sum(y_pred * (1 - y_true) * mask).cpu().item()
            self.fn += torch.sum((1 - y_pred) * y_true * mask).cpu().item()

    def add_direct(self, tp, tn, fp, fn):
        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn

    def get(self):
        """Compute the accuracy, precision, recall."""
        if self.tp + self.tn + self.fp + self.fn == 0:
            accuracy = -1.0
        else:
            accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        if self.tp + self.fp == 0:
            precision = 0.0
        else:
            precision = self.tp / (self.tp + self.fp)
        if self.tp + self.fn == 0:
            recall = 0.0
        else:
            recall = self.tp / (self.tp + self.fn)
        return accuracy, precision, recall

    def write_to_dict(self, x: dict):
        accuracy, precision, recall = self.get()
        x[self.name + "_accuracy"] = accuracy
        x[self.name + "_precision"] = precision
        x[self.name + "_recall"] = recall

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def get_iou(self):
        if self.tp + self.fp + self.fn == 0:
            iou = 1.0
        else:
            iou = self.tp / (self.tp + self.fp + self.fn)
        return iou

    def report_print(self, report_iou=False):
        accuracy, precision, recall = self.get()
        print("--------------------- {} ---------------------".format(self.name))
        if report_iou:
            iou = self.get_iou()
            print("Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, IoU: {:.4f}".format(accuracy, precision, recall, iou))
        else:
            print("Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}".format(accuracy, precision, recall))

    def report_print_to_file(self, file: io.TextIOWrapper, report_iou=False):
        accuracy, precision, recall = self.get()
        file.write("--------------------- {} ---------------------\n".format(self.name))
        if report_iou:
            iou = self.get_iou()
            file.write("Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, IoU: {:.4f}\n".format(accuracy, precision, recall, iou))
        else:
            file.write("Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}\n".format(accuracy, precision, recall))

