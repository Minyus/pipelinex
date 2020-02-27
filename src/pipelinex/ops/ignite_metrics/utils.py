import torch
from ignite.utils import to_onehot


class ClassificationOutputTransform:
    def __init__(self, num_classes=None):
        self._num_classes = num_classes

    def __call__(self, output):
        if isinstance(output, tuple):
            y_pred, y = output
        elif isinstance(output, dict):
            y_pred = output["y_pred"]
            y = output["y"]
        else:
            raise ValueError

        if self._num_classes:
            y_pred = y_pred.clamp(min=0, max=self._num_classes - 1).long()
            y = y.clamp(min=0, max=self._num_classes - 1).long()
            y_pred = to_onehot(y_pred, self._num_classes)
        else:
            y_pred = y_pred.long()
            y = y.long()
        return y_pred, y
