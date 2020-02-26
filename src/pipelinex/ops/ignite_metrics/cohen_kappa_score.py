from typing import Union, Sequence

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from sklearn.metrics import cohen_kappa_score

__all__ = ["CohenKappaScore"]


class CohenKappaScore(Metric):
    """
    Calculates the cohen kappa score.
    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super(CohenKappaScore, self).__init__(device="cpu")

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_cohen_kappa = 0.0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output
        assert y_pred.shape == y.shape
        y_pred_arr = y_pred.numpy().flatten().astype(int)
        y_arr = y.numpy().flatten().astype(int)
        cohen_kappa = cohen_kappa_score(y_pred_arr, y_arr, *self.args, **self.kwargs)
        self._sum_of_cohen_kappa += cohen_kappa
        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_of_squared_errors", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError(
                "MeanSquaredError must have at least one example before it can be computed."
            )
        return self._sum_of_cohen_kappa / self._num_examples
