from ignite.metrics import Metric, Precision, Recall

from typing import Sequence, Callable, Optional, Union

import torch

from ignite.metrics.metric import reinit__is_reduced

__all__ = ["FbetaScore"]


class FbetaScore(Metric):
    def __init__(
        self,
        beta: int = 1,
        output_transform: Callable = lambda x: x,
        average: str = "macro",
        is_multilabel: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self._beta = beta
        self._average = average
        _average_flag = self._average != "macro"
        self._precision = Precision(
            output_transform=output_transform,
            average=_average_flag,
            is_multilabel=is_multilabel,
            device=device,
        )

        self._recall = Recall(
            output_transform=output_transform,
            average=_average_flag,
            is_multilabel=is_multilabel,
            device=device,
        )
        super(FbetaScore, self).__init__(
            output_transform=output_transform, device=device
        )

    @reinit__is_reduced
    def reset(self) -> None:
        self._precision.reset()
        self._recall.reset()

    def compute(self) -> torch.Tensor:
        precision_val = self._precision.compute()
        recall_val = self._recall.compute()
        fbeta_val = (
            (1.0 + self._beta ** 2)
            * precision_val
            * recall_val
            / (self._beta ** 2 * precision_val + recall_val + 1e-15)
        )
        if self._average == "macro":
            fbeta_val = torch.mean(fbeta_val).item()
        return fbeta_val

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._precision.update(output)
        self._recall.update(output)
