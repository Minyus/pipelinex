import time
from importlib.util import find_spec
from logging import getLogger
from typing import Any, Callable  # NOQA

from kedro.io import AbstractTransformer

log = getLogger(__name__)


class MLflowIOTimeLoggerTransformer(AbstractTransformer):
    """Log duration time to load and save each dataset."""

    def __init__(
        self,
        enable_mlflow: bool = True,
        metric_name_prefix: str = "_time_to_",
    ):
        """
        Args:
            enable_mlflow: Enable logging to MLflow.
            metric_name_prefix: Prefix for the metric names. The metric names are
                `metric_name_prefix` concatenated with 'load <data_set_name>' or
                'save <data_set_name>'
        """
        self.enable_mlflow = find_spec("mlflow") and enable_mlflow
        self.metric_name_prefix = metric_name_prefix

    def _log_time(self, time_begin, data_set_name, action):
        time_ = time.time() - time_begin
        data_set_name = data_set_name.replace(":", "..")
        time_dict = {
            "{}{} {}".format(self.metric_name_prefix, action, data_set_name)[
                :250
            ]: time_
        }

        log.info("Time duration: {}".format(time_dict))
        if self.enable_mlflow:
            from mlflow import log_metrics

            log_metrics(time_dict)

    def load(self, data_set_name: str, load: Callable[[], Any]) -> Any:
        time_begin = time.time()
        data = load()
        self._log_time(time_begin, data_set_name, "load")
        return data

    def save(self, data_set_name: str, save: Callable[[Any], None], data: Any) -> None:
        time_begin = time.time()
        save(data)
        self._log_time(time_begin, data_set_name, "save")
