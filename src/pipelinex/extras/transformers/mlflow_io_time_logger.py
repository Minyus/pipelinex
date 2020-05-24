from importlib.util import find_spec
from logging import getLogger
import time
from typing import Any, Callable

from kedro.io import AbstractTransformer

log = getLogger(__name__)


class MLflowIOTimeLoggerTransformer(AbstractTransformer):
    def __init__(self, enable_mlflow=True):
        self.enable_mlflow = find_spec("mlflow") and enable_mlflow

    def _log_time(self, time_begin, data_set_name, action):
        time_ = time.time() - time_begin
        data_set_name = data_set_name.replace(":", "..")
        time_dict = {"_time_to_{} {}".format(action, data_set_name): time_}

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
