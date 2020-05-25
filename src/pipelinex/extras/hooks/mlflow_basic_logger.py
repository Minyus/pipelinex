from importlib.util import find_spec
from logging import getLogger
from datetime import datetime, timedelta
import time
from typing import Any, Dict  # NOQA

from kedro.io import DataCatalog
from kedro.pipeline import Pipeline


log = getLogger(__name__)

try:
    from kedro.framework.hooks import hook_impl
except ModuleNotFoundError:

    def hook_impl(func):
        return func


def get_timestamp(dt=None, offset_hours=0, fmt="%Y-%m-%dT%H:%M:%S"):
    dt = dt or datetime.now()
    return (dt + timedelta(hours=offset_hours)).strftime(fmt)


def get_timestamp_int(dt=None, offset_hours=0):
    return int(get_timestamp(dt=dt, offset_hours=offset_hours, fmt="%Y%m%d%H%M%S"))


def get_timestamps(dt=None, offset_hours=0):
    dt = dt or datetime.now()
    timestamp = get_timestamp(dt, offset_hours=offset_hours)
    timestamp_int = get_timestamp_int(dt, offset_hours=offset_hours)
    return timestamp, timestamp_int


class MLflowBasicLoggerHook:
    def __init__(
        self,
        enable_mlflow=True,
        uri=None,
        experiment_name=None,
        artifact_location=None,
        offset_hours=None,
    ):
        self.enable_mlflow = find_spec("mlflow") and enable_mlflow
        self.uri = uri
        self.experiment_name = experiment_name
        self.artifact_location = artifact_location
        self.offset_hours = offset_hours or 0

    @hook_impl
    def after_catalog_created(self, catalog: DataCatalog):

        if self.enable_mlflow:

            from mlflow import (
                create_experiment,
                set_experiment,
                start_run,
                set_tracking_uri,
            )
            from mlflow.exceptions import MlflowException

            if self.uri:
                set_tracking_uri(self.uri)

            if self.experiment_name:
                try:
                    experiment_id = create_experiment(
                        self.experiment_name, artifact_location=self.artifact_location,
                    )
                    start_run(experiment_id=experiment_id)
                except MlflowException:
                    set_experiment(self.experiment_name)

    @hook_impl
    def before_pipeline_run(
        self, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
    ):
        timestamp, timestamp_int = get_timestamps(offset_hours=self.offset_hours)
        log.info("__time_begin: {}".format(timestamp))

        self.time_begin = time.time()

        if self.enable_mlflow:
            from mlflow import log_metric, log_param

            log_metric("__time_begin", timestamp_int)
            log_param("__time_begin", timestamp)

    @hook_impl
    def after_pipeline_run(
        self, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
    ):
        timestamp, timestamp_int = get_timestamps(offset_hours=self.offset_hours)
        log.info("__time_end: {}".format(timestamp))

        self.time_end = time.time()
        self.time = self.time_end - self.time_begin
        log.info("__time: {}".format(self.time))

        if self.enable_mlflow:

            from mlflow import end_run, log_metric, log_param

            log_metric("__time_end", timestamp_int)
            log_param("__time_end", timestamp)
            log_metric("__time", self.time)

            end_run()
