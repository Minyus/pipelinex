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


def get_timestamp(offset_hours=0, fmt="%Y-%m-%dT%H:%M:%S"):
    return (datetime.now() + timedelta(hours=offset_hours)).strftime(fmt)


def get_timestamp_int(offset_hours=0):
    return int(get_timestamp(offset_hours=offset_hours, fmt="%Y%m%d%H%M%S"))


class MLflowBasicLoggerHook:
    def __init__(
        self,
        enable_mlflow=True,
        uri=None,
        experiment_name=None,
        artifact_location=None,
        offset_hours=None,
        mlflow_logging_config_key="MLFLOW_LOGGING_CONFIG",
    ):
        self.enable_mlflow = find_spec("mlflow") and enable_mlflow
        self.uri = uri
        self.experiment_name = experiment_name
        self.artifact_location = artifact_location
        self.offset_hours = offset_hours or 0
        self.mlflow_logging_config_key = mlflow_logging_config_key

    @hook_impl
    def after_catalog_created(self, catalog: DataCatalog):
        parameters = catalog._data_sets["parameters"].load()
        mlflow_logging_params = parameters.get(self.mlflow_logging_config_key, {})

        self.enable_mlflow = mlflow_logging_params.get(
            "enable_mlflow", self.enable_mlflow
        )
        self.uri = mlflow_logging_params.get("uri") or self.uri
        self.experiment_name = (
            mlflow_logging_params.get("experiment_name") or self.experiment_name
        )
        self.artifact_location = (
            mlflow_logging_params.get("artifact_location") or self.artifact_location
        )
        self.offset_hours = (
            mlflow_logging_params.get("offset_hours") or self.offset_hours
        )

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
        if self.enable_mlflow:
            from mlflow import log_metric, log_param

            log_metric(
                "__time_begin", get_timestamp_int(offset_hours=self.offset_hours)
            )
            log_param("__time_begin", get_timestamp(offset_hours=self.offset_hours))
            self.time_begin = time.time()

    @hook_impl
    def after_pipeline_run(
        self, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
    ):
        if self.enable_mlflow:

            from mlflow import end_run, log_metric, log_param

            log_metric("__time_end", get_timestamp_int(offset_hours=self.offset_hours))
            log_param("__time_end", get_timestamp(offset_hours=self.offset_hours))
            log_metric("__time", (time.time() - self.time_begin))

            end_run()
