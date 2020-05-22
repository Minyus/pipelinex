from importlib.util import find_spec
from logging import getLogger
from datetime import datetime, timedelta
from pathlib import Path
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


if find_spec("mlflow"):
    from mlflow import log_metric, log_param

    class MLflowBasicLoggerHook:
        mlflow_logging_config_key = "MLFLOW_LOGGING_CONFIG"

        @hook_impl
        def before_pipeline_run(
            self, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
        ):
            parameters = catalog._data_sets["parameters"].load()
            mlflow_logging_params = parameters.get(self.mlflow_logging_config_key)

            if mlflow_logging_params:

                self.uri = mlflow_logging_params.get("uri")
                self.experiment_name = mlflow_logging_params.get("experiment_name")
                self.artifact_location = mlflow_logging_params.get("artifact_location")
                self.offset_hours = mlflow_logging_params.get("offset_hours") or 0
                self.logging_artifacts = (
                    mlflow_logging_params.get("logging_artifacts") or []
                )

                from mlflow import (
                    create_experiment,
                    set_experiment,
                    start_run,
                    end_run,
                    set_tracking_uri,
                    log_artifact,
                    log_metric,
                    log_param,
                )
                from mlflow.exceptions import MlflowException

                if self.uri:
                    set_tracking_uri(self.uri)

                if self.experiment_name:
                    try:
                        experiment_id = create_experiment(
                            self.experiment_name,
                            artifact_location=self.artifact_location,
                        )
                        start_run(experiment_id=experiment_id)
                    except MlflowException:
                        set_experiment(self.experiment_name)

                log_metric(
                    "__time_begin", get_timestamp_int(offset_hours=self.offset_hours)
                )
                log_param("__time_begin", get_timestamp(offset_hours=self.offset_hours))
                self.time_begin = time.time()

            self.mlflow_logging_params = mlflow_logging_params

        @hook_impl
        def after_pipeline_run(
            self, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
        ):
            if self.mlflow_logging_params:

                from mlflow import (
                    create_experiment,
                    set_experiment,
                    start_run,
                    end_run,
                    set_tracking_uri,
                    log_artifact,
                    log_metric,
                    log_param,
                )
                from mlflow.exceptions import MlflowException

                log_metric(
                    "__time_end", get_timestamp_int(offset_hours=self.offset_hours)
                )
                log_param("__time_end", get_timestamp(offset_hours=self.offset_hours))
                log_metric("__time", (time.time() - self.time_begin))

                for d in self.logging_artifacts:
                    ds = getattr(catalog.datasets, d, None)
                    if ds:
                        fp = getattr(ds, "_filepath", None)
                        if not fp:
                            low_ds = getattr(ds, "_dataset", None)
                            if low_ds:
                                fp = getattr(low_ds, "_filepath", None)
                        if fp:
                            log_artifact(fp)
                            log.info("'{}' was logged by MLflow.".format(fp))
                        else:
                            log.warning(
                                "_filepath of '{}' could not be found.".format(d)
                            )

                end_run()


else:

    class MLflowBasicLoggerHook:
        def before_pipeline_run(
            self, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
        ):
            pass

        def after_pipeline_run(
            self, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
        ):
            pass
