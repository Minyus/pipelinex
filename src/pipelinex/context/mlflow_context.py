from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Any, Iterable  # NOQA
import logging

from kedro.context import KedroContext

from .flexible_context import FlexibleContext

log = logging.getLogger(__name__)


class MLflowContext(KedroContext):
    uri = ""  # type: str
    experiment_name = ""  # type: str
    artifact_location = ""  # type: str
    offset_hours = 0  # type: int
    logging_artifacts = []  # type: Iterable[str]

    def _format_kedro_dataset(self, ds_name, ds_dict):
        ds_name, ds_dict = self._set_filepath(ds_name, ds_dict)
        ds_name, ds_dict = self._get_mlflow_logging_flag(ds_name, ds_dict)
        ds_name, ds_dict = self._enable_caching(ds_name, ds_dict)
        return ds_name, ds_dict

    def _get_mlflow_logging_flag(self, ds_name, ds_dict):
        if "mlflow_logging" in ds_dict:
            mlflow_logging = ds_dict.pop("mlflow_logging")
            if mlflow_logging and ds_name not in self.logging_artifacts:
                self.logging_artifacts.append(ds_name)
        return ds_name, ds_dict

    def run(
        self,
        *args,  # type: Any
        **kwargs  # type: Any
    ):
        parameters = self.catalog._data_sets["parameters"].load()
        mlflow_logging_params = parameters.get("MLFLOW_LOGGING_CONFIG")
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
                        self.experiment_name, artifact_location=self.artifact_location
                    )
                    start_run(experiment_id=experiment_id)
                except MlflowException:
                    set_experiment(self.experiment_name)

            conf_path = Path(self.config_loader.conf_paths[0]) / "parameters.yml"
            log_artifact(conf_path)

            log_metric("__t0", get_timestamp_int(offset_hours=self.offset_hours))
            log_param("time_begin", get_timestamp(offset_hours=self.offset_hours))
            time_begin = time.time()

        nodes = super().run(*args, **kwargs)

        if mlflow_logging_params:
            log_metric("__t1", get_timestamp_int(offset_hours=self.offset_hours))
            log_param("time_end", get_timestamp(offset_hours=self.offset_hours))
            log_metric("__time", (time.time() - time_begin))

            for d in self.logging_artifacts:
                ds = getattr(self.catalog.datasets, d, None)
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
                        log.warning("_filepath of '{}' could not be found.".format(d))

            end_run()

        return nodes


def get_timestamp(offset_hours=0, fmt="%Y-%m-%dT%H:%M:%S"):
    return (datetime.now() + timedelta(hours=offset_hours)).strftime(fmt)


def get_timestamp_int(offset_hours=0):
    return int(get_timestamp(offset_hours=offset_hours, fmt="%Y%m%d%H%M"))


class MLflowFlexibleContext(MLflowContext, FlexibleContext):
    pass
