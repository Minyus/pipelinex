import time
from datetime import datetime, timedelta
from importlib.util import find_spec
from logging import getLogger
from typing import Any, Dict, List, Optional, Union  # NOQA

from kedro.io import DataCatalog
from kedro.pipeline import Pipeline

from .mlflow_utils import (
    hook_impl,
    mlflow_end_run,
    mlflow_log_metrics,
    mlflow_log_params,
    mlflow_start_run,
)

log = getLogger(__name__)


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
    """Configures and logs duration time for the pipeline to MLflow"""

    def __init__(
        self,
        uri: str = None,
        experiment_name: str = None,
        artifact_location: str = None,
        run_name: str = None,
        run_id: str = None,
        nested: bool = False,
        tags: Optional[Dict[str, Any]] = None,
        offset_hours: float = 0,
        enable_logging_time_begin: bool = True,
        enable_logging_time_end: bool = True,
        enable_logging_time: bool = True,
        logging_kedro_run_params: Union[List[str], str] = [],
        enable_mlflow: bool = True,
    ):
        """
        Args:
            uri: The MLflow tracking server URI.
                `uri` arg fed to:
                https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri
            experiment_name: The experiment name.
                `name` arg fed to:
                https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.create_experiment
            artifact_location: `artifact_location` arg fed to:
                https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.create_experiment
            run_name: Shown as 'Run Name' in MLflow UI.
            run_id: An existing MLflow experiment run UUID instead of letting MLflow create
                a new run under the experiment_name. `run_id` arg fed to:
                https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run
            nested: `nested` arg fed to:
                https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run
            tags: `tags` arg fed to:
                https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run
            offset_hours: The offset hour (e.g. 0 for UTC+00:00) to log in MLflow. 0 in default.
            enable_logging_time_begin: Enable logging the time the Kedro pipeline began. True in default.
            enable_logging_time_end: Enable logging the time the Kedro pipeline ended. True in default.
            enable_logging_time: Enable logging the time duration the Kedro pipeline ran. True in default.
            logging_kedro_run_params: List of Kedro Run Params to log to MLflow or "__ALL__" to log all.
                [] (Empty) in default.
            enable_mlflow: Enable configuring and logging to MLflow.
        """
        self.enable_mlflow = find_spec("mlflow") and enable_mlflow
        self.uri = uri
        self.experiment_name = experiment_name
        self.artifact_location = artifact_location
        self.run_name = run_name
        self.offset_hours = offset_hours
        self.enable_logging_time_begin = enable_logging_time_begin
        self.enable_logging_time_end = enable_logging_time_end
        self.enable_logging_time = enable_logging_time
        self.logging_kedro_run_params = logging_kedro_run_params

    @hook_impl
    def after_catalog_created(self):
        mlflow_start_run(
            uri=self.uri,
            experiment_name=self.experiment_name,
            artifact_location=self.artifact_location,
            run_name=self.run_name,
            enable_mlflow=self.enable_mlflow,
        )

    @hook_impl
    def before_pipeline_run(
        self, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
    ):
        if self.logging_kedro_run_params:
            run_params_renamed = {
                ("___" + k): v
                for (k, v) in run_params.items()
                if (
                    isinstance(self.logging_kedro_run_params, str)
                    and self.logging_kedro_run_params == "__ALL__"
                )
                or k in self.logging_kedro_run_params
            }
            mlflow_log_params(run_params_renamed, enable_mlflow=self.enable_mlflow)

        if self.enable_logging_time_begin:
            timestamp, timestamp_int = get_timestamps(offset_hours=self.offset_hours)
            time_dict = {"__time_begin": timestamp}
            time_int_dict = {"__time_begin": timestamp_int}

            mlflow_log_params(time_dict, enable_mlflow=self.enable_mlflow)
            mlflow_log_metrics(time_int_dict, enable_mlflow=self.enable_mlflow)

        if self.enable_logging_time:
            self._time_begin = time.time()

    @hook_impl
    def after_pipeline_run(
        self, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
    ):
        if self.enable_logging_time_end:
            timestamp, timestamp_int = get_timestamps(offset_hours=self.offset_hours)
            time_dict = {"__time_end": timestamp}
            time_int_dict = {"__time_end": timestamp_int}

            mlflow_log_params(
                time_dict,
                enable_mlflow=self.enable_mlflow,
            )
            mlflow_log_metrics(
                time_int_dict,
                enable_mlflow=self.enable_mlflow,
            )

        if self.enable_logging_time:
            self._time_end = time.time()
            self._time = self._time_end - self._time_begin

            time_dict = {"__time": self._time}
            mlflow_log_metrics(time_dict, enable_mlflow=self.enable_mlflow)

        mlflow_end_run(self.enable_mlflow)
