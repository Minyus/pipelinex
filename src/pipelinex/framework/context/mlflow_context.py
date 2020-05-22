from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Any, Dict  # NOQA
import logging

from kedro.io import DataCatalog
from kedro.pipeline import Pipeline

from .context import KedroContext
from ...extras.hooks.mlflow_basic_logger import MLflowBasicLoggerHook


log = logging.getLogger(__name__)


class MLflowContext(KedroContext):
    mlflow_logging_config_key = "MLFLOW_LOGGING_CONFIG"
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

    def run(self, *args, **kwargs):
        hooks = [MLflowBasicLoggerHook()]
        hooks_to_run = [
            hook for hook in hooks if not self._hook_manager.is_registered(hook)
        ]
        for hook in hooks_to_run:
            hook.before_pipeline_run(
                run_params=None, pipeline=None, catalog=self.catalog
            )

        try:
            from mlflow import log_artifact

            conf_path = Path(self.config_loader.conf_paths[0]) / "parameters.yml"
            log_artifact(conf_path)
        except Exception:
            pass

        nodes = super().run(*args, **kwargs)
        for hook in hooks_to_run:
            hook.after_pipeline_run(
                run_params=None, pipeline=None, catalog=self.catalog
            )
        return nodes
