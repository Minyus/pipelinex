from importlib.util import find_spec
from logging import getLogger
from typing import Any, Dict  # NOQA

from kedro.io import DataCatalog
from kedro.pipeline import Pipeline


log = getLogger(__name__)

try:
    from kedro.framework.hooks import hook_impl
except ModuleNotFoundError:

    def hook_impl(func):
        return func


class MLflowArtifactsLoggerHook:
    def __init__(
        self,
        enable_mlflow=True,
        filepaths_before_pipeline_run=None,
        datasets_after_node_run=None,
        filepaths_after_pipeline_run=None,
    ):
        self.enable_mlflow = find_spec("mlflow") and enable_mlflow
        self.filepaths_before_pipeline_run = filepaths_before_pipeline_run or []
        self.datasets_after_node_run = datasets_after_node_run or []
        self.filepaths_after_pipeline_run = filepaths_after_pipeline_run or []

    def _log_artifacts(self, artifacts=[]):
        if self.enable_mlflow:

            from mlflow import log_artifact

            for path in artifacts:
                log_artifact(path)

    @hook_impl
    def before_pipeline_run(
        self, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
    ):
        self._log_artifacts(self.filepaths_before_pipeline_run)

    @hook_impl
    def after_pipeline_run(
        self, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
    ):
        self._log_artifacts(self.filepaths_after_pipeline_run)

    def _log_datasets(self, catalog, datasets):
        if self.enable_mlflow:

            from mlflow import log_artifact

            for d in datasets:
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
                        log.warning("_filepath of '{}' was not found.".format(d))

    @hook_impl
    def after_node_run(self, node, catalog, inputs, outputs):
        datasets = [d for d in outputs.keys() if d in self.datasets_after_node_run]
        self._log_datasets(catalog, datasets)
