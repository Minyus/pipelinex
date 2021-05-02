from importlib.util import find_spec
from logging import getLogger
from typing import Any, Dict, List  # NOQA

from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node

from .mlflow_utils import hook_impl, mlflow_log_artifacts

log = getLogger(__name__)


class MLflowArtifactsLoggerHook:
    """Logs artifacts of specified file paths and dataset names to MLflow"""

    def __init__(
        self,
        filepaths_before_pipeline_run: List[str] = None,
        filepaths_after_pipeline_run: List[str] = None,
        datasets_after_node_run: List[str] = None,
        enable_mlflow: bool = True,
    ):
        """
        Args:
            filepaths_before_pipeline_run: The file paths of artifacts to log
                before the pipeline is run.
            filepaths_after_pipeline_run: The file paths of artifacts to log
                after the pipeline is run.
            datasets_after_node_run: The dataset names to log after the node is run.
            enable_mlflow: Enable logging to MLflow.
        """
        self.enable_mlflow = find_spec("mlflow") and enable_mlflow
        self.filepaths_before_pipeline_run = filepaths_before_pipeline_run or []
        self.datasets_after_node_run = datasets_after_node_run or []
        self.filepaths_after_pipeline_run = filepaths_after_pipeline_run or []

    def _log_artifacts(self, artifacts=[]):
        mlflow_log_artifacts(artifacts, enable_mlflow=self.enable_mlflow)

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
        for d in datasets:
            ds = getattr(catalog.datasets, d, None)
            if not ds:
                log.warning("{} was not found in catalog.datasets.".format(d))
                return
            fp = getattr(ds, "_filepath", None)
            if not fp:
                low_ds = getattr(ds, "_dataset", None)
                if low_ds:
                    fp = getattr(low_ds, "_filepath", None)
            if not fp:
                log.warning("_filepath of '{}' was not found.".format(d))
                return
            mlflow_log_artifacts(fp, enable_mlflow=self.enable_mlflow)

    @hook_impl
    def after_node_run(
        self,
        node: Node,
        catalog: DataCatalog,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ):
        datasets = [d for d in outputs.keys() if d in self.datasets_after_node_run]
        self._log_datasets(catalog, datasets)
