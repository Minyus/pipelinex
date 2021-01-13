from importlib.util import find_spec
from logging import getLogger
import tempfile
from typing import Any, Dict, Union  # NOQA

from kedro.io import DataCatalog
from kedro.pipeline.node import Node
from kedro.io import AbstractDataSet

from .mlflow_utils import (
    hook_impl,
    mlflow_log_metrics,
    mlflow_log_params,
    mlflow_log_artifacts,
)


log = getLogger(__name__)

datasets_dict = {}
try:
    from kedro.extras.datasets.json import JSONDataSet

    datasets_dict["json"] = JSONDataSet
except ImportError:
    pass
try:
    from kedro.extras.datasets.pandas import CSVDataSet, ExcelDataSet, ParquetDataSet

    datasets_dict["csv"] = CSVDataSet
    datasets_dict["xls"] = ExcelDataSet
    datasets_dict["parquet"] = ParquetDataSet
except ImportError:
    pass
try:
    from kedro.extras.datasets.pickle import PickleDataSet

    datasets_dict["pkl"] = PickleDataSet
    datasets_dict["pickle"] = PickleDataSet

except ImportError:
    pass
try:
    from kedro.extras.datasets.pillow import ImageDataSet

    datasets_dict["png"] = ImageDataSet
    datasets_dict["jpg"] = ImageDataSet
    datasets_dict["jpeg"] = ImageDataSet
    datasets_dict["img"] = ImageDataSet
except ImportError:
    pass
try:
    from kedro.extras.datasets.text import TextDataSet

    datasets_dict["txt"] = TextDataSet
except ImportError:
    pass
try:
    from kedro.extras.datasets.yaml import YAMLDataSet

    datasets_dict["yml"] = YAMLDataSet
    datasets_dict["yaml"] = YAMLDataSet
except ImportError:
    pass


def mlflow_log_dataset(dataset, enable_mlflow=True):
    fp = getattr(dataset, "_filepath", None)
    if not fp:
        low_ds = getattr(dataset, "_dataset", None)
        if low_ds:
            fp = getattr(low_ds, "_filepath", None)
    if not fp:
        log.warning("_filepath of '{}' was not found.".format(d))
        return
    mlflow_log_artifacts([fp], enable_mlflow=enable_mlflow)


class MLflowCatalogLoggerHook:
    """Logs datasets to MLflow"""

    _logged_set = set()

    def __init__(
        self,
        enable_mlflow: bool = True,
        mlflow_catalog: Dict[str, Union[str, AbstractDataSet]] = {},
    ):
        """
        Args:
            enable_mlflow: Enable logging to MLflow.
        """
        self.enable_mlflow = find_spec("mlflow") and enable_mlflow
        self.mlflow_catalog = mlflow_catalog

    @hook_impl
    def after_node_run(
        self,
        node: Node,
        catalog: DataCatalog,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ):

        for name, value in inputs.items():
            if name not in self._logged_set:
                self._logged_set.add(name)
                self._log_dataset(name, value)

        for name, value in outputs.items():
            if name not in self._logged_set:
                self._logged_set.add(name)
                self._log_dataset(name, value)

    def _log_dataset(self, name: str, value: Any):
        if name not in self.mlflow_catalog:
            return
        catalog_instance = self.mlflow_catalog.get(name)
        if catalog_instance is None:
            log.warning(
                "'{}' is not supported as mlflow_catalog entry and ignored.".format(
                    catalog_instance
                )
            )
            return
        elif isinstance(catalog_instance, str):
            if catalog_instance in {"param", "p", "$", ""}:
                mlflow_log_params({name: value}, enable_mlflow=self.enable_mlflow)
            elif catalog_instance in {"metric", "m", "#"}:
                mlflow_log_metrics({name: value}, enable_mlflow=self.enable_mlflow)
            elif catalog_instance in datasets_dict:
                ds = datasets_dict.get(catalog_instance)
                fp = tempfile.gettempdir() + "/" + name + "." + catalog_instance
                ds(filepath=fp).save(value)
                mlflow_log_artifacts([fp], enable_mlflow=self.enable_mlflow)
            elif isinstance(catalog_instance, str):
                log.warning(
                    "'{}' is not supported as mlflow_catalog entry and ignored.".format(
                        catalog_instance
                    )
                )
                return
        else:
            if not hasattr(catalog_instance, "save"):
                log.warning("'save' attr is not found in mlflow_catalog instance.")
            catalog_instance.save(value)
            mlflow_log_dataset(catalog_instance, enable_mlflow=self.enable_mlflow)
