from importlib.util import find_spec
from logging import getLogger
import tempfile
from typing import Any, Dict, Union  # NOQA

from kedro.io import AbstractDataSet, DataCatalog
from kedro.pipeline.node import Node
from kedro.pipeline import Pipeline

from .mlflow_utils import (
    hook_impl,
    mlflow_log_artifacts,
    mlflow_log_metrics,
    mlflow_log_params,
    mlflow_log_values,
)


log = getLogger(__name__)


def get_kedro_runner():
    import inspect
    from kedro.runner import AbstractRunner

    return next(
        caller[0].f_locals.get("runner")
        for caller in inspect.stack()
        if isinstance(caller[0].f_locals.get("runner"), AbstractRunner)
    )


def running_parallel():
    from kedro.runner import ParallelRunner

    return isinstance(get_kedro_runner(), ParallelRunner)


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
        auto: bool = True,
        mlflow_catalog: Dict[str, Union[str, AbstractDataSet]] = {},
        enable_mlflow: bool = True,
    ):
        """
        Args:
            auto: If True, each dataset (Python func input/output) not listed in the catalog
            will be logged following the same rule as "a" option below.
            mlflow_catalog: [Deprecated in favor of MLflowDataSet] Specify how to log each dataset
            (Python func input/output).

                - If set to "p", the value will be saved/loaded as an MLflow parameter (string).

                - If set to "m", the value will be saved/loaded as an MLflow metric (numeric).

                - If set to "a", the value will be saved/loaded based on the data type.

                    - If the data type is either {float, int}, the value will be saved/loaded as an MLflow metric.

                    - If the data type is either {str, list, tuple, set}, the value will be saved/load as an MLflow parameter.

                    - If the data type is dict, the value will be flattened with dot (".") as the separator and then saved/loaded as either an MLflow metric or parameter based on each data type as explained above.

                - If set to either {"json", "csv", "xls", "parquet", "png", "jpg", "jpeg", "img", "pkl", "txt", "yml", "yaml"}, the backend dataset instance will be created accordingly to save/load as an MLflow artifact.

                - If set to a Kedro DataSet object or a dictionary, it will be used as the backend dataset to save/load as an MLflow artifact.

                - If set to None (default), MLflow logging will be skipped.
            enable_mlflow: Enable logging to MLflow.
        """
        self.enable_mlflow = find_spec("mlflow") and enable_mlflow
        self.mlflow_catalog = mlflow_catalog
        self.auto = auto

    @hook_impl
    def before_pipeline_run(
        self, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
    ) -> None:
        for dataset_name in catalog._data_sets:
            if catalog._data_sets[dataset_name].__class__.__name__ == "MLflowDataSet":
                setattr(catalog._data_sets[dataset_name], "_dataset_name", dataset_name)
                setattr(
                    catalog._data_sets[dataset_name],
                    "_running_parallel",
                    running_parallel(),
                )
                catalog._data_sets[dataset_name]._init_dataset()

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
            if not self.auto:
                return

            mlflow_log_values({name: value}, enable_mlflow=self.enable_mlflow)
            return

        catalog_instance = self.mlflow_catalog.get(name, None)
        if not catalog_instance:
            return
        elif isinstance(catalog_instance, str):
            if catalog_instance in {"p"}:
                mlflow_log_params({name: value}, enable_mlflow=self.enable_mlflow)
            elif catalog_instance in {"m"}:
                mlflow_log_metrics({name: value}, enable_mlflow=self.enable_mlflow)
            elif catalog_instance in {"a"}:
                mlflow_log_values({name: value}, enable_mlflow=self.enable_mlflow)
            elif catalog_instance in datasets_dict:
                ds = datasets_dict.get(catalog_instance)
                fp = tempfile.gettempdir() + "/" + name + "." + catalog_instance
                ds(filepath=fp).save(value)
                mlflow_log_artifacts([fp], enable_mlflow=self.enable_mlflow)
            else:
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
