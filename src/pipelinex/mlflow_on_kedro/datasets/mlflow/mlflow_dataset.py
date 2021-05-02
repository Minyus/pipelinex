import logging
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Union

from kedro.extras.datasets.pickle import PickleDataSet
from kedro.io import MemoryDataSet
from kedro.io.core import AbstractDataSet

from pipelinex.mlflow_on_kedro.hooks.mlflow.mlflow_utils import (
    mlflow_log_artifacts,
    mlflow_log_metrics,
    mlflow_log_params,
    mlflow_log_values,
)

log = logging.getLogger(__name__)

dataset_dicts = {
    "json": {"type": "json.JSONDataSet"},
    "csv": {"type": "pandas.CSVDataSet"},
    "xls": {"type": "pandas.ExcelDataSet"},
    "parquet": {"type": "pandas.ParquetDataSet"},
    "pkl": {"type": "pickle.PickleDataSet"},
    "png": {"type": "pillow.ImageDataSet"},
    "jpg": {"type": "pillow.ImageDataSet"},
    "jpeg": {"type": "pillow.ImageDataSet"},
    "img": {"type": "pillow.ImageDataSet"},
    "txt": {"type": "text.TextDataSet"},
    "yaml": {"type": "yaml.YAMLDataSet"},
    "yml": {"type": "yaml.YAMLDataSet"},
}


class MLflowDataSet(AbstractDataSet):
    """``MLflowDataSet`` saves data to, and loads data from MLflow.

    You can also specify a ``MLflowDataSet`` in catalog.yml

    Example:
    ::

        >>> test_ds:
        >>>    type: MLflowDataSet
        >>>    dataset: pkl
    """

    def __init__(
        self,
        dataset: Union[AbstractDataSet, Dict, str] = None,
        filepath: str = None,
        dataset_name: str = None,
        saving_tracking_uri: str = None,
        saving_experiment_name: str = None,
        saving_run_id: str = None,
        loading_tracking_uri: str = None,
        loading_run_id: str = None,
        caching: bool = True,
        copy_mode: str = None,
        file_caching: bool = True,
    ):
        """

        Args:
            dataset: Specify how to treat the dataset as an MLflow metric, parameter, or artifact.

                - If set to "p", the value will be saved/loaded as an MLflow parameter (string).

                - If set to "m", the value will be saved/loaded as an MLflow metric (numeric).

                - If set to "a", the value will be saved/loaded based on the data type.

                    - If the data type is either {float, int}, the value will be saved/loaded as an MLflow metric.

                    - If the data type is either {str, list, tuple, set}, the value will be saved/load as an MLflow parameter.

                    - If the data type is dict, the value will be flattened with dot (".") as the separator and then saved/loaded as either an MLflow metric or parameter based on each data type as explained above.

                - If set to either {"json", "csv", "xls", "parquet", "png", "jpg", "jpeg", "img", "pkl", "txt", "yml", "yaml"}, the backend dataset instance will be created accordingly to save/load as an MLflow artifact.

                - If set to a Kedro DataSet object or a dictionary, it will be used as the backend dataset to save/load as an MLflow artifact.

                - If set to None (default), MLflow logging will be skipped.
            filepath: File path, usually in local file system, to save to and load from.
                Used only if the dataset arg is a string.
                If None (default), ``<temp directory>/<dataset_name arg>.<dataset arg>`` is used.
            dataset_name: Used only if the dataset arg is a string and filepath arg is None.
                If None (default), Python object ID is used, but will be overwritten by
                MLflowCatalogLoggerHook.
            saving_tracking_uri: MLflow Tracking URI to save to.
                If None (default), MLFLOW_TRACKING_URI environment variable is used.
            saving_experiment_name: MLflow experiment name to save to.
                If None (default), new experiment will not be created or started.
                Ignored if saving_run_id is set.
            saving_run_id: An existing MLflow experiment run ID to save to.
                If None (default), no existing experiment run will be resumed.
            loading_tracking_uri: MLflow Tracking URI to load from.
                If None (default), MLFLOW_TRACKING_URI environment variable is used.
            loading_run_id: MLflow experiment run ID to load from.
                If None (default), current active run ID will be used if available.
            caching: Enable caching if parallel runner is not used. True in default.
            copy_mode: The copy mode used to copy the data. Possible
                values are: "deepcopy", "copy" and "assign". If not
                provided, it is inferred based on the data type.
                Ignored if caching arg is False.
            file_caching: Attempt to use the file at filepath when loading if no cache found
                in memory. True in default.

        """
        self.dataset = dataset or MemoryDataSet()
        self.filepath = filepath
        self.dataset_name = dataset_name
        self.saving_tracking_uri = saving_tracking_uri
        self.saving_experiment_name = saving_experiment_name
        self.saving_run_id = saving_run_id
        self.loading_tracking_uri = loading_tracking_uri
        self.loading_run_id = loading_run_id
        self.caching = caching
        self.file_caching = file_caching
        self.copy_mode = copy_mode
        self._dataset_name = str(id(self))

        if isinstance(dataset, str):
            if (dataset not in {"p", "m"}) and (dataset not in dataset_dicts):
                raise ValueError(
                    "`dataset`: {} not supported. Specify one of {}.".format(
                        dataset, list(dataset_dicts.keys())
                    )
                )
        self._ready = False
        self._running_parallel = None
        self._cache = None

    def _init_dataset(self):

        if not getattr(self, "_ready", None):
            self._ready = True
            self.dataset_name = self.dataset_name or self._dataset_name
            _dataset = self.dataset
            if isinstance(self.dataset, str):
                dataset_dict = dataset_dicts.get(
                    self.dataset, {"type": "pickle.PickleDataSet"}
                )
                dataset_dict["filepath"] = self.filepath = (
                    self.filepath
                    or tempfile.gettempdir()
                    + "/"
                    + self.dataset_name
                    + "."
                    + self.dataset
                )
                _dataset = dataset_dict

            if isinstance(_dataset, dict):
                self._dataset = AbstractDataSet.from_config(
                    self._dataset_name, _dataset
                )
            elif isinstance(_dataset, AbstractDataSet):
                self._dataset = _dataset
            else:
                raise ValueError(
                    "The argument type of `dataset` should be either a dict/YAML "
                    "representation of the dataset, or the actual dataset object."
                )

            _filepath = getattr(self._dataset, "_filepath", None)
            if _filepath:
                self.filepath = str(_filepath)

            if self.caching and (not self._running_parallel):
                self._cache = MemoryDataSet(copy_mode=self.copy_mode)

    def _release(self) -> None:
        self._init_dataset()
        self._dataset.release()
        if self._cache:
            self._cache.release()

    def _describe(self) -> Dict[str, Any]:
        return {
            "dataset": self._dataset._describe()
            if getattr(self, "_ready", None)
            else self.dataset,  # pylint: disable=protected-access
            "filepath": self.filepath,
            "saving_tracking_uri": self.saving_tracking_uri,
            "saving_experiment_name": self.saving_experiment_name,
            "saving_run_id": self.saving_run_id,
            "loading_tracking_uri": self.loading_tracking_uri,
            "loading_run_id": self.loading_run_id,
        }

    def _load(self):
        self._init_dataset()

        if self._cache and self._cache.exists():
            return self._cache.load()

        if self.file_caching and self._dataset.exists():
            return self._dataset.load()

        import mlflow

        client = mlflow.tracking.MlflowClient(tracking_uri=self.loading_tracking_uri)

        self.loading_run_id = self.loading_run_id or mlflow.active_run().info.run_id

        if self.dataset in {"p"}:
            run = client.get_run(self.loading_run_id)
            value = run.data.params.get(self.dataset_name, None)
            if value is None:
                raise KeyError(
                    "param '{}' not found in run_id '{}'.".format(
                        self.dataset_name, self.loading_run_id
                    )
                )

            PickleDataSet(filepath=self.filepath).save(value)

        elif self.dataset in {"m"}:
            run = client.get_run(self.loading_run_id)
            value = run.data.metrics.get(self.dataset_name, None)
            if value is None:
                raise KeyError(
                    "metric '{}' not found in run_id '{}'.".format(
                        self.dataset_name, self.loading_run_id
                    )
                )
            PickleDataSet(filepath=self.filepath).save(value)

        else:
            p = Path(self.filepath)

            dst_path = tempfile.gettempdir()
            downloaded_path = client.download_artifacts(
                run_id=self.loading_run_id,
                path=p.name,
                dst_path=dst_path,
            )
            if Path(downloaded_path) != p:
                Path(downloaded_path).rename(p)

        return self._dataset.load()

    def _save(self, data: Any) -> None:
        self._init_dataset()

        self._dataset.save(data)

        if self._cache:
            self._cache.save(data)

        if find_spec("mlflow"):
            import mlflow

            if self.saving_tracking_uri:
                mlflow.set_tracking_uri(self.saving_tracking_uri)

            if self.saving_run_id:
                mlflow.start_run(run_id=self.saving_run_id)

            elif self.saving_experiment_name:
                experiment_id = mlflow.get_experiment_by_name(
                    self.saving_experiment_name
                ).experiment_id
                mlflow.start_run(run_id=self.saving_run_id, experiment_id=experiment_id)

            if self.dataset in {"p"}:
                mlflow_log_params({self.dataset_name: data})
            elif self.dataset in {"m"}:
                mlflow_log_metrics({self.dataset_name: data})
            elif self.dataset in {"a"}:
                mlflow_log_values({self.dataset_name: data})
            else:
                mlflow_log_artifacts([self.filepath])

            if self.saving_run_id or self.saving_experiment_name:
                mlflow.end_run()

    def _exists(self) -> bool:
        self._init_dataset()
        if self._cache:
            return self._cache.exists()
        else:
            return False

    def __getstate__(self):
        return self.__dict__
