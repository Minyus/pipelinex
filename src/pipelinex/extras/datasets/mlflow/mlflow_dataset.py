import logging
from importlib.util import find_spec
from pathlib import Path
import tempfile
from typing import Any, Dict, Union

from kedro.io.core import AbstractDataSet, Version
from kedro.io import CachedDataSet, MemoryDataSet
from kedro.extras.datasets.pickle import PickleDataSet

from pipelinex.extras.hooks.mlflow.mlflow_utils import (
    mlflow_log_artifacts,
    mlflow_log_metrics,
    mlflow_log_params,
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


class MLflowDataSet(CachedDataSet):
    """``MLflowDataSet`` saves data to, and loads data from MLflow.
    ``MLflowDataSet`` inherits ``CachedDataSet``.
    You can also specify a ``MLflowDataSet`` in catalog.yml:
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
        version: Version = None,
        copy_mode: str = None,
    ):
        """
        dataset: A Kedro DataSet object or a dictionary used to save/load.
            If set to either {"json", "csv", "xls", "parquet", "png", "jpg", "jpeg", "img",
            "pkl", "txt", "yml", "yaml"}, dataset instance will be created accordingly with
            filepath arg.
            If set to "p", the value will be saved/loaded as a parameter (string).
            If set to "m", the value will be saved/loaded as a metric (numeric).
            If None (default), MLflow will not be used.
        filepath: File path, usually in local file system, to save to and load from.
            Used only if the dataset arg is a string.
            If None (default), `<temp directory>/<dataset_name arg>.<dataset arg>` is used.
        dataset_name: Used only if the dataset arg is a string and filepath arg is None.
            If None (default), Python object ID is used, but recommended to overwrite by
            a Kedro hook.
        saving_tracking_uri: MLflow Tracking URI to save to.
            If None (default), MLFLOW_TRACKING_URI environment variable is used.
        saving_experiment_name: MLflow experiment name to save to.
            If None (default), new experiment will not be created or started.
            Ignored if saving_run_id is set.
        saving_run_id: MLflow experiment ID to save to.
            If None (default), existing experiment  will not be resumed.
        loading_tracking_uri: MLflow Tracking URI to load from.
            If None (default), MLFLOW_TRACKING_URI environment variable is used.
        loading_run_id: MLflow experiment run ID to load from.
            If None (default), attempt to load will fail.
        version: If specified, should be an instance of
            ``kedro.io.core.Version``. If its ``load`` attribute is
            None, the latest version will be loaded. If its ``save``
            attribute is None, save version will be autogenerated.
        copy_mode: The copy mode used to copy the data. Possible
            values are: "deepcopy", "copy" and "assign". If not
            provided, it is inferred based on the data type.
        """
        self.dataset = dataset or MemoryDataSet(copy_mode=copy_mode)
        self.filepath = filepath
        self.dataset_name = dataset_name
        self.saving_tracking_uri = saving_tracking_uri
        self.saving_experiment_name = saving_experiment_name
        self.saving_run_id = saving_run_id
        self.loading_tracking_uri = loading_tracking_uri
        self.loading_run_id = loading_run_id
        self.version = version
        self.copy_mode = copy_mode
        self._dataset_name = str(id(self))

        if isinstance(dataset, str):
            if (dataset not in {"p", "m"}) and (dataset not in dataset_dicts):
                raise ValueError(
                    "`dataset`: {} not supported. Specify one of {}.".format(
                        dataset, list(dataset_dicts.keys())
                    )
                )

    def _init_dataset(self):
        if not hasattr(self, "_dataset"):
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

            super().__init__(
                dataset=_dataset,
                version=self.version,
                copy_mode=self.copy_mode,
            )

            self.filepath = getattr(self._dataset, "_filepath", None) or self.filepath

    def _describe(self) -> Dict[str, Any]:
        self._init_dataset()
        return {
            "dataset": getattr(
                getattr(self, "_dataset", self.dataset), "_describe", lambda: None
            )(),  # pylint: disable=protected-access
            "filepath": self.filepath,
            "saving_tracking_uri": self.saving_tracking_uri,
            "saving_experiment_name": self.saving_experiment_name,
            "saving_run_id": self.saving_run_id,
            "loading_tracking_uri": self.loading_tracking_uri,
            "loading_run_id": self.loading_run_id,
            "cache": getattr(
                getattr(self, "_cache", None), "_describe", lambda: None
            )(),  # pylint: disable=protected-access
        }

    def _load(self):
        self._init_dataset()
        if not self._exists():
            if find_spec("mlflow"):
                import mlflow

                client = mlflow.tracking.MLflowClient(
                    tracking_uri=self.loading_tracking_uri
                )
                if self.dataset in {"P"}:
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
                    downloaded_path = client.download_artifacts(
                        run_id=self.loading_run_id,
                        path=self.path,
                        dst_path=tempfile.gettempdir(),
                    )
                    Path(downloaded_path).rename(self.filepath)
        return super()._load()

    def _save(self, data: Any) -> None:
        self._init_dataset()
        super()._save(data)

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
            else:
                mlflow_log_artifacts([self.filepath])

            if self.saving_run_id or self.saving_experiment_name:
                mlflow.end_run()
