import copy
import logging
from pathlib import Path
from typing import Any, Dict  # NOQA

from kedro.io import DataCatalog

from pipelinex import HatchDict

from .context import KedroContext

log = logging.getLogger(__name__)


class FlexibleCatalogContext(KedroContext):
    """Convert Kedrex's Syntactic Sugar to pure Kedro Catalog."""

    def _create_catalog(  # pylint: disable=no-self-use,too-many-arguments
        self,
        conf_catalog: Dict[str, Any],
        conf_creds: Dict[str, Any],
        save_version: str = None,
        journal: Any = None,
        load_versions: Dict[str, str] = None,
    ) -> DataCatalog:
        """DataCatalog instantiation.
        Allow whether to apply CachedDataSet using `cached` key.
        Returns:
            DataCatalog defined in `catalog.yml`.
        """
        conf_catalog = self._format_kedro_catalog(conf_catalog)
        return DataCatalog.from_config(
            conf_catalog,
            conf_creds,
            save_version=save_version,
            load_versions=load_versions,
        )

    def _format_kedro_catalog(self, conf_catalog):

        conf_catalog = HatchDict(conf_catalog).get()

        default_dict = {}
        if "/" in conf_catalog:
            default_dict = conf_catalog.pop("/")

        if "PIPELINE_JSON_TEXT" in conf_catalog:
            pipeline_json_text_dataset = conf_catalog.pop("PIPELINE_JSON_TEXT")
            assert isinstance(pipeline_json_text_dataset, dict)
            pipeline_json_text_dataset.setdefault(
                "type", "kedro.extras.datasets.text.TextDataSet"
            )
            self._pipeline_json_text_dataset = HatchDict(
                pipeline_json_text_dataset, obj_key="type"
            ).get()

        conf_catalog_processed = {}

        for ds_name, ds_dict_ in conf_catalog.items():
            ds_dict = copy.deepcopy(default_dict)
            if isinstance(ds_dict_, dict):
                ds_dict.update(ds_dict_)
            _check_type(ds_dict)
            ds_name, ds_dict = self._format_kedro_dataset(ds_name, ds_dict)
            conf_catalog_processed[ds_name] = ds_dict
        return conf_catalog_processed

    def _format_kedro_dataset(self, ds_name, ds_dict):
        ds_name, ds_dict = self._set_filepath(ds_name, ds_dict)
        ds_name, ds_dict = self._enable_caching(ds_name, ds_dict)
        return ds_name, ds_dict

    def _set_filepath(self, ds_name, ds_dict):
        if not any(
            [
                (key in ds_dict)
                for key in ["filepath", "path", "url", "urls", "table_name", "sql"]
            ]
        ):
            ds_dict["filepath"] = ds_name
            ds_name = Path(ds_name).stem
        return ds_name, ds_dict

    def _enable_caching(self, ds_name, ds_dict):
        cached = False
        if "cached" in ds_dict:
            cached = ds_dict.pop("cached")
        if cached and (ds_dict.get("type") != "kedro.io.CachedDataSet"):
            ds_dict = {
                "type": "kedro.io.CachedDataSet",
                "dataset": ds_dict,
            }
        return ds_name, ds_dict


def _check_type(ds_dict):
    type = ds_dict.get("type")
    if type and (not type.endswith("DataSet")):
        log.warning("type: '{}' does not end with 'DataSet'.".format(type))
