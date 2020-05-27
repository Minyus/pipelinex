from typing import Dict

from kedro.io import AbstractDataSet, DataCatalog

try:
    from kedro.framework.hooks import hook_impl
except ModuleNotFoundError:

    def hook_impl(func):
        return func


class AddCatalogDictHook:
    """ Hook to add data sets.
    """

    def __init__(
        self, catalog_dict: Dict[str, AbstractDataSet],
    ):
        """
        Args:
            catalog_dict: catalog_dict to add.
        """
        assert isinstance(catalog_dict, dict), "{} is not a dict.".format(catalog_dict)
        self._catalog_dict = catalog_dict

    @hook_impl
    def after_catalog_created(self, catalog: DataCatalog) -> None:
        catalog.add_feed_dict(self._catalog_dict)
