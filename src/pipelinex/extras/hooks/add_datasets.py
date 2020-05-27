from typing import Dict

from kedro.io import AbstractDataSet, DataCatalog

try:
    from kedro.framework.hooks import hook_impl
except ModuleNotFoundError:

    def hook_impl(func):
        return func


class AddDataSetsHook:
    """ Hook to add data sets.
    """

    def __init__(
        self, dataset_dict: Dict[str, AbstractDataSet],
    ):
        """
        Args:
            dataset_dict: datasets to add.
        """
        assert isinstance(dataset_dict, dict), "{} is not a dict.".format(dataset_dict)
        self._dataset_dict = dataset_dict

    @hook_impl
    def after_catalog_created(self, catalog: DataCatalog) -> None:
        catalog.add_feed_dict(self._dataset_dict)
