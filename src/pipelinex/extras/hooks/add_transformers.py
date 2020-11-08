from typing import List, Tuple, Union

from kedro.io import DataCatalog
from kedro.io.transformers import AbstractTransformer

try:
    from kedro.framework.hooks import hook_impl
except ModuleNotFoundError:

    def hook_impl(func):
        return func


class AddTransformersHook:
    """Hook to add transformers."""

    def __init__(
        self,
        transformers: Union[
            AbstractTransformer, List[AbstractTransformer], Tuple[AbstractTransformer]
        ],
    ):
        """
        Args:
            transformers: transformers to add.
        """
        transformers = transformers or []
        if isinstance(transformers, AbstractTransformer):
            transformers = [transformers]
        self._transformers = transformers

    @hook_impl
    def after_catalog_created(self, catalog: DataCatalog) -> None:
        for t in self._transformers:
            catalog.add_transformer(t)
