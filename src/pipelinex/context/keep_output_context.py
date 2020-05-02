from kedro.context import KedroContext
import logging

log = logging.getLogger(__name__)


class KeepOutputContext(KedroContext):
    """Keep the output datasets in the catalog."""

    def run(
        self, **kwargs  # type: Any
    ):
        # type: (...) -> Dict[str, Any]
        d = super().run(**kwargs)
        self.catalog.add_feed_dict(d, replace=True)
        return d
