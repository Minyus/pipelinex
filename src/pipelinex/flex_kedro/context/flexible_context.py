import kedro

from .flexible_catalog_context import FlexibleCatalogContext
from .flexible_parameters_context import FlexibleParametersContext
from .flexible_run_context import FlexibleRunContext


class FlexibleContext(
    FlexibleParametersContext,
    FlexibleCatalogContext,
    FlexibleRunContext,
):
    project_name = "KedroProject"
    project_version = kedro.__version__


class MLflowFlexibleContext(FlexibleContext):
    """Deprecated alias for FlexibleContext for backward-compatibility"""

    pass
