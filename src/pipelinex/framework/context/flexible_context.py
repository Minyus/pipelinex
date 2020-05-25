import kedro
from .flexible_parameters_context import FlexibleParametersContext
from .flexible_run_context import FlexibleRunContext

from .catalog_sugar_context import CatalogSyntacticSugarContext


class FlexibleContext(
    FlexibleParametersContext, CatalogSyntacticSugarContext, FlexibleRunContext,
):
    project_name = "KedroProject"
    project_version = kedro.__version__


class MLflowFlexibleContext(FlexibleContext):
    """ Deprecated alias for FlexibleContext for backward-compatibility """

    pass
