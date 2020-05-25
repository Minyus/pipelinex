import kedro
from .pipelines_in_parameters_context import PipelinesInParametersContext
from .flexible_run_context import FlexibleRunContext

from .catalog_sugar_context import CatalogSyntacticSugarContext


class FlexibleContext(
    PipelinesInParametersContext, CatalogSyntacticSugarContext, FlexibleRunContext,
):
    project_name = "KedroProject"
    project_version = kedro.__version__


class MLflowFlexibleContext(FlexibleContext):
    """ Deprecated alias for FlexibleContext for backward-compatibility """

    pass
