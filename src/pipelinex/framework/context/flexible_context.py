import kedro
from .pipelines_in_parameters_context import PipelinesInParametersContext
from .only_missing_string_runner_context import (
    OnlyMissingStringRunnerDefaultOptionContext,
)

from .catalog_sugar_context import CatalogSyntacticSugarContext


class FlexibleContext(
    PipelinesInParametersContext,
    CatalogSyntacticSugarContext,
    OnlyMissingStringRunnerDefaultOptionContext,
):
    project_name = "KedroProject"
    project_version = kedro.__version__


class MLflowFlexibleContext(FlexibleContext):
    """ Deprecated alias for FlexibleContext for backward-compatibility """

    pass
