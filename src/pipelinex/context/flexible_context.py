from .pipelines_in_parameters_context import PipelinesInParametersContext
from .only_missing_string_runner_context import (
    OnlyMissingStringRunnerDefaultOptionContext,
)
from .catalog_sugar_context import CatalogSyntacticSugarContext
from .mlflow_context import MLflowContext


class BaseFlexibleContext(
    PipelinesInParametersContext,
    CatalogSyntacticSugarContext,
    OnlyMissingStringRunnerDefaultOptionContext,
):
    project_name = "KedroProject"
    project_version = "0.15.9"  # Kedro version


class FlexibleContext(MLflowContext, BaseFlexibleContext):
    """ The Kedro context with flexible options """

    pass


class MLflowFlexibleContext(FlexibleContext):
    """ Deprecated alias for FlexibleContext for backward-compatibility """

    pass
