from importlib.util import find_spec

from .mlflow_utils import hook_impl, mlflow_log_metrics, mlflow_log_params


class MLflowDataSetsLoggerHook:
    """Logs datasets of (list of) float/int and str classes to MLflow"""

    _logged_set = set()

    def __init__(self, enable_mlflow: bool = True):
        """
        Args:
            enable_mlflow: Enable logging to MLflow.
        """
        self.enable_mlflow = find_spec("mlflow") and enable_mlflow

    @hook_impl
    def after_node_run(self, node, catalog, inputs, outputs):
        for name, value in inputs.items():
            if name not in self._logged_set:
                self._logged_set.add(name)
                self._log_dataset(name, value)

        for name, value in outputs.items():
            if name not in self._logged_set:
                self._logged_set.add(name)
                self._log_dataset(name, value)

    def _log_dataset(self, name, value):
        if isinstance(value, str):
            mlflow_log_params({name: value}, enable_mlflow=self.enable_mlflow)
        elif isinstance(value, (float, int)):
            mlflow_log_metrics({name: value}, enable_mlflow=self.enable_mlflow)
        elif isinstance(value, (list, tuple, set, dict)):
            mlflow_log_params(
                {name: "{}".format(value)[:100]}, enable_mlflow=self.enable_mlflow
            )


class MLflowOutputsLoggerHook(MLflowDataSetsLoggerHook):
    """Deprecated alias for `MLflowOutputsLoggerHook`"""
