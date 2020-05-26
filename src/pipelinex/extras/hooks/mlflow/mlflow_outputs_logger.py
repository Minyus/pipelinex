from importlib.util import find_spec

from .mlflow_utils import hook_impl, mlflow_log_metrics, mlflow_log_params


class MLflowOutputsLoggerHook:
    """ Logs output datasets of (list of) float/int and str classes to MLflow
    """

    def __init__(self, enable_mlflow: bool = True):
        """
        Args:
            enable_mlflow: Enable logging to MLflow.
        """
        self.enable_mlflow = find_spec("mlflow") and enable_mlflow

    @hook_impl
    def after_node_run(self, node, catalog, inputs, outputs):

        for name, value in outputs.items():
            if isinstance(value, str):
                mlflow_log_params({name: value}, enable_mlflow=self.enable_mlflow)
                continue
            if isinstance(value, (float, int)):
                mlflow_log_metrics({name: value}, enable_mlflow=self.enable_mlflow)
                continue
            if isinstance(value, (list, tuple)):
                if all([isinstance(e, str) for e in value]):
                    mlflow_log_params(
                        {name: ", ".join(value)}, enable_mlflow=self.enable_mlflow
                    )
                    continue
                for i, e in enumerate(value):
                    if isinstance(e, (float, int)):
                        mlflow_log_metrics(
                            {name: e}, step=i, enable_mlflow=self.enable_mlflow
                        )
                    else:
                        break
