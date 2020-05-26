from importlib.util import find_spec


try:
    from kedro.framework.hooks import hook_impl
except ModuleNotFoundError:

    def hook_impl(func):
        return func


class MLflowOutputsLoggerHook:
    """ Log output datasets of (list of) float, int, and str classes.
    """

    def __init__(self, enable_mlflow: bool = True):

        """
        Args:
            enable_mlflow: Enable logging to MLflow.
        """
        self.enable_mlflow = find_spec("mlflow") and enable_mlflow

    @hook_impl
    def after_node_run(self, node, catalog, inputs, outputs):
        if self.enable_mlflow:

            from mlflow import log_metric, log_param

            for name, value in outputs.items():
                if isinstance(value, str):
                    log_param(name, value[:250])
                    continue
                if isinstance(value, (float, int)):
                    log_metric(name, float(value))
                    continue
                if isinstance(value, (list, tuple)):
                    if all([isinstance(e, str) for e in value]):
                        log_param(name, ", ".join(value))
                        continue
                    for i, e in enumerate(value):
                        if isinstance(e, (float, int)):
                            log_metric(name, float(e), step=i)
                        else:
                            break
