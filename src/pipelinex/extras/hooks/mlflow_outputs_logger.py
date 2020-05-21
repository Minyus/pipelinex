from importlib.util import find_spec
from logging import getLogger

try:
    from kedro.framework.hooks import hook_impl
except ModuleNotFoundError:

    def hook_impl(func):
        return func


log = getLogger(__name__)


if find_spec("mlflow"):
    from mlflow import log_metric, log_param

    class MLflowOutputsLogger:
        @hook_impl
        def after_node_run(self, node, catalog, inputs, outputs):
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


else:

    class MLflowOutputsLogger:
        @hook_impl
        def after_node_run(self, node, catalog, inputs, outputs):
            pass
