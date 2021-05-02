import os
from importlib.util import find_spec
from logging import getLogger
from typing import List  # NOQA

from .mlflow_utils import hook_impl, mlflow_log_metrics, mlflow_log_params

log = getLogger(__name__)


def env_vars_to_dict(env_vars=[], prefix=""):
    if isinstance(env_vars, str):
        if env_vars == "__ALL__":
            return dict(os.environ)
        else:
            env_vars = [env_vars]
    elif not isinstance(env_vars, (list, tuple, set)):
        raise "{} not supported as env_vars arg".format(type(env_vars))
    return {(prefix + ev): os.environ.get(ev) for ev in env_vars}


def log_param_env_vars(env_vars=[], prefix="", enable_mlflow=True):
    env_var_dict = env_vars_to_dict(env_vars=env_vars, prefix=prefix)
    mlflow_log_params(env_var_dict, enable_mlflow)


def log_metric_env_vars(env_vars=[], prefix="", enable_mlflow=True):
    env_var_dict = env_vars_to_dict(env_vars=env_vars, prefix=prefix)
    metric_env_var_dict = {ev: float(value) for (ev, value) in env_var_dict.items()}
    mlflow_log_metrics(metric_env_var_dict, enable_mlflow)


class MLflowEnvVarsLoggerHook:
    """Logs environment variables to MLflow"""

    def __init__(
        self,
        param_env_vars: List[str] = None,
        metric_env_vars: List[str] = None,
        prefix: str = None,
        enable_mlflow: bool = True,
    ):
        """
        Args:
            param_env_vars: Environment variables to log to MLflow as parameters
            metric_env_vars: Environment variables to log to MLflow as metrics
            prefix: Prefix to add to each name of MLflow parameters and metrics ("env.." in default)
            enable_mlflow: Enable logging to MLflow.
        """
        self.enable_mlflow = find_spec("mlflow") and enable_mlflow
        self.param_env_vars = param_env_vars or []
        self.metric_env_vars = metric_env_vars or []
        self.prefix = prefix or "env.."

    def _log_env_vars(self):
        log_param_env_vars(
            env_vars=self.param_env_vars,
            prefix=self.prefix,
            enable_mlflow=self.enable_mlflow,
        )
        log_metric_env_vars(
            env_vars=self.metric_env_vars,
            prefix=self.prefix,
            enable_mlflow=self.enable_mlflow,
        )

    @hook_impl
    def before_pipeline_run(self):
        self._log_env_vars()

    @hook_impl
    def after_pipeline_run(self):
        self._log_env_vars()
