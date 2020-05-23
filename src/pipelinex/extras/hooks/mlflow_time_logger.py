from importlib.util import find_spec
import time
from logging import getLogger
from pprint import pformat

log = getLogger(__name__)


try:
    from kedro.framework.hooks import hook_impl
except ModuleNotFoundError:

    def hook_impl(func):
        return func


def _get_node_name(node):
    return "_time_ {} -- {}".format(node._func_name, " - ".join(node.outputs))


class MLflowTimeLoggerHook:
    _time_begin_dict = {}
    _time_end_dict = {}
    _time_dict = {}

    def __init__(
        self, enable_mlflow=True, node_name_func=_get_node_name,
    ):
        self.enable_mlflow = find_spec("mlflow") and enable_mlflow
        self._node_name_func = node_name_func

    @hook_impl
    def before_node_run(self, node, catalog, inputs):
        node_name = self._node_name_func(node)
        time_begin_dict = {node_name: time.time()}
        self._time_begin_dict.update(time_begin_dict)

    @hook_impl
    def after_node_run(self, node, catalog, inputs, outputs):
        node_name = self._node_name_func(node)
        time_end_dict = {node_name: time.time()}
        self._time_end_dict.update(time_end_dict)

        time_dict = {
            node_name: (
                self._time_end_dict.get(node_name)
                - self._time_begin_dict.get(node_name)
            )
        }

        log.info("Time duration: {}".format(time_dict))

        if self.enable_mlflow:

            from mlflow import log_metrics

            log_metrics(time_dict)

        self._time_dict.update(time_dict)

    @hook_impl
    def after_pipeline_run(self, run_params, pipeline, catalog):
        log.info("Time duration: \n{}".format(pformat(self._time_dict)))
