from importlib.util import find_spec
import time
from logging import getLogger
from pprint import pformat
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict  # NOQA

from kedro.pipeline.node import Node  # NOQA

from .mlflow_utils import hook_impl, mlflow_log_metrics, mlflow_log_artifacts


log = getLogger(__name__)


def _get_task_name(node: Node) -> str:
    func_name = (
        node._func_name.replace("<", "")
        .replace(">", "")
        .split(" ")[0]
        .split(".")[-1][:250]
    )
    return "{} -- {}".format(func_name, " - ".join(node.outputs))


class MLflowTimeLoggerHook:
    """
    Logs duration time to run each node (task) to MLflow.
    Optionally, the execution logs can be visualized as a Gantt chart by
    `plotly.figure_factory.create_gantt`
    (https://plotly.github.io/plotly.py-docs/generated/plotly.figure_factory.create_gantt.html)
    if `plotly` is installed.
    """

    _time_begin_dict = {}
    _time_end_dict = {}
    _time_dict = {}

    def __init__(
        self,
        enable_mlflow: bool = True,
        enable_plotly: bool = True,
        gantt_filepath: str = None,
        gantt_params: Dict[str, Any] = {},
        metric_name_prefix: str = "_time_to_run ",
        task_name_func: Callable[[Node], str] = _get_task_name,
    ):
        """
        Args:
            enable_mlflow: Enable logging to MLflow.
            enable_plotly: Enable visualization of logged time as a gantt chart.
            gantt_filepath: File path to save the generated gantt chart.
            gantt_params: Args fed to:
                https://plotly.github.io/plotly.py-docs/generated/plotly.figure_factory.create_gantt.html
            metric_name_prefix: Prefix for the metric names. The metric names are
                `metric_name_prefix` concatenated with the string returned by `task_name_func`.
            task_name_func: Callable to return the task name using ``kedro.pipeline.node.Node``
                object.
        """
        self.enable_mlflow = find_spec("mlflow") and enable_mlflow
        self.enable_plotly = find_spec("plotly") and enable_plotly
        self.gantt_filepath = gantt_filepath
        self.gantt_params = gantt_params
        self.metric_name_prefix = metric_name_prefix
        self.task_name_func = task_name_func

    @hook_impl
    def before_node_run(self, node, catalog, inputs):
        task_name = self.task_name_func(node)
        time_begin_dict = {task_name: time.time()}
        self._time_begin_dict.update(time_begin_dict)

    @hook_impl
    def after_node_run(self, node, catalog, inputs, outputs):
        task_name = self.task_name_func(node)

        time_end_dict = {task_name: time.time()}
        self._time_end_dict.update(time_end_dict)

        time_dict = {
            task_name: (
                self._time_end_dict.get(task_name)
                - self._time_begin_dict.get(task_name)
            )
        }

        log.info("Time duration: {}".format(time_dict))

        self._time_dict.update(time_dict)

        metric_time_dict = {
            (self.metric_name_prefix + k): v for (k, v) in time_dict.items()
        }
        mlflow_log_metrics(metric_time_dict, enable_mlflow=self.enable_mlflow)

    @hook_impl
    def after_pipeline_run(self, run_params, pipeline, catalog):
        log.info("Time duration: \n{}".format(pformat(self._time_dict)))

        if self.enable_plotly and self._time_begin_dict:

            tasks_reversed = list(self._time_begin_dict.keys())[::-1]

            from plotly.figure_factory import create_gantt

            df = [
                dict(
                    Task=t,
                    Start=self._time_begin_dict.get(t) * 1000,
                    Finish=self._time_end_dict.get(t) * 1000,
                )
                for t in tasks_reversed
            ]

            fig = create_gantt(df, **self.gantt_params)

            fp = self.gantt_filepath or (tempfile.gettempdir() + "/_gantt.html")
            Path(fp).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(fp)

            mlflow_log_artifacts(fp, enable_mlflow=self.enable_mlflow)
