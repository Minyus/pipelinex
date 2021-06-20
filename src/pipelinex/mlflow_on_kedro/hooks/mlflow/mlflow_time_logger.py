import json
import re
import tempfile
import time
from importlib.util import find_spec
from logging import getLogger
from pathlib import Path
from pprint import pformat
from typing import Any, Callable, Dict  # NOQA

from kedro.pipeline.node import Node  # NOQA

from .mlflow_utils import hook_impl, mlflow_log_artifacts, mlflow_log_metrics

log = getLogger(__name__)


def _get_task_name(node: Node) -> str:
    func_name = (
        node._func_name.replace("<", "")
        .replace(">", "")
        .split(" ")[0]
        .split("(")[0]
        .split("=")[0]
        .split(".")[-1]
    )
    task_name = "{} -- {}".format(func_name[:20], " - ".join(node.outputs))
    task_name = re.sub(r"[^A-Za-z0-9\_\-\.\ \/]", " ", task_name)
    return task_name[:50]


def dump_dict(filepath: str, d: dict):
    with open(filepath, "w") as outfile:
        json.dump(d, outfile)


def load_dict(filepath: str):
    with open(filepath, "r") as outfile:
        d = json.load(outfile)
    return d


class MLflowTimeLoggerHook:
    """
    Logs duration time to run each node (task) to MLflow.
    Optionally, the execution logs can be visualized as a Gantt chart by
    `plotly.figure_factory.create_gantt`
    (https://plotly.github.io/plotly.py-docs/generated/plotly.figure_factory.create_gantt.html)
    if `plotly` is installed.
    """

    def __init__(
        self,
        gantt_filepath: str = None,
        gantt_params: Dict[str, Any] = {},
        metric_name_prefix: str = "_time_to_run ",
        task_name_func: Callable[[Node], str] = _get_task_name,
        time_log_filepath: str = None,
        enable_plotly: bool = True,
        enable_mlflow: bool = True,
    ):
        """
        Args:
            gantt_filepath: File path to save the generated gantt chart.
            gantt_params: Args fed to:
                https://plotly.github.io/plotly.py-docs/generated/plotly.figure_factory.create_gantt.html
            metric_name_prefix: Prefix for the metric names. The metric names are
                `metric_name_prefix` concatenated with the string returned by `task_name_func`.
            task_name_func: Callable to return the task name using ``kedro.pipeline.node.Node``
                object.
            time_log_filepath: File path to save the time log in JSON format.
            enable_plotly: Enable visualization of logged time as a gantt chart.
            enable_mlflow: Enable logging to MLflow.
        """
        self.enable_mlflow = find_spec("mlflow") and enable_mlflow
        self.enable_plotly = find_spec("plotly") and enable_plotly
        self.gantt_filepath = gantt_filepath
        self.gantt_params = gantt_params
        self.metric_name_prefix = metric_name_prefix
        self.task_name_func = task_name_func
        self.time_log_filepath = time_log_filepath or (
            tempfile.gettempdir() + "/_time_log.json"
        )
        Path(self.time_log_filepath).parent.mkdir(parents=True, exist_ok=True)
        dump_dict(
            self.time_log_filepath, {"time_begin": {}, "time_end": {}, "time": {}}
        )

        self._time_begin_dict = {}
        self._time_end_dict = {}
        self._time_dict = {}

    def update_time_dict(self, key: str, d: dict):
        dumping_dict = load_dict(self.time_log_filepath)
        dumping_dict[key].update(d)
        dump_dict(self.time_log_filepath, dumping_dict)

    def load_time_dict(self, key: str):
        return load_dict(self.time_log_filepath).get(key)

    @hook_impl
    def before_node_run(self, node, catalog, inputs):
        task_name = self.task_name_func(node)
        time_begin_dict = {task_name: time.time()}
        self._time_begin_dict.update(time_begin_dict)
        self.update_time_dict("time_begin", time_begin_dict)

    @hook_impl
    def after_node_run(self, node, catalog, inputs, outputs):
        task_name = self.task_name_func(node)

        time_end_dict = {task_name: time.time()}
        self._time_end_dict.update(time_end_dict)
        self.update_time_dict("time_end", time_end_dict)

        time_dict = {
            task_name: (
                self._time_end_dict.get(task_name)
                - self._time_begin_dict.get(task_name)
            )
        }

        log.info("Time duration: {}".format(time_dict))

        self._time_dict.update(time_dict)

        self.update_time_dict("time", time_dict)

        metric_time_dict = {
            (self.metric_name_prefix + k): v for (k, v) in time_dict.items()
        }
        mlflow_log_metrics(metric_time_dict, enable_mlflow=self.enable_mlflow)

    @hook_impl
    def after_pipeline_run(self, run_params, pipeline, catalog):

        self._time_begin_dict = self._time_begin_dict or self.load_time_dict(
            "time_begin"
        )
        self._time_end_dict = self._time_end_dict or self.load_time_dict("time_end")
        self._time_dict = self._time_dict or self.load_time_dict("time")

        log.info("Time duration: \n{}".format(pformat(self._time_dict)))

        if self.enable_plotly:
            if not (self._time_begin_dict and self._time_end_dict):
                log.warning(
                    "Time log dicts are not found. Skipping generating the Gantt Chart."
                )
                return

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
