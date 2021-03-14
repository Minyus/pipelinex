from pathlib import Path
from logging import getLogger

from pipelinex.hatch_dict.hatch_dict import dot_flatten

log = getLogger(__name__)

try:
    from kedro.framework.hooks import hook_impl  # NOQA
except ModuleNotFoundError:

    def hook_impl(func):
        return func


def mlflow_log_artifacts(paths, artifact_path=None, enable_mlflow=True):
    if enable_mlflow:
        paths = paths or []
        if not isinstance(paths, (list, tuple)):
            paths = [paths]

        try:
            from mlflow import log_artifact, log_artifacts

            for path in paths:
                resolved_path = Path(path).resolve()
                if Path(path).is_file():
                    log_artifact(path, artifact_path)
                    log.info("File at '{}' was logged by MLflow.".format(resolved_path))
                elif Path(path).is_dir():
                    log_artifacts(path, artifact_path)
                    log.info(
                        "Directory at '{}' was logged by MLflow.".format(resolved_path)
                    )
                else:
                    log.warning("Path '{}' does not exist.".format(resolved_path))
        except Exception:
            log.warning(
                "{} failed to be logged by MLflow.".format(paths), exc_info=True
            )


def mlflow_log_metrics(metrics, step=None, enable_mlflow=True):
    assert isinstance(metrics, dict)
    log.info("{}".format(metrics))

    if enable_mlflow:
        try:
            metrics = {
                k.replace(":", ".."): float(v)
                for (k, v) in metrics.items()
                if isinstance(v, (float, int))
            }

            from mlflow import log_metrics

            log_metrics(metrics, step)
        except Exception:
            log.warning(
                "{} failed to be logged by MLflow.".format(metrics), exc_info=True
            )


def mlflow_log_params(params, enable_mlflow=True):
    assert isinstance(params, dict)
    log.info("{}".format(params))

    if enable_mlflow:
        try:
            params = {
                k.replace(":", ".."): ("{}".format(v)[:250])
                for (k, v) in params.items()
                if isinstance(v, str) or v
            }

            from mlflow import log_params

            log_params(params)
        except Exception:
            log.warning(
                "{} failed to be logged by MLflow.".format(params), exc_info=True
            )


def mlflow_log_values(d, enable_mlflow=True):
    assert isinstance(d, dict)
    log.info("{}".format(d))

    d = dot_flatten(d)

    if enable_mlflow:

        metrics = {k: v for (k, v) in d.items() if isinstance(v, (float, int))}
        mlflow_log_metrics(metrics)

        params = {
            k: v for (k, v) in d.items() if isinstance(v, (str, list, tuple, set))
        }
        mlflow_log_params(params)


def mlflow_start_run(
    uri, experiment_name, artifact_location, run_name=None, enable_mlflow=True
):
    if enable_mlflow:

        from mlflow import (
            create_experiment,
            get_experiment_by_name,
            start_run,
            set_tracking_uri,
        )
        from mlflow.exceptions import MlflowException

        if uri:
            set_tracking_uri(uri)

        if experiment_name:
            try:
                experiment_id = create_experiment(
                    experiment_name,
                    artifact_location=artifact_location,
                )
            except MlflowException:
                experiment_id = get_experiment_by_name(experiment_name).experiment_id
            start_run(experiment_id=experiment_id, run_name=run_name)


def mlflow_end_run(enable_mlflow=True):
    if enable_mlflow:
        try:
            from mlflow import end_run

            end_run()
        except Exception:
            log.warning("Failed to end MLflow run.", exc_info=True)
