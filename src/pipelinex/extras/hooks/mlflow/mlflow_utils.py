from pathlib import Path
from logging import getLogger

log = getLogger(__name__)

try:
    from kedro.framework.hooks import hook_impl
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
                if Path(path).is_file():
                    log_artifact(path, artifact_path)
                    log.info("File at '{}' was logged by MLflow.".format(path))
                elif Path(path).is_dir():
                    log_artifacts(path, artifact_path)
                    log.info("Directory at '{}' was logged by MLflow.".format(path))
                else:
                    log.warning("'{}' was not found.".format(path))
        except Exception:
            log.warning(
                "{} failed to be logged by MLflow.".format(paths), exc_info=True
            )


def mlflow_log_metrics(metrics, step=None, enable_mlflow=True):
    log.info("{}".format(metrics))

    if enable_mlflow:
        try:
            metrics = {k.replace(":", ".."): float(v) for (k, v) in metrics.items()}

            from mlflow import log_metrics

            log_metrics(metrics, step)
        except Exception:
            log.warning(
                "{} failed to be logged by MLflow.".format(metrics), exc_info=True
            )


def mlflow_log_params(params, enable_mlflow=True):
    log.info("{}".format(params))

    if enable_mlflow:
        try:
            params = {
                k.replace(":", ".."): ("{}".format(v)[:250] if v else "")
                for (k, v) in params.items()
            }

            from mlflow import log_params

            log_params(params)
        except Exception:
            log.warning(
                "{} failed to be logged by MLflow.".format(params), exc_info=True
            )


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
