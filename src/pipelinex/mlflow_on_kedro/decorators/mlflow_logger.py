import logging
import time
from functools import wraps
from typing import Callable

log = logging.getLogger(__name__)


def _func_full_name(func: Callable):
    return getattr(func, "__qualname__", repr(func))


def _func_name(func: Callable):
    return getattr(func, "__name__", "") or (
        getattr(getattr(func, "__class__", ""), "__name__", "") + ".__call__"
    )


def _human_readable_time(elapsed: float):  # pragma: no cover
    mins, secs = divmod(elapsed, 60)
    hours, mins = divmod(mins, 60)

    if hours > 0:
        message = "%dh%02dm%02ds" % (hours, mins, secs)
    elif mins > 0:
        message = "%dm%02ds" % (mins, secs)
    elif secs >= 1:
        message = "%.2fs" % secs
    else:
        message = "%.0fms" % (secs * 1000.0)

    return message


def mlflow_log_time(func: Callable) -> Callable:
    """A function decorator which logs the time taken for executing a function.

    Args:
        func: The function to be logged.

    Returns:
        A wrapped function, which will execute the provided function and log
        the running time.

    """

    @wraps(func)
    def with_time(*args, **kwargs):
        log = logging.getLogger(__name__)
        t_start = time.time()
        result = func(*args, **kwargs)
        t_end = time.time()
        elapsed = t_end - t_start

        log.info(
            "Running %r took %s [%.3fs]",
            _func_full_name(func),
            _human_readable_time(elapsed),
            elapsed,
        )

        try:
            from mlflow import log_metric

            log_metric("__time_for_" + _func_name(func), elapsed)
        except Exception:
            log.warning("Exception from MLflow: ", exc_info=True)

        return result

    return with_time
