import logging
from functools import wraps
from importlib.util import find_spec
from typing import Callable

log = logging.getLogger(__name__)


if find_spec("memory_profiler"):
    from memory_profiler import memory_usage
else:
    raise ImportError("Install memory_profiler.")


def _func_full_name(func: Callable):
    return getattr(func, "__qualname__", repr(func))


def mem_profile(func: Callable) -> Callable:
    """A function decorator which profiles the memory used when executing the
    function. The logged memory is collected by using the memory_profiler
    python module and includes memory used by children processes. The usage
    is collected by taking memory snapshots every 100ms. This decorator will
    only work with functions taking at least 0.5s to execute due to a bug in
    the memory_profiler python module. For more information about the bug,
    please see https://github.com/pythonprofilers/memory_profiler/issues/216

    Args:
        func: The function to be profiled.

    Returns:
        A wrapped function, which will execute the provided function and log
        its max memory usage upon completion.

    """

    @wraps(func)
    def with_memory(*args, **kwargs):
        log = logging.getLogger(__name__)
        mem_usage, result = memory_usage(
            (func, args, kwargs),
            interval=0.1,
            timeout=1,
            max_usage=True,
            retval=True,
            include_children=True,
        )
        # memory_profiler < 0.56.0 returns list instead of float
        mem_usage = mem_usage[0] if isinstance(mem_usage, (list, tuple)) else mem_usage
        log.info(
            "Running %r consumed %2.2fMiB memory at peak time",
            _func_full_name(func),
            mem_usage,
        )
        return result

    return with_memory
