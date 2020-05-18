import kedro

try:
    from kedro.framework.context import *  # NOQA
except (ImportError, ModuleNotFoundError):
    # if kedro.__version__ <= "0.15.9"
    from kedro.context import *  # NOQA
