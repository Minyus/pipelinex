__version__ = "0.8.0"

from importlib.util import find_spec
from warnings import warn

from .extras.decorators.decorators import *  # NOQA
from .extras.ops.argparse_ops import *  # NOQA
from .hatch_dict.hatch_dict import *  # NOQA
from .mlflow_on_kedro.decorators.mlflow_logger import *  # NOQA
from .utils import *  # NOQA


def _warn_optional_import_failure(group: str, exc: Exception) -> None:
    warn("Skipping optional pipelinex imports for {}: {}".format(group, exc))


if find_spec("kedro"):
    try:
        from .extras.hooks.add_catalog_dict import *  # NOQA
        from .flex_kedro.pipeline.pipeline import *  # NOQA
        from .flex_kedro.pipeline.sub_pipeline import *  # NOQA
        from .mlflow_on_kedro.datasets.mlflow.mlflow_dataset import *  # NOQA
        from .mlflow_on_kedro.hooks.mlflow.mlflow_artifacts_logger import *  # NOQA
        from .mlflow_on_kedro.hooks.mlflow.mlflow_basic_logger import *  # NOQA
        from .mlflow_on_kedro.hooks.mlflow.mlflow_catalog_logger import *  # NOQA
        from .mlflow_on_kedro.hooks.mlflow.mlflow_datasets_logger import *  # NOQA
        from .mlflow_on_kedro.hooks.mlflow.mlflow_env_vars_logger import *  # NOQA
        from .mlflow_on_kedro.hooks.mlflow.mlflow_time_logger import *  # NOQA

        import kedro

        kedro_major, kedro_minor = kedro.__version__.split(".", 2)[:2]
        if int(kedro_major) == 0 and int(kedro_minor) < 18:
            from .extras.hooks.add_transformers import *  # NOQA
            from .flex_kedro.configure import *  # NOQA
            from .flex_kedro.context.flexible_context import *  # NOQA
            from .mlflow_on_kedro.transformers.mlflow.mlflow_io_time_logger import *  # NOQA
    except (ImportError, AttributeError) as exc:
        _warn_optional_import_failure("kedro/mlflow", exc)

if find_spec("pandas"):
    try:
        from .extras.datasets.pandas.csv_local import *  # NOQA
        from .extras.datasets.pandas.efficient_csv_local import *  # NOQA
        from .extras.datasets.pandas.fixed_width_csv_dataset import *  # NOQA
        from .extras.datasets.pandas.pandas_cat_matrix import *  # NOQA
        from .extras.datasets.pandas.pandas_describe import *  # NOQA

        if find_spec("matplotlib"):
            from .extras.datasets.pandas.histgram import *  # NOQA

            if find_spec("seaborn"):
                from .extras.datasets.seaborn.seaborn_pairplot import *  # NOQA
    except (ImportError, AttributeError) as exc:
        _warn_optional_import_failure("pandas datasets", exc)

if find_spec("pandas_profiling"):
    try:
        from .extras.datasets.pandas_profiling.pandas_profiling import *  # NOQA
    except (ImportError, AttributeError) as exc:
        _warn_optional_import_failure("pandas_profiling", exc)

if find_spec("PIL"):
    try:
        from .extras.datasets.pillow.images_dataset import *  # NOQA
    except (ImportError, AttributeError) as exc:
        _warn_optional_import_failure("pillow datasets", exc)


if find_spec("torchvision"):
    try:
        from .extras.datasets.torchvision.iterable_images_dataset import *  # NOQA
    except (ImportError, AttributeError) as exc:
        _warn_optional_import_failure("torchvision datasets", exc)

if find_spec("cv2"):
    try:
        from .extras.datasets.opencv.images_dataset import *  # NOQA
    except (ImportError, AttributeError) as exc:
        _warn_optional_import_failure("opencv datasets", exc)

if find_spec("requests"):
    try:
        from .extras.datasets.requests.api_dataset import *  # NOQA

        if find_spec("httpx"):
            from .extras.datasets.httpx.async_api_dataset import *  # NOQA
    except (ImportError, AttributeError) as exc:
        _warn_optional_import_failure("requests/httpx datasets", exc)

if find_spec("numpy"):
    try:
        from .extras.ops.numpy_ops import *  # NOQA
    except (ImportError, AttributeError) as exc:
        _warn_optional_import_failure("numpy ops", exc)

if find_spec("pandas"):
    try:
        from .extras.decorators.pandas_decorators import *  # NOQA
        from .extras.ops.pandas_ops import *  # NOQA
    except (ImportError, AttributeError) as exc:
        _warn_optional_import_failure("pandas ops", exc)

if find_spec("torch"):
    try:
        from .extras.ops.pytorch_ops import *  # NOQA
    except (ImportError, AttributeError) as exc:
        _warn_optional_import_failure("torch ops", exc)

if find_spec("ignite"):
    try:
        from .extras.ops.ignite.declaratives.declarative_trainer import *  # NOQA
    except (ImportError, AttributeError) as exc:
        _warn_optional_import_failure("ignite ops", exc)

if find_spec("shap"):
    try:
        from .extras.ops.shap_ops import *  # NOQA
    except (ImportError, AttributeError) as exc:
        _warn_optional_import_failure("shap ops", exc)

if find_spec("sklearn"):
    try:
        from .extras.ops.sklearn_ops import *  # NOQA
    except (ImportError, AttributeError) as exc:
        _warn_optional_import_failure("sklearn ops", exc)

if find_spec("allennlp"):
    try:
        from .extras.ops.allennlp_ops import *  # NOQA
    except (ImportError, AttributeError) as exc:
        _warn_optional_import_failure("allennlp ops", exc)

if find_spec("cv2"):
    try:
        from .extras.ops.opencv_ops import *  # NOQA
    except (ImportError, AttributeError) as exc:
        _warn_optional_import_failure("opencv ops", exc)

if find_spec("skimage"):
    try:
        from .extras.ops.skimage_ops import *  # NOQA
    except Exception as exc:
        _warn_optional_import_failure("skimage ops", exc)

if find_spec("memory_profiler"):
    try:
        from .extras.decorators.memory_profiler import *  # NOQA
    except Exception as exc:
        _warn_optional_import_failure("memory_profiler decorators", exc)

if find_spec("pynvml") or find_spec("py3nvml"):
    try:
        from .extras.decorators.nvml_profiler import *  # NOQA
    except Exception as exc:
        _warn_optional_import_failure("nvml decorators", exc)
