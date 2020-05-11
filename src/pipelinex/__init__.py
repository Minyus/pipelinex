__version__ = "0.2.3"

from importlib.util import find_spec
from .utils import *  # NOQA
from .hatch_dict.hatch_dict import *  # NOQA
from .decorators.decorators import *  # NOQA
from .ops.argparse_ops import *  # NOQA

if find_spec("kedro"):
    from .pipeline.pipeline import *  # NOQA
    from .pipeline.sub_pipeline import *  # NOQA
    from .context.catalog_sugar_context import *  # NOQA
    from .context.flexible_context import *  # NOQA
    from .context.only_missing_string_runner_context import *  # NOQA
    from .context.pipelines_in_parameters_context import *  # NOQA
    from .context.mlflow_context import *  # NOQA

if find_spec("pandas"):
    from .io.pandas.csv_local import *  # NOQA
    from .io.pandas.efficient_csv_local import *  # NOQA
    from .io.pandas.pandas_cat_matrix import *  # NOQA
    from .io.pandas.pandas_describe import *  # NOQA
    from .io.pandas.histgram import *  # NOQA

if find_spec("pandas_profiling"):
    from .io.pandas_profiling.pandas_profiling import *  # NOQA

if find_spec("PIL"):
    from .io.pillow.images import *  # NOQA

if find_spec("seaborn"):
    from .io.seaborn.seaborn_pairplot import *  # NOQA

if find_spec("torchvision"):
    from .io.torchvision.iterable_images import *  # NOQA

if find_spec("cv2"):
    from .io.opencv.images import *  # NOQA

if find_spec("requests"):
    from .io.requests.api_dataset import *  # NOQA

if find_spec("httpx"):
    from .io.httpx.async_api_dataset import *  # NOQA

if find_spec("pandas"):
    from .ops.pandas_ops import *  # NOQA
    from .decorators.pandas_decorators import *  # NOQA

if find_spec("torch"):
    from .ops.pytorch_ops import *  # NOQA

if find_spec("ignite"):
    from .ops.ignite.declaratives.declarative_trainer import *  # NOQA

if find_spec("shap"):
    from .ops.shap_ops import *  # NOQA

if find_spec("sklearn"):
    from .ops.sklearn_ops import *  # NOQA

if find_spec("allennlp"):
    from .ops.allennlp_ops import *  # NOQA

if find_spec("cv2"):
    from .ops.opencv_ops import *  # NOQA

if find_spec("skimage"):
    from .ops.skimage_ops import *  # NOQA

if find_spec("memory_profiler"):
    from .decorators.memory_profiler import *  # NOQA

if find_spec("pynvml") or find_spec("py3nvml"):
    from .decorators.nvml_profiler import *  # NOQA
