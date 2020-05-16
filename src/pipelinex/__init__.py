__version__ = "0.2.7"

from importlib.util import find_spec
from .configure import *  # NOQA
from .utils import *  # NOQA
from .hatch_dict.hatch_dict import *  # NOQA
from .extras.decorators.decorators import *  # NOQA
from .extras.ops.argparse_ops import *  # NOQA

if find_spec("kedro"):
    from .pipeline.pipeline import *  # NOQA
    from .pipeline.sub_pipeline import *  # NOQA
    from .context.catalog_sugar_context import *  # NOQA
    from .context.flexible_context import *  # NOQA
    from .context.only_missing_string_runner_context import *  # NOQA
    from .context.pipelines_in_parameters_context import *  # NOQA
    from .context.mlflow_context import *  # NOQA

if find_spec("pandas"):
    from .extras.datasets.pandas.csv_local import *  # NOQA
    from .extras.datasets.pandas.efficient_csv_local import *  # NOQA
    from .extras.datasets.pandas.pandas_cat_matrix import *  # NOQA
    from .extras.datasets.pandas.pandas_describe import *  # NOQA
    from .extras.datasets.pandas.histgram import *  # NOQA

if find_spec("pandas_profiling"):
    from .extras.datasets.pandas_profiling.pandas_profiling import *  # NOQA

if find_spec("PIL"):
    from .extras.datasets.pillow.images import *  # NOQA

if find_spec("seaborn"):
    from .extras.datasets.seaborn.seaborn_pairplot import *  # NOQA

if find_spec("torchvision"):
    from .extras.datasets.torchvision.iterable_images import *  # NOQA

if find_spec("cv2"):
    from .extras.datasets.opencv.images import *  # NOQA

if find_spec("requests"):
    from .extras.datasets.requests.api_dataset import *  # NOQA

if find_spec("httpx"):
    from .extras.datasets.httpx.async_api_dataset import *  # NOQA

if find_spec("numpy"):
    from .extras.ops.numpy_ops import *  # NOQA

if find_spec("pandas"):
    from .extras.ops.pandas_ops import *  # NOQA
    from .extras.decorators.pandas_decorators import *  # NOQA

if find_spec("torch"):
    from .extras.ops.pytorch_ops import *  # NOQA

if find_spec("ignite"):
    from .extras.ops.ignite.declaratives.declarative_trainer import *  # NOQA

if find_spec("shap"):
    from .extras.ops.shap_ops import *  # NOQA

if find_spec("sklearn"):
    from .extras.ops.sklearn_ops import *  # NOQA

if find_spec("allennlp"):
    from .extras.ops.allennlp_ops import *  # NOQA

if find_spec("cv2"):
    from .extras.ops.opencv_ops import *  # NOQA

if find_spec("skimage"):
    from .extras.ops.skimage_ops import *  # NOQA

if find_spec("memory_profiler"):
    from .extras.decorators.memory_profiler import *  # NOQA

if find_spec("pynvml") or find_spec("py3nvml"):
    from .extras.decorators.nvml_profiler import *  # NOQA
