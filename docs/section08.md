## Kedro-Extras: Kedro plugin to use various Python packages 

[API document](https://pipelinex.readthedocs.io/en/latest/pipelinex.extras.html)

Kedro-Extras provides Kedro DataSets and decorators not available in [kedro.extras](https://github.com/quantumblacklabs/kedro/tree/master/kedro/extras).

Contributors who are willing to help preparing the test code and send pull request to Kedro following Kedro's [CONTRIBUTING.md](https://github.com/quantumblacklabs/kedro/blob/master/CONTRIBUTING.md#contribute-a-new-feature) are welcomed.

### Additional Kedro datasets (data interface sets)
  
[pipelinex.extras.datasets](https://github.com/Minyus/pipelinex/tree/master/src/pipelinex/extras/datasets) provides the following Kedro Datasets (data interface sets) mainly for Computer Vision applications using PyTorch/torchvision, OpenCV, and Scikit-image.

- [pipelinex.ImagesLocalDataSet](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/datasets/pillow/images_dataset.py
)
  - loads/saves multiple numpy arrays (RGB, BGR, or monochrome image) from/to a folder in local storage using `pillow` package, working like ``kedro.extras.datasets.pillow.ImageDataSet`` and
  ``kedro.io.PartitionedDataSet`` with conversion between numpy arrays and Pillow images.
  - an example project is at [pipelinex_image_processing](https://github.com/Minyus/pipelinex_image_processing)
- [pipelinex.APIDataSet](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/datasets/requests/api_dataset.py)
  - modified version of [kedro.extras.APIDataSet](https://github.com/quantumblacklabs/kedro/blob/master/kedro/extras/datasets/api/api_dataset.py) with more flexible options including downloading multiple contents (such as images and json) by HTTP requests to multiple URLs using `requests` package
  - an example project is at [pipelinex_image_processing](https://github.com/Minyus/pipelinex_image_processing)
- [pipelinex.AsyncAPIDataSet](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/datasets/httpx/async_api_dataset.py)
  - downloads multiple contents (such as images and json) by asynchronous HTTP requests to multiple URLs using `httpx` package
  - an example project is at [pipelinex_image_processing](https://github.com/Minyus/pipelinex_image_processing)

- [pipelinex.IterableImagesDataSet](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/datasets/torchvision/iterable_images_dataset.py)
  - wrapper of [`torchvision.datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) that loads images in a folder as an iterable data loader to use with PyTorch.

- [pipelinex.PandasProfilingDataSet](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/datasets/pandas_profiling/pandas_profiling.py)
  - generates a pandas dataframe summary report using [pandas-profiling](https://github.com/pandas-profiling/pandas-profiling)

- [more data interface sets for pandas dataframe summarization/visualization provided by PipelineX](https://github.com/Minyus/pipelinex/tree/master/src/pipelinex/extras/datasets)

### Additional function decorators for benchmarking

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Minyus/pipelinex/blob/master/notebooks/decorators_demo.ipynb)

[pipelinex.extras.decorators](https://github.com/Minyus/pipelinex/tree/master/src/pipelinex/extras/decorators) provides Python decorators for benchmarking.

- [log_time](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/decorators/decorators.py)
  - logs the duration time of a function (difference of timestamp before and after running the function).
  - Slightly modified version of Kedro's [log_time](https://github.com/quantumblacklabs/kedro/blob/develop/kedro/pipeline/decorators.py#L59)

- [mem_profile](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/decorators/memory_profiler.py)
  - logs the peak memory usage during running the function.
  - `memory_profiler` needs to be installed.
  - Slightly modified version of Kedro's [mem_profile](https://github.com/quantumblacklabs/kedro/blob/develop/kedro/extras/decorators/memory_profiler.py#L48)

- [nvml_profile](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/decorators/nvml_profiler.py)
  - logs the difference of NVIDIA GPU usage before and after running the function.
  - `pynvml` or `py3nvml` needs to be installed.

```python
from pipelinex import log_time
from pipelinex import mem_profile  # Need to install memory_profiler for memory profiling
from pipelinex import nvml_profile  # Need to install pynvml for NVIDIA GPU profiling
from time import sleep
import logging

logging.basicConfig(level=logging.INFO)

@nvml_profile
@mem_profile
@log_time
def foo_func(i=1):
    sleep(0.5)  # Needed to avoid the bug reported at https://github.com/pythonprofilers/memory_profiler/issues/216
    return "a" * i

output = foo_func(100_000_000)
```

```
INFO:pipelinex.decorators.decorators:Running 'foo_func' took 549ms [0.549s]
INFO:pipelinex.decorators.memory_profiler:Running 'foo_func' consumed 579.02MiB memory at peak time
INFO:pipelinex.decorators.nvml_profiler:Ran: 'foo_func', NVML returned: {'_Driver_Version': '418.67', '_NVML_Version': '10.418.67', 'Device_Count': 1, 'Devices': [{'_Name': 'Tesla P100-PCIE-16GB', 'Total_Memory': 17071734784, 'Free_Memory': 17071669248, 'Used_Memory': 65536, 'GPU_Utilization_Rate': 0, 'Memory_Utilization_Rate': 0}]}, Used memory diff: [0]
```

### Use with PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Minyus/pipelinex/blob/master/notebooks/PyTorch_demo.ipynb)

To develop a simple neural network, it is convenient to use Sequential API
(e.g. `torch.nn.Sequential`, `tf.keras.Sequential`).

- Hardcoded:

```python
from torch.nn import Sequential, Conv2d, ReLU

model = Sequential(
    Conv2d(in_channels=3, out_channels=16, kernel_size=[3, 3]),
    ReLU(),
)

print("### model object by hard-coding ###")
print(model)
```

```
### model object by hard-coding ###
Sequential(
  (0): Conv2d(3, 16, kernel_size=[3, 3], stride=(1, 1))
  (1): ReLU()
)
```

- Using import-less Python object feature:

```python
from pipelinex import HatchDict
import yaml
from pprint import pprint  # pretty-print for clearer look

# Read parameters dict from a YAML file in actual use
params_yaml="""
model:
  =: torch.nn.Sequential
  _:
    - {=: torch.nn.Conv2d, in_channels: 3, out_channels: 16, kernel_size: [3, 3]}
    - {=: torch.nn.ReLU, _: }
"""
parameters = yaml.safe_load(params_yaml)

model_dict = parameters.get("model")

print("### Before ###")
pprint(model_dict)

model = HatchDict(parameters).get("model")

print("\n### After ###")
print(model)
```

```
### Before ###
{'=': 'torch.nn.Sequential',
 '_': [{'=': 'torch.nn.Conv2d',
        'in_channels': 3,
        'kernel_size': [3, 3],
        'out_channels': 16},
       {'=': 'torch.nn.ReLU', '_': None}]}

### After ###
Sequential(
  (0): Conv2d(3, 16, kernel_size=[3, 3], stride=(1, 1))
  (1): ReLU()
)
```

In addition to `Sequential`, TensorFLow/Keras provides modules to merge branches such as
`tf.keras.layers.Concatenate`, but PyTorch provides only functional interface such as `torch.cat`.

PipelineX provides modules to merge branches such as `ModuleConcat`, `ModuleSum`, and `ModuleAvg`.

- Hardcoded:

```python
from torch.nn import Sequential, Conv2d, AvgPool2d, ReLU
from pipelinex import ModuleConcat

model = Sequential(
    ModuleConcat(
        Conv2d(in_channels=3, out_channels=16, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),
        AvgPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),
    ),
    ReLU(),
)
print("### model object by hard-coding ###")
print(model)
```

```
### model object by hard-coding ###
Sequential(
  (0): ModuleConcat(
    (0): Conv2d(3, 16, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1])
    (1): AvgPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1])
  )
  (1): ReLU()
)
```

- Using import-less Python object feature:

```python
from pipelinex import HatchDict
import yaml
from pprint import pprint  # pretty-print for clearer look

# Read parameters dict from a YAML file in actual use
params_yaml="""
model:
  =: torch.nn.Sequential
  _:
    - =: pipelinex.ModuleConcat
      _:
        - {=: torch.nn.Conv2d, in_channels: 3, out_channels: 16, kernel_size: [3, 3], stride: [2, 2], padding: [1, 1]}
        - {=: torch.nn.AvgPool2d, kernel_size: [3, 3], stride: [2, 2], padding: [1, 1]}
    - {=: torch.nn.ReLU, _: }
"""
parameters = yaml.safe_load(params_yaml)

model_dict = parameters.get("model")

print("### Before ###")
pprint(model_dict)

model = HatchDict(parameters).get("model")

print("\n### After ###")
print(model)
```

```
### Before ###
{'=': 'torch.nn.Sequential',
 '_': [{'=': 'pipelinex.ModuleConcat',
        '_': [{'=': 'torch.nn.Conv2d',
               'in_channels': 3,
               'kernel_size': [3, 3],
               'out_channels': 16,
               'padding': [1, 1],
               'stride': [2, 2]},
              {'=': 'torch.nn.AvgPool2d',
               'kernel_size': [3, 3],
               'padding': [1, 1],
               'stride': [2, 2]}]},
       {'=': 'torch.nn.ReLU', '_': None}]}

### After ###
Sequential(
  (0): ModuleConcat(
    (0): Conv2d(3, 16, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1])
    (1): AvgPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1])
  )
  (1): ReLU()
)
```

### Use with PyTorch Ignite

Wrappers of PyTorch Ignite provides most of features available in Ignite, including integration with MLflow, in an easy declarative way.

In addition, the following optional features are available in PipelineX.

- Use only partial samples in dataset (Useful for quick preliminary check before using the whole dataset)
- Time limit for training (Useful for code-only (Kernel-only) Kaggle competitions with time limit)

Here are the arguments for [`NetworkTrain`](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/ops/ignite/declaratives/declarative_trainer.py):

```
loss_fn (callable): Loss function used to train.
    Accepts an instance of loss functions at https://pytorch.org/docs/stable/nn.html#loss-functions
epochs (int, optional): Max epochs to train
seed (int, optional): Random seed for training.
optimizer (torch.optim, optional): Optimizer used to train.
    Accepts optimizers at https://pytorch.org/docs/stable/optim.html
optimizer_params (dict, optional): Parameters for optimizer.
train_data_loader_params (dict, optional): Parameters for data loader for training.
    Accepts args at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
val_data_loader_params (dict, optional): Parameters for data loader for validation.
    Accepts args at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
evaluation_metrics (dict, optional): Metrics to compute for evaluation.
    Accepts dict of metrics at https://pytorch.org/ignite/metrics.html
evaluate_train_data (str, optional): When to compute evaluation_metrics using training dataset.
    Accepts events at https://pytorch.org/ignite/engine.html#ignite.engine.Events
evaluate_val_data (str, optional): When to compute evaluation_metrics using validation dataset.
    Accepts events at https://pytorch.org/ignite/engine.html#ignite.engine.Events
progress_update (bool, optional): Whether to show progress bar using tqdm package
scheduler (ignite.contrib.handle.param_scheduler.ParamScheduler, optional): Param scheduler.
    Accepts a ParamScheduler at
    https://pytorch.org/ignite/contrib/handlers.html#module-ignite.contrib.handlers.param_scheduler
scheduler_params (dict, optional): Parameters for scheduler
model_checkpoint (ignite.handlers.ModelCheckpoint, optional): Model Checkpoint.
    Accepts a ModelCheckpoint at https://pytorch.org/ignite/handlers.html#ignite.handlers.ModelCheckpoint
model_checkpoint_params (dict, optional): Parameters for ModelCheckpoint at
    https://pytorch.org/ignite/handlers.html#ignite.handlers.ModelCheckpoint
early_stopping_params (dict, optional): Parameters for EarlyStopping at
    https://pytorch.org/ignite/handlers.html#ignite.handlers.EarlyStopping
time_limit (int, optioinal): Time limit for training in seconds.
train_dataset_size_limit (int, optional): If specified, only the subset of training dataset is used.
    Useful for quick preliminary check before using the whole dataset.
val_dataset_size_limit (int, optional): If specified, only the subset of validation dataset is used.
    useful for qucik preliminary check before using the whole dataset.
cudnn_deterministic (bool, optional): Value for torch.backends.cudnn.deterministic.
    See https://pytorch.org/docs/stable/notes/randomness.html for details.
cudnn_benchmark (bool, optional): Value for torch.backends.cudnn.benchmark.
    See https://pytorch.org/docs/stable/notes/randomness.html for details.
mlflow_logging (bool, optional): If True and MLflow is installed, MLflow logging is enabled.
```

Please see the [example code using MNIST dataset](https://github.com/Minyus/pipelinex/blob/master/examples/mnist/mnist_with_declarative_trainer.py) prepared based on the [original code](https://github.com/pytorch/ignite/blob/master/examples/mnist/mnist.py).

It is also possible to use:

- [FlexibleModelCheckpoint](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/ops/ignite/handlers/flexible_checkpoint.py) handler which enables to use timestamp in the model checkpoint file name to clarify which one is the latest.
- [CohenKappaScore](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/ops/ignite/metrics/cohen_kappa_score.py) metric which can compute Quadratic Weighted Kappa Metric used in some Kaggle competitions. See [sklearn.metrics.cohen_kappa_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html) for details.

It is planned to port some [code used with PyTorch Ignite](https://github.com/Minyus/pipelinex/tree/master/src/pipelinex/ops/ignite) to [PyTorch Ignite](https://github.com/pytorch/ignite) repository once test and example codes are prepared.

### Use with OpenCV

A challenge of image processing is that the parameters and algorithms that work with an image often do not work with another image. You will want to output intermediate images from each image processing pipeline step for visual check during development, but you will not want to output all the intermediate images to save time and disk space in production.

Wrappers of OpenCV and `ImagesLocalDataSet` are the solution. You can concentrate on developping your image processing pipeline for an image (3-D or 2-D numpy array), and it will run for all the images in a folder.

If you are devepping an image processing pipeline consisting of 5 steps and you have 10 images, for example, you can check 10 generated images in each of 5 folders, 50 images in total, during development.


