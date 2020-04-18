<p align="center">
<img src="img/PipelineX_Logo.png">
</p>

# PipelineX

Pipeline for eXperimentation

[![PyPI version](https://badge.fury.io/py/pipelinex.svg)](https://badge.fury.io/py/pipelinex)
![Python Version](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue.svg)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/ambv/black)

## Overview

PipelineX is a Python package to develop pipelines for rapid Machine/Deep Learning experimentation.

<p align="center">
<img src="img/ML_project_from_experimentation_to_production.png">
</p

Please refer [here](https://github.com/Minyus/Python_Packages_for_Pipeline_Workflow) to find out how PipelineX differs from other pipeline/workflow packages: Airflow, Luigi, Gokart, Metaflow, and Kedro.

PipelineX includes integration with:

- [Kedro](https://github.com/quantumblacklabs/kedro) (A Python library for building robust production-ready data and analytics pipelines.)
  - Optional integration with [MLflow](https://github.com/mlflow/mlflow) (Open source platform for the machine learning lifecycle)
- [PyTorch](https://github.com/pytorch/pytorch)
- [Ignite](https://github.com/pytorch/ignite) (High-level library to help with training neural networks in PyTorch)
- [pandas](https://github.com/pandas-dev/pandas)
- [Shap](https://github.com/slundberg/shap) (A unified approach to explain the output of any machine learning model.)
- [OpenCV](https://github.com/skvark/opencv-python)

These wrappers are all independent and optional. You do _not_ need to install all of these packages.

PipelineX shares similar philosophy, concepts, or API styles with:

- [Kedro](https://github.com/quantumblacklabs/kedro)
- [PyTorch](https://github.com/pytorch/pytorch)
- [TensorFlow/Keras](https://github.com/tensorflow/tensorflow)
- [Allennlp](https://github.com/allenai/allennlp)
- [Ludwig](https://uber.github.io/ludwig/)
- [Jsonnet](https://github.com/google/jsonnet)

## Installation

Option 1: install from the PyPI

```bash
$ pip3 install pipelinex
```

Option 2: install from the GitHub repository

```bash
$ pip3 install git+https://github.com/Minyus/pipelinex.git
```

Option 3: clone the [GitHub repository](https://github.com/Minyus/pipelinex.git), cd into the
downloaded repository, and run:

```bash
$ python setup.py install
```

## Why and how PipelineX was born

When I was working on a Deep Learning project, it was very time-consuming to develop the pipeline for experimentation.
I wanted 2 features.

First one was an option to resume the pipeline using the intermediate data files instead of running the whole pipeline.
This was important for rapid Machine/Deep Learning experimentation.

Second one was modularity, which means keeping the 3 components, task processing, file/database access, and DAG definition, independent.
This was important for efficient software engineering.

After this project, I explored for a long-term solution.
I researched about 3 Python packages for pipeline development, Airflow, Luigi, and Kedro, but none of these could be a solution.

Luigi provided resuming feature, but did not offer modularity.
Kedro offered modularity, but did not provide resuming feature.

After this research, I decided to develop my own package that works on top of Kedro.
Besides, I added syntactic sugars including Sequential API similar to Keras and PyTorch to define DAG.
Furthermore, I added integration with MLflow, PyTorch, Ignite, pandas, OpenCV, etc. while working on more Machine/Deep Learning projects.

After I confirmed my package worked well with the Kaggle competition, I released it as PipelineX.

## Example/Demo Projects

- [Computer Vision using PyTorch](https://github.com/Minyus/pipelinex_pytorch)

  - Packages: PyTorch, Ignite, Shap, Kedro, MLflow
  - Application: Image classification
  - Data: MNIST images
  - Model: CNN (Convolutional Neural Network)
  - Loss: Cross-entropy

- [Kaggle competition using PyTorch](https://github.com/Minyus/kaggle_nfl)

  - Packages: PyTorch, Ignite, pandas, numpy, Kedro, MLflow
  - Application: Kaggle competition to predict the results of American Football plays
  - Data: Sparse heatmap-like field images and tabular data
  - Model: Combination of CNN and MLP
  - Loss: Continuous Rank Probability Score (CRPS)

- [Computer Vision using OpenCV](https://github.com/Minyus/pipelinex_image_processing)
  - Packages: OpenCV, Scikit-image, numpy, TensorFlow (pretrained model), Kedro, MLflow
  - Application: Image processing to estimate the empty area ratio of cuboid container on a truck
  - Data: container images

## Template

General PipelineX project template is available at:
https://github.com/Minyus/pipelinex_template

## Use as a powerful YAML/JSON parser

### Python objects as parameters

To manage experiments, it is a common practice to store parameters in YAML or JSON config files.
Parameters for Machine Learning are, however not limited to (list of) numbers or string.
Which _package_ or _class/function_ (model, neural network module, optimizer, etc.) to use is also a parameter.

PipelineX can parse Python dictionaries read from YAML or JSON files and convert to objects or instances.

- Use `=` key to specify the package, module, and class/function.
- Use `_` key to specify positional arguments.

Example:

- Hardcoded:

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0, random_state=42, max_iter=100)
print("model object: \n", model, "\n")
```

```
> model object:
>  LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
> 				   intercept_scaling=1, l1_ratio=None, max_iter=100,
> 				   multi_class='warn', n_jobs=None, penalty='l2',
> 				   random_state=42, solver='warn', tol=0.0001, verbose=0,
> 				   warm_start=False)
```

- Using experimentation config:

```python
from pipelinex import HatchDict
import yaml
from pprint import pformat

# Read parameters dict from a YAML file in actual use
params_yaml="""
model:
  =: sklearn.linear_model.LogisticRegression
  C: 1.23456
  max_iter: 987
  random_state: 42
"""
parameters = yaml.safe_load(params_yaml)

model_dict = parameters.get("model")
print("model dict: \n", pformat(model_dict), "\n")

model = HatchDict(parameters).get("model")
print("model object: \n", model, "\n")
```

```
> model dict:
>  {'=': 'sklearn.linear_model.LogisticRegression',
>  'C': 1.23456,
>  'max_iter': 987,
>  'random_state': 42}
>
> model object:
>  LogisticRegression(C=1.23456, class_weight=None, dual=False, fit_intercept=True,
>                    intercept_scaling=1, l1_ratio=None, max_iter=987,
>                    multi_class='warn', n_jobs=None, penalty='l2',
>                    random_state=42, solver='warn', tol=0.0001, verbose=0,
>                    warm_start=False)
>
```

### Self-lookup

You can look up another value in the YAML/JSON file using `=` key, which is simpler than YAML's Anchor&Alias and Jsonnet's Variable.
To specify the nested key (key in a dict of dict), use `.` as the separator.

```python
from pipelinex import HatchDict
import yaml
from pprint import pformat

# Read parameters dict from a YAML file in actual use
params_yaml="""
train_params:
  train_batch_size: 32
  val_batch_size: {=: train_params.train_batch_size}
"""
parameters = yaml.safe_load(params_yaml)

train_params_dict = parameters.get("train_params")
print("train_params dict: \n", pformat(train_params_dict), "\n")

train_params = HatchDict(parameters).get("train_params")
print("train_params object: \n", train_params, "\n")
```

```
> train_params dict:
>  {'train_batch_size': 32,
>  'val_batch_size': {'=': 'train_params.train_batch_size'}}
>
> train_params object:
>  {'train_batch_size': 32, 'val_batch_size': 32}
```

### Expression by parentheses

Strings wrapped in parentheses are evaluated as an expression.

```python
from pipelinex import HatchDict
import yaml
from pprint import pformat

# Read parameters dict from a YAML file in actual use
params_yaml="""
train_params:
  train_batch_size: (8+8+8+8)
"""
parameters = yaml.safe_load(params_yaml)

train_params_dict = parameters.get("train_params")
print("train_params dict: \n", pformat(train_params_dict), "\n")

train_params = HatchDict(parameters).get("train_params")
print("train_params object: \n", train_params, "\n")
```

```
> train_params dict:
>  {'train_batch_size': '(8+8+8+8)'}
>
> train_params object:
>  {'train_batch_size': 32}
```

## Use with Kedro

Kedro is a Python package to develop pipelines consisting of tasks (called `node`).

Regarding Kedro, please see the [Kedro's document](https://kedro.readthedocs.io/en/latest/)
and comparison with other pipeline/workflow packages [here](https://github.com/Minyus/Python_Packages_for_Pipeline_Workflow).

PipelineX enables you to use Kedro in more convenient ways:

- Configure Kedro run config in `parameters.yml` using `RUN_CONFIG` key
  - Optionally run only missing nodes (skip tasks which have already been run to resume pipeline using the intermediate data files or databases.)
  - Optionally run nodes in parallel
- Define Kedro pipeline in `parameters.yml` using `PIPELINES` key
  - Optionally specify the default Python module (path of .py file) if you want to omit the module name
  - Optionally specify the Python function decorator to apply to each node
  - Specify `inputs`, `func`, and `outputs` for each node
    - For sub-pipelines consisting of nodes of only single input and single output, you can optionally use Sequential API similar to PyTorch (`torch.nn.Sequential`) and Keras (`tf.keras.Sequential`)
- Integration with MLflow
  - Optionally specify the MLflow tracking database URI
    (For more details, see [SQLAlchemy database uri](https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls))
  - Optionally specify the experiment name
  - Optionally specify the location to save artifacts
  - Optionally specify the offset hour (local time zone) to show in MLflow log (e.g. 0 for UK, 8 for Singapore)
  - Optionally specify the artifacts (e.g. parameters, trained model, prediction) to save
- Syntactic sugar for `catalog.yml`
  - Optionally specify the `filepath` as the catalog entry name so the file name without extension is used as the DataSet instance name
  - Optionally specify the artifact (file) to log to MLflow's directory using `mlflow_logging` key (`mlflow.log_artifact` function is used under the hood.)
  - Optionally enable caching using `cached` key set to True if you do not want Kedro to load the data from disk/database which were in the memory. (`kedro.contrib.io.cached.CachedDataSet` is used under the hood.)
  - Optionally specify the default `DataSet` and its parameters using `/` key so you can reduce copying.

```yaml
# parameters.yml

RUN_CONFIG:
  pipeline_name: __default__
  only_missing: False # Set True to run only missing nodes
  runner: SequentialRunner # Set to "ParallelRunner" to run in parallel

PIPELINES:
  __default__:
    =: pipelinex.FlexiblePipeline
    module: # Optionally specify the default Python module so you can omit the module name to which functions belongs
    decorator: # Optionally specify function decorator(s) to apply to each node
    nodes:
      - inputs: input_data_1
        func: my_module.processing_task_1
        outputs: [intermediate_data_1, intermediate_data_2]

      - inputs: intermediate_data_1
        func:
          - my_module.processing_task_2
          - my_module.processing_task_3
        outputs: output_data

MLFLOW_LOGGING_CONFIG:
  uri: sqlite:///mlruns/sqlite.db
  experiment_name: experiment_001
  artifact_location: ./mlruns/experiment_001
  offset_hours: 0 # Specify the offset hour (local time zone) to show in MLflow tracking
  logging_artifacts: # Optionally specify artifacts (e.g. parameters, trained model, prediction) to save
```

```yaml
#  catalog.yml

/: # Optionally specify the default DataSet
  type: CSVLocalDataSet
  cached: True

data/input/input_data_1.csv: # Use the default DataSet

data/load/intermediate_data_1.pickle:
  type: PickleLocalDataSet

data/load/intermediate_data_2.csv: # Use the default DataSet

data/load/output_data.csv: # Use the default DataSet
  mlflow_logging: True
```

## Use with PyTorch

To develop a simple neural network, it is convenient to use Sequential API
(e.g. `torch.nn.Sequential`, `tf.keras.Sequential`).

- Hardcoded:

```python
from torch.nn import Sequential, Conv2d, ReLU

model = Sequential(
    Conv2d(in_channels=3, out_channels=16, kernel_size=[3, 3]),
    ReLU(),
)
print("model object: \n", model, "\n")
```

```
> model object:
>  Sequential(
>   (0): Conv2d(3, 16, kernel_size=[3, 3], stride=(1, 1))
>   (1): ReLU()
> )
```

- Using experimentation config:

```python
from pipelinex import HatchDict
import yaml
from pprint import pformat

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
print("model dict: \n", pformat(model_dict), "\n")

model = HatchDict(parameters).get("model")
print("model object: \n", model, "\n")
```

```
> model dict:
>  {'=': 'torch.nn.Sequential',
>  '_': [{'=': 'torch.nn.Conv2d',
>         'in_channels': 3,
>         'kernel_size': [3, 3],
>         'out_channels': 16},
>        {'=': 'torch.nn.ReLU', '_': None}]}
>
> model object:
>  Sequential(
>   (0): Conv2d(3, 16, kernel_size=[3, 3], stride=(1, 1))
>   (1): ReLU()
> )
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
print("model object: \n", model, "\n")
```

```
> model object:
>  Sequential(
>   (0): ModuleConcat(
>     (0): Conv2d(3, 16, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1])
>     (1): AvgPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1])
>   )
>   (1): ReLU()
> )
>
```

- Using experimentation config:

```python
from pipelinex import HatchDict
import yaml
from pprint import pformat

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
print("model dict: \n", pformat(model_dict), "\n")

model = HatchDict(parameters).get("model")
print("model object: \n", model, "\n")
```

```
> model dict:
>  {'=': 'torch.nn.Sequential',
>  '_': [{'=': 'pipelinex.ModuleConcat',
>         '_': [{'=': 'torch.nn.Conv2d',
>                'in_channels': 3,
>                'kernel_size': [3, 3],
>                'out_channels': 16,
>                'padding': [1, 1],
>                'stride': [2, 2]},
>               {'=': 'torch.nn.AvgPool2d',
>                'kernel_size': [3, 3],
>                'padding': [1, 1],
>                'stride': [2, 2]}]},
>        {'=': 'torch.nn.ReLU', '_': None}]}
>
> model object:
>  Sequential(
>   (0): ModuleConcat(
>     (0): Conv2d(3, 16, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1])
>     (1): AvgPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1])
>   )
>   (1): ReLU()
> )
```

## Use with Ignite

Wrappers of Ignite provides features useful for training including:

- Integration with MLflow
- Use only partial samples in dataset for prototyping
- Flexible model checkpoint using timestamp in the model file name
- Time limit for training

## Use with OpenCV

A challenge of image processing is that the parameters and algorithms that work with an image often do not work with another image. You will want to output intermediate images from each image processing pipeline step for visual check during development, but you will not want to output all the intermediate images to save time and disk space in production.

Wrappers of OpenCV and `ImagesLocalDataSet` are the solution. You can concentrate on developping your image processing pipeline for an image (3-D or 2-D numpy array), and it will run for all the images in a folder.

If you are devepping an image processing pipeline consisting of 5 steps and you have 10 images, for example, you can check 10 generated images in each of 5 folders, 50 images in total, during development.

## Use with Docker container

### Build Docker image

```bash
# docker build -t pipelinex:3.7.7-slim -f dockerfiles/dockerfile .
```

### Use with Docker container

```bash
# docker run -it --name pipelinex pipelinex:3.7.7-slim /bin/bash
```

## Author

Yusuke Minami

- [GitHub](https://github.com/Minyus)
- [Linkedin](https://www.linkedin.com/in/yusukeminami/)
- [Twitter](https://twitter.com/Minyus86)

## Contributors

- [shibuiwilliam](https://github.com/shibuiwilliam)
