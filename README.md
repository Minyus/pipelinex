# PipelineX

[![PyPI version](https://badge.fury.io/py/pipelinex.svg)](
https://badge.fury.io/py/pipelinex
)

Pipeline for eXperimentation

## Overview

PipelineX is a Python package to develop pipelines for Machine Learning experimentation.

PipelineX includes thin wrappers of:
- [Kedro](https://github.com/quantumblacklabs/kedro) (A Python library for building robust production-ready data and analytics pipelines.)
- [MLflow](https://github.com/mlflow/mlflow) (Open source platform for the machine learning lifecycle)
- [PyTorch](https://github.com/pytorch/pytorch)
- [Ignite](https://github.com/pytorch/ignite) (High-level library to help with training neural networks in PyTorch)
- [Shap](https://github.com/slundberg/shap) (A unified approach to explain the output of any machine learning model.)
- [pandas](https://github.com/pandas-dev/pandas)

PipelineX shares similar philosophy, concepts, or API styles with:

- [Kedro](https://github.com/quantumblacklabs/kedro)
- [PyTorch](https://github.com/pytorch/pytorch)
- [TensorFlow/Keras](https://github.com/tensorflow/tensorflow)
- [Allennlp](https://github.com/allenai/allennlp)
- [Ludwig](https://uber.github.io/ludwig/)
- [Jsonnet](https://github.com/google/jsonnet)

If efficient experimentation is not important, other pipeline packages might be more suitable:
- [Luigi](https://github.com/spotify/luigi)
- [Oozie](https://github.com/apache/oozie)
- [Azkaban](https://github.com/azkaban/azkaban)

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

## Basic usage

To manage experiments, it is a common practice to store parameters in YAML or JSON config files.
Parameters for Machine Learning are, however not limited to (list of) numbers or string.
Which *package* or *class/function* (model, neural network module, optimizer, etc.) to use is also a parameter.

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


## Use with PyTorch

To develop a simple neural network, it is convenient to use Sequential API 
(`torch.nn.Sequential for PyTorch`, `tf.keras.Sequential` for Keras).

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

In addition to Sequential, TensorFLow/Keras provides modules to merge branches such as 
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

## Examples

- [Basic example](https://github.com/Minyus/pipelinex_pytorch)
  - Packages: PyTorch, Ignite, Shap
  - Application: Image classification
  - Model: CNN (Convolutional Neural Network)
  - Data: MNIST images
  - Loss: Cross-entropy

- [Complex example](https://github.com/Minyus/kaggle_nfl)
  - Packages: PyTorch, Ignite, pandas, numpy, Kedro, MLflow
  - Application: Kaggle competition to predict the results of American Football plays
  - Model: Combination of CNN and MLP
  - Data: Sparse heatmap-like field images and tabular data
  - Loss: Continuous Rank Probability Score (CRPS)


## Use with Ignite

Wrappers of Ignite provides features including:
- Integration with MLflow
- Use only partial samples in dataset for prototyping
- Flexible model checkpoint using timestamp in the model file name
- Time limit for training

## Use with Kedro

Wrappers of Kedro provides features including:
- Integration with MLflow
- Run only missing nodes (skip tasks which have already been run)
- Define kedro pipeline in parameters.yml instead of Python code
- List nodes (tasks) in similar way with Sequential API 
(`torch.nn.Sequential` for PyTorch, `tf.keras.Sequential` for Keras)
- Syntactic sugar for catalog.yml

## Template

General project template is available at:
https://github.com/Minyus/pipelinex_template

## Author
Yusuke Minami

- [GitHub](https://github.com/Minyus)
- [Linkedin](https://www.linkedin.com/in/yusukeminami/)
- [Twitter](https://twitter.com/Minyus86)

