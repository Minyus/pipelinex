<p align="center">
<img src="img/PipelineX_Logo.png">
</p>

# PipelineX

Pipeline for eXperimentation

[![PyPI version](https://badge.fury.io/py/pipelinex.svg)](https://badge.fury.io/py/pipelinex)
![Python Version](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue.svg)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/ambv/black)

## Overview

PipelineX is a Python package designed to make Machine Learning projects efficient with modular, reusable, and easy-to-use features for experimentation.

<p align="center">
<img src="img/ML_project_from_experimentation_to_production.png">
</p>

Please refer [here](https://github.com/Minyus/Python_Packages_for_Pipeline_Workflow) to find out how PipelineX differs from other pipeline/workflow packages: Airflow, Luigi, Gokart, Metaflow, and Kedro.

PipelineX provides [enhancements for YAML/JSON](https://github.com/Minyus/pipelinex#pythonic-enhanced-yamljson) useful for parameter management summarized as follows.

- Import-less Python object: Include (nested) Python classes and functions in a YAML/JSON file
- Anchor-less aliasing: Look up another key in the same YAML/JSON file
- Python expression in YAML/JSON files

PipelineX provides [enhancements for Kedro](https://github.com/Minyus/pipelinex#yamlconfigurable-enhanced-kedro) summarized as follows.

- Kedro pipeline/DAG definition in a YAML/JSON file with more options
- Additional Kedro-compatible data interface sets ("DataSets") for Computer Vision applications
- Additional decorators for benchmarking
- Integration with MLflow that enables to save metrics to a database supported by SQLAlchemy (SQLite, PostgreSQL, etc.)

PipelineX includes integration with the following Python packages.

- [Kedro](https://github.com/quantumblacklabs/kedro)
- [MLflow](https://github.com/mlflow/mlflow)
- [PyTorch](https://github.com/pytorch/pytorch)
- [Ignite](https://github.com/pytorch/ignite)
- [Pandas](https://github.com/pandas-dev/pandas)
- [OpenCV](https://github.com/skvark/opencv-python)
- [Memory Profiler](https://github.com/pythonprofilers/memory_profiler)
- [Python bindings to the NVIDIA Management Library](https://github.com/gpuopenanalytics/pynvml)
- [Shap](https://github.com/slundberg/shap)

These wrappers are all independent and optional. You do _not_ need to install all of these Python packages.

PipelineX shares similar philosophy/concepts with:

- Pipeline/workflow packages: [Apache Beam](https://github.com/apache/beam), [Argo](https://github.com/argoproj/argo), [Kubeflow](https://github.com/kubeflow/kubeflow),  [Apache Airflow](https://github.com/apache/airflow), [Luigi](https://github.com/spotify/luigi), [Gokart](https://github.com/m3dev/gokart), [Metaflow](https://github.com/Netflix/metaflow) 

- Parameter/Config management packages: [Hydra](https://github.com/facebookresearch/hydra), [Jsonnet](https://github.com/google/jsonnet), [Helm](https://github.com/helm/helm), [ytt](https://github.com/k14s/ytt)


PipelineX shares similar API styles with 

- Pipeline/workflow package: [Kubeflow](https://github.com/kubeflow/kubeflow)

- Domain-specific packages: [Allennlp](https://github.com/allenai/allennlp), [Ludwig](https://uber.github.io/ludwig/), [Detectron2](https://github.com/facebookresearch/detectron2)


## Installation

- [Option 1] To install the latest release from the PyPI:

```bash
$ pip install pipelinex
```

- [Option 2] To install the latest pre-release:

```bash
$ pip install git+https://github.com/Minyus/pipelinex.git
```

- [Option 3] To install the latest pre-release without need to reinstall even after modifying the source code:

```bash
$ git clone https://github.com/Minyus/pipelinex.git
$ cd pipelinex
$ python setup.py develop
```

## Example/Demo Projects

- [Entry example using Scikit-learn to demonstrate PipelineX's Kedro-MLflow integration](https://github.com/Minyus/kedro_mlflow)

  - `parameters.yml` at [conf/base/parameters.yml](https://github.com/Minyus/pipelinex_sklearn/blob/master/conf/base/parameters.yml)

  - Essential packages: Scikit-learn, pandas, Kedro, MLflow
  - Application: Kaggle's exercise competition to predict which Titanic's passengers survived
  - Data: [Kaggle's Titanic](https://www.kaggle.com/c/titanic/data)
  - Model: Logistic Regression

- [Entry example using Scikit-learn to demonstrate more PipelineX's options](https://github.com/Minyus/pipelinex_sklearn)

  - Adds more PipelineX's options, such as declaring Python objects in YAML, to the previous example.

- [Computer Vision using PyTorch](https://github.com/Minyus/pipelinex_pytorch)

  - `parameters.yml` at [conf/base/parameters.yml](https://github.com/Minyus/pipelinex_pytorch/blob/master/conf/base/parameters.yml)

  - Essential packages: PyTorch, Ignite, Shap, Kedro, MLflow
  - Application: Image classification
  - Data: MNIST images
  - Model: CNN (Convolutional Neural Network)
  - Loss: Cross-entropy

- [Kaggle competition using PyTorch](https://github.com/Minyus/kaggle_nfl)

  - `parameters.yml` at [kaggle/conf/base/parameters.yml](https://github.com/Minyus/kaggle_nfl/blob/master/kaggle/conf/base/parameters.yml)

  - Essential packages: PyTorch, Ignite, pandas, numpy, Kedro, MLflow
  - Application: [Kaggle competition to predict the results of American Football plays](https://www.kaggle.com/c/nfl-big-data-bowl-2020/data)
  - Data: Sparse heatmap-like field images and tabular data
  - Model: Combination of CNN and MLP
  - Loss: Continuous Rank Probability Score (CRPS)

- [Computer Vision using OpenCV](https://github.com/Minyus/pipelinex_image_processing)

  - `parameters.yml` at [conf/base/parameters.yml](https://github.com/Minyus/pipelinex_image_processing/blob/master/conf/base/parameters.yml)
  - Essential packages: OpenCV, Scikit-image, numpy, TensorFlow (pretrained model), Kedro, MLflow
  - Application: Image processing to estimate the empty area ratio of cuboid container on a truck
  - Data: container images

- [Uplift Modeling using CausalLift](https://github.com/Minyus/pipelinex_causallift)

  - `parameters.yml` at [conf/base/parameters.yml](https://github.com/Minyus/pipelinex_causallift/blob/master/conf/base/parameters.yml)
  - Essential packages: CausalLift, Scikit-learn, XGBoost, pandas, Kedro
  - Application: Uplift Modeling to find which customers should be targeted and which customers should not for a marketing campaign (treatment)
  - Data: generated by simulation

## Template repositories

The following 2 template repositories for PipelineX are available.

- [Template repository to use YAML less than Kedro](https://github.com/Minyus/kedro_template): likely preferable for first-time users.

- [Template repository to use YAML more than Kedro](https://github.com/Minyus/pipelinex_template): potentially preferable for users who are not satisfied with Kedro as is.

These were simplified versions of the template project created by `kedro new` command which uses Cookiecutter.

To use for a new project, fork the template repository and hit `Use this template` button next to `Clone or download`.

<p align="center">
<img src="https://help.github.com/assets/images/help/repository/use-this-template-button.png">
</p>

## Pythonic enhanced YAML/JSON (`HatchDict`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Minyus/pipelinex/blob/master/notebooks/HatchDict_demo.ipynb)

### Import-less Python object (class and function)

YAML is a common text format used for application config files.

YAML's most notable advantage is allowing users to mix 2 styles, block style and flow style.

Example:

```python
import yaml
from pprint import pprint  # pretty-print for clearer look

# Read parameters dict from a YAML file in actual use
params_yaml="""
block_style_demo:
  key1: value1
  key2: value2
flow_style_demo: {key1: value1, key2: value2}
"""
parameters = yaml.safe_load(params_yaml)

print("### 2 styles in YAML ###")
pprint(parameters)
```

```
### 2 styles in YAML ###
{'block_style_demo': {'key1': 'value1', 'key2': 'value2'},
 'flow_style_demo': {'key1': 'value1', 'key2': 'value2'}}
```

To store highly nested (hierarchical) dict or list, YAML is more conveinient than hard-coding in Python code.

- YAML's block style, which uses indentation, allows users to omit opening and closing symbols to specify a Python dict or list (`{}` or `[]`).

- YAML's flow style, which uses opening and closing symbols, allows users to specify a Python dict or list within a single line.

So simply using YAML with Python will be the best way for Machine Learning experimentation?

Let's check out the next example.

Example:

```python
import yaml
from pprint import pprint  # pretty-print for clearer look


# Read parameters dict from a YAML file in actual use
params_yaml = """
model_kind: LogisticRegression
model_params:
  C: 1.23456
  max_iter: 987
  random_state: 42
"""
parameters = yaml.safe_load(params_yaml)

print("### Before ###")
pprint(parameters)

model_kind = parameters.get("model_kind")
model_params_dict = parameters.get("model_params")

if model_kind == "LogisticRegression":
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(**model_params_dict)

elif model_kind == "DecisionTree":
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(**model_params_dict)

elif model_kind == "RandomForest":
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(**model_params_dict)

else:
    raise ValueError("Unsupported model_kind.")

print("\n### After ###")
print(model)
```

```
### Before ###
{'model_kind': 'LogisticRegression',
 'model_params': {'C': 1.23456, 'max_iter': 987, 'random_state': 42}}

### After ###
LogisticRegression(C=1.23456, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=987,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=42, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)
```

This way is inefficient as we need to add `import` and `if` statements for the options in the Python code in addition to modifying the YAML config file.

Any better way?

PyYAML provides [UnsafeLoader](<https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation>) which can load Python objects without `import`.

```python
import yaml
# You do not need `import sklearn.linear_model` using PyYAML's UnsafeLoader


# Read parameters dict from a YAML file in actual use
params_yaml = """
model:
  !!python/object:sklearn.linear_model.LogisticRegression
  C: 1.23456
  max_iter: 987
  random_state: 42
"""

parameters = yaml.unsafe_load(params_yaml)  # unsafe_load required

model = parameters.get("model")

print("### model object by PyYAML's UnsafeLoader ###")
print(model)
```

```
### model object by PyYAML's UnsafeLoader ###
LogisticRegression(C=1.23456, class_weight=None, dual=None, fit_intercept=None,
                   intercept_scaling=None, l1_ratio=None, max_iter=987,
                   multi_class=None, n_jobs=None, penalty=None, random_state=42,
                   solver=None, tol=None, verbose=None, warm_start=None)
```

[PyYAML's `!!python/object` and `!!python/name`](https://pyyaml.org/wiki/PyYAMLDocumentation), however, has the following problems.

- `!!python/object` or `!!python/name` are too long to write.
- Positional (non-keyword) arguments are apparently not supported.

Any better way?

PipelineX provides the solution.

PipelineX's HatchDict provides an easier syntax, as follows, to convert Python dictionaries read from YAML or JSON files to Python objects without `import`.

- Use `=` key to specify the package, module, and class/function with `.` separator in `foo_package.bar_module.baz_class` format.
- [Optional] Use `_` key to specify (list of) positional arguments (args) if any.
- [Optional] Add keyword arguments (kwargs) if any.

To return an object instance like PyYAML's `!!python/object`, feed positional and/or keyword arguments. If there is no arguments, just feed null (known as `None` in Python) to `_` key.

To return an uninstantiated (raw) object like PyYAML's `!!python/name`, just feed `=` key without positional nor keyword arugments.

Example:

```python
from pipelinex import HatchDict
import yaml
from pprint import pprint  # pretty-print for clearer look
# You do not need `import sklearn.linear_model` using PipelineX's HatchDict

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

print("### Before ###")
pprint(model_dict)

model = HatchDict(parameters).get("model")

print("\n### After ###")
print(model)
```

```
### Before ###
{'=': 'sklearn.linear_model.LogisticRegression',
 'C': 1.23456,
 'max_iter': 987,
 'random_state': 42}

### After ###
LogisticRegression(C=1.23456, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=987,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=42, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)
```

This import-less Python object supports nested objects (objects that receives object arguments) by recursive depth-first search.

For more examples, please see [Use with PyTorch](https://github.com/Minyus/pipelinex#use-with-pytorch) and `parameters.yml` in [example/demo projects](https://github.com/Minyus/pipelinex#exampledemo-projects) .

This import-less Python object feature, inspired by the fact that Kedro uses `load_obj` for file I/O (`DataSet`), uses `load_obj` copied from [kedro.utils](https://github.com/quantumblacklabs/kedro/blob/0.15.4/kedro/utils.py) which dynamically imports Python objects using [`importlib`](https://docs.python.org/3.6/library/importlib.html), a Python standard library.

### Anchor-less aliasing (self-lookup)

To avoid repeating, YAML natively provides Anchor&Alias [Anchor&Alias](https://confluence.atlassian.com/bitbucket/yaml-anchors-960154027.html) feature, and [Jsonnet](https://github.com/google/jsonnet) provides [Variable](https://github.com/google/jsonnet/blob/master/examples/variables.jsonnet) feature to JSON.

Example:

```python
import yaml
from pprint import pprint  # pretty-print for clearer look

# Read parameters dict from a YAML file in actual use
params_yaml="""
train_params:
  train_batch_size: &batch_size 32
  val_batch_size: *batch_size
"""
parameters = yaml.safe_load(params_yaml)

train_params_dict = parameters.get("train_params")

print("### Conversion by YAML's Anchor&Alias feature ###")
pprint(train_params_dict)
```

```
### Conversion by YAML's Anchor&Alias feature ###
{'train_batch_size': 32, 'val_batch_size': 32}
```

Unfortunately, YAML and Jsonnet require a medium to share the same value.

This is why PipelineX provides Anchor-less aliasing feature.

You can directly look up another value in the same YAML/JSON file using `$` key without an anchor nor variable.

To specify the nested key (key in a dict of dict), use `.` as the separator.

Example:

```python
from pipelinex import HatchDict
import yaml
from pprint import pprint  # pretty-print for clearer look

# Read parameters dict from a YAML file in actual use
params_yaml="""
train_params:
  train_batch_size: 32
  val_batch_size: {$: train_params.train_batch_size}
"""
parameters = yaml.safe_load(params_yaml)

train_params_dict = parameters.get("train_params")

print("### Before ###")
pprint(train_params_dict)

train_params = HatchDict(parameters).get("train_params")

print("\n### After ###")
pprint(train_params)
```

```
### Before ###
{'train_batch_size': 32,
 'val_batch_size': {'$': 'train_params.train_batch_size'}}

### After ###
{'train_batch_size': 32, 'val_batch_size': 32}
```

### Python expression

Strings wrapped in parentheses are evaluated as a Python expression.

```python
from pipelinex import HatchDict
import yaml
from pprint import pprint  # pretty-print for clearer look

# Read parameters dict from a YAML file in actual use
params_yaml = """
train_params:
  param1_tuple_python: (1, 2, 3)
  param1_tuple_yaml: !!python/tuple [1, 2, 3]
  param2_formula_python: (2 + 3)
  param3_neg_inf_python: (float("-Inf"))
  param3_neg_inf_yaml: -.Inf
  param4_float_1e9_python: (1e9)
  param4_float_1e9_yaml: 1.0e+09
  param5_int_1e9_python: (int(1e9))
"""
parameters = yaml.load(params_yaml)

train_params_raw = parameters.get("train_params")

print("### Before ###")
pprint(train_params_raw)

train_params_converted = HatchDict(parameters).get("train_params")

print("\n### After ###")
pprint(train_params_converted)
```

```
### Before ###
{'param1_tuple_python': '(1, 2, 3)',
 'param1_tuple_yaml': (1, 2, 3),
 'param2_formula_python': '(2 + 3)',
 'param3_neg_inf_python': '(float("-Inf"))',
 'param3_neg_inf_yaml': -inf,
 'param4_float_1e9_python': '(1e9)',
 'param4_float_1e9_yaml': 1000000000.0,
 'param5_int_1e9_python': '(int(1e9))'}

### After ###
{'param1_tuple_python': (1, 2, 3),
 'param1_tuple_yaml': (1, 2, 3),
 'param2_formula_python': 5,
 'param3_neg_inf_python': -inf,
 'param3_neg_inf_yaml': -inf,
 'param4_float_1e9_python': 1000000000.0,
 'param4_float_1e9_yaml': 1000000000.0,
 'param5_int_1e9_python': 1000000000}
```

## The unified data interface framework

Machine Learning projects involves with loading and saving various data in various ways such as:

- files in local/network file system, Hadoop File System (HDFS), Amazon S3, Google Cloud Storage
  - e.g. CSV, JSON, YAML, pickle, images, models, etc.
- databases 
  - Postgresql, MySQL etc.
- Spark
- REST API (HTTP(S) requests)

It is often the case that many Machine Learning Engineers code both data loading/saving and data transformation mixed in the same Python module or Jupyter notebook during experimentation/prototyping phase and suffer later on because:

- During experimentation/prototyping, we often want to save the intermediate data after each transformation. 
- In production environments, we often want to skip saving data to minimize latency and storage space.
- To benchmark the performance or troubleshoot, we often want to switch the data source.
  - e.g. read image files in local storage or download images through REST API

The proposed solution is the unified data interface.

Here is a simple demo example to predict survival on the [Titanic](https://www.kaggle.com/c/titanic/data).


<p align="center">
<img src="img/example_kedro_pipeline.PNG">
Pipeline visualized by Kedro-viz
</p>

Common code to define the tasks/operations/transformations:

```python
# Define tasks

def train_model(model, df, cols_features, col_target):
    # train a model here
    return model

def run_inference(model, df, cols_features):
    # run inference here
    return df
```

It is notable that you do _not_ need to add any Kedro-related code here to use Kedro later on.

Furthermore, you do _not_ need to add any MLflow-related code here to use MLflow later on as Kedro hooks provided by PipelineX can handle behind the scenes.

This advantage enables you to keep your pipelines for experimentation/prototyping/benchmarking production-ready.


1. Plain code:

```python
# Configure: can be written in a config file (YAML, JSON, etc.)

train_data_filepath = "data/input/train.csv"
train_data_load_args = {"float_precision": "high"}

test_data_filepath = "data/input/test.csv"
test_data_load_args = {"float_precision": "high"}

pred_data_filepath = "data/load/pred.csv"
pred_data_save_args = {"index": False, "float_format": "%.16e"}

model_kind = "LogisticRegression"
model_params_dict = {
  "C": 1.23456
  "max_iter": 987
  "random_state": 42
}

# Run tasks

import pandas as pd

if model_kind == "LogisticRegression":
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(**model_params_dict)

train_df = pd.read_csv(train_data_filepath, **train_data_load_args)
model = train_model(model, train_df)

test_df = pd.read_csv(test_data_filepath, **test_data_load_args)
pred_df = run_inference(model, test_df)
pred_df.to_csv(pred_data_filepath, **pred_data_save_args)

```

2. Following the data interface framework, objects with `_load`, and `_save` methods,  proposed by [Kedro](https://github.com/quantumblacklabs/kedro) and supported by PipelineX:

```python

# Define a data interface: better ones such as "CSVDataSet" are provided by Kedro

import pandas as pd
from pathlib import Path


class CSVDataSet:
    def __init__(self, filepath, load_args={}, save_args={}):
        self._filepath = filepath
        self._load_args = {}
        self._load_args.update(load_args)
        self._save_args = {"index": False}
        self._save_args.update(save_args)

    def _load(self) -> pd.DataFrame:
        return pd.read_csv(self._filepath, **self._load_args)

    def _save(self, data: pd.DataFrame) -> None:
        save_path = Path(self._filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(str(save_path), **self._save_args)


# Configure data interface: can be written in catalog config file using Kedro

train_dataset = CSVDataSet(
    filepath="data/input/train.csv",
    load_args={"float_precision": "high"},
    # save_args={"float_format": "%.16e"},  # You can set save_args for future use
)

test_dataset = CSVDataSet(
    filepath="data/input/test.csv",
    load_args={"float_precision": "high"},
    # save_args={"float_format": "%.16e"},  # You can set save_args for future use
)

pred_dataset = CSVDataSet(
    filepath="data/load/pred.csv",
    # load_args={"float_precision": "high"},  # You can set load_args for future use
    save_args={"float_format": "%.16e"},
)

model_kind = "LogisticRegression"
model_params_dict = {
  "C": 1.23456
  "max_iter": 987
  "random_state": 42
}
cols_features = [
  "Pclass",  # The passenger's ticket class
  "Parch",  # # of parents / children aboard the Titanic
]
col_target = "Survived"  # Column used as the target: whether the passenger survived or not


# Run tasks: can be configured as a pipeline using Kedro
# and can be written in parameters config file using PipelineX

if model_kind == "LogisticRegression":
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(**model_params_dict)

train_df = train_dataset._load()
model = train_model(model, train_df, cols_features, col_target)

test_df = test_dataset._load()
pred_df = run_inference(model, test_df, cols_features)

pred_dataset._save(pred_df)

```

Just following the data interface framework might be somewhat beneficial in the long run, but not enough.

Let's see what Kedro and PipelineX can do.


## Kedro

Kedro is a Python package to develop pipelines consisting of:

- data interface sets (data loading/saving wrappers, called "DataSets", that follows the unified data interface framework) such as:
  - [`pandas.CSVDataSet`](https://kedro.readthedocs.io/en/stable/kedro.extras.datasets.pandas.CSVDataSet.html#kedro.extras.datasets.pandas.CSVDataSet): a CSV file in local or cloud (Amazon S3, Google Cloud Storage) utilizing [filesystem_spec (`fsspec`)](https://github.com/intake/filesystem_spec)
  - [`pickle.PickleDataSet`](https://kedro.readthedocs.io/en/latest/kedro.extras.datasets.pickle.PickleDataSet.html): a pickle file  in local or cloud (Amazon S3, Google Cloud Storage) utilizing [filesystem_spec (`fsspec`)](https://github.com/intake/filesystem_spec)
  - [`pandas.SQLTableDataSet`](https://kedro.readthedocs.io/en/stable/kedro.extras.datasets.pandas.SQLTableDataSet.html#kedro.extras.datasets.pandas.SQLTableDataSet): a table data in an SQL database supported by [SQLAlchemy](https://www.sqlalchemy.org/features.html)
  - [data interface sets for Spark, Google BigQuery, Feather, HDF, Parquet, Matplotlib, NetworkX, Excel, and more provided by Kedro](https://kedro.readthedocs.io/en/stable/kedro.extras.datasets.html#data-sets)
  - Custom data interface sets provided by Kedro users

- tasks/operations/transformations (called "Nodes") provided by Kedro users such as:
  - data pre-processing
  - training a model
  - inference using a model

- inter-task dependency provided by Kedro users

Kedro pipelines can be run sequentially or in parallel.

Regarding Kedro, please see:
- [Kedro's document](https://kedro.readthedocs.io/en/latest/)
- [YouTube playlist: Writing Data Pipelines with Kedro](https://www.youtube.com/playlist?list=PLTU89LAWKRwEdiDKeMOU2ye6yU9Qd4MRo)
- [Python Packages for Pipeline/Workflow](https://github.com/Minyus/Python_Packages_for_Pipeline_Workflow)

Here is a simple example Kedro project.


```yaml
#  catalog.yml

train_df:
  type: pandas.CSVDataSet # short for kedro.extras.datasets.pandas.CSVDataSet
  filepath: data/input/train.csv
  load_args:
    float_precision: high
  # save_args: # You can set save_args for future use
  # float_format": "%.16e"

test_df:
  type: pandas.CSVDataSet # short for kedro.extras.datasets.pandas.CSVDataSet
  filepath: data/input/test.csv
  load_args:
    float_precision: high
  # save_args: # You can set save_args for future use
  # float_format": "%.16e"

pred_df:
  type: pandas.CSVDataSet # short for kedro.extras.datasets.pandas.CSVDataSet
  filepath: data/load/pred.csv
  # load_args: # You can set load_args for future use
  # float_precision: high
  save_args:
    float_format: "%.16e"
```

```yaml
# parameters.yml

model:
  !!python/object:sklearn.linear_model.LogisticRegression
  C: 1.23456
  max_iter: 987
  random_state: 42
cols_features: # Columns used as features in the Titanic data table
  - Pclass # The passenger's ticket class
  - Parch # # of parents / children aboard the Titanic
col_target: Survived # Column used as the target: whether the passenger survived or not
```

```python
# pipeline.py

from kedro.pipeline import Pipeline, node

from my_module import train_model, run_inference

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=train_model,
                inputs=["params:model", "train_df", "params:cols_features", "params:col_target"],
                outputs="model",
            ),
            node(
                func=run_inference,
                inputs=["model", "test_df", "params:cols_features"],
                outputs="pred_df",
            ),
        ]
    )
```

```python
# run.py

from kedro.runner import SequntialRunner

# Set up ProjectContext here

context = ProjectContext()
context.run(pipeline_name="__default__", runner=SequentialRunner())
```

Kedro pipelines can be visualized using [kedro-viz](https://github.com/quantumblacklabs/kedro-viz).

Kedro pipelines can be productionized using:
- [kedro-airflow](https://github.com/quantumblacklabs/kedro-airflow): converts a Kedro pipeline into Airflow Python operators.
- [kedro-docker](https://github.com/quantumblacklabs/kedro-docker): builds a Docker image that can run a Kedro pipeline 
- [kedro-argo](https://github.com/nraw/kedro-argo): converts a Kedro pipeline into an Argo (backend of Kubeflow) pipeline


## Supplements for Kedro

[pipelinex.extras](https://github.com/Minyus/pipelinex/tree/master/src/pipelinex/extras) provides Kedro hooks, data interface sets, and decorators to supplement [kedro.extras](https://github.com/quantumblacklabs/kedro/tree/develop/kedro/extras) as follows.


### Integration with MLflow by Kedro hooks (callbacks)

  [pipelinex.extras.hooks](https://github.com/Minyus/pipelinex/tree/master/src/pipelinex/extras/hooks) provides Kedro hooks (callbacks) to use MLflow without adding any MLflow-related code in the node (task) functions.

  - [`pipelinex.MLflowBasicLoggerHook`](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/hooks/mlflow/mlflow_basic_logger.py): Configures and logs duration time for the pipeline to MLflow with args:
  
    - enable_mlflow: Enable configuring and logging to MLflow.
    - uri: `uri` arg fed to:
        https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri
        as the MLflow tracking server URI.
        Local file path, databases supported by SQLAlchemy (sqlite, mysql, mssql, and 
        postgresql), HTTP server, Databricks workspace are supported. 
        See MLflow's document at:
        https://mlflow.org/docs/latest/tracking.html#where-runs-are-recorded
    - experiment_name: `name` arg fed to:
        https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.create_experiment
        as the MLflow experiment name.
    - artifact_location: `artifact_location` arg fed to:
        https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.create_experiment
        as the URI to store the artifacts.
        Local file paths, Amazon S3, Azure Blob Storage, Google Cloud Storage, SFTP server, 
        NFS, and HDFS are supported. 
        See MLflow's document at:
        https://mlflow.org/docs/latest/tracking.html#id10
    - run_name: Shown as 'Run Name' in the MLflow UI.
    - offset_hours: The offset hour (e.g. 0 for UTC+00:00) used for `__time_begin` and `__time_end` parameters. 

  - [`pipelinex.MLflowArtifactsLoggerHook`](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/hooks/mlflow/mlflow_artifacts_logger.py): Logs artifacts of specified file paths and dataset names to MLflow with args:

    - enable_mlflow: Enable logging to MLflow.
    - filepaths_before_pipeline_run: The file paths of artifacts to log before the pipeline is run.
    - datasets_after_node_run: The dataset names to log after the node is run.
    - filepaths_after_pipeline_run: The file paths of artifacts to log after the pipeline is run.
  
  - [`pipelinex.MLflowDataSetsLoggerHook`](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/hooks/mlflow/mlflow_outputs_logger.py): Logs datasets of (list of) float/int and str classes to MLflow with arg:

    - enable_mlflow: Enable logging to MLflow.
  
  - [`pipelinex.MLflowTimeLoggerHook`](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/hooks/mlflow/mlflow_time_logger.py): Logs duration time for each node (task) to MLflow and optionally visualizes the execution logs as a Gantt chart by [`plotly.figure_factory.create_gantt`](https://plotly.github.io/plotly.py-docs/generated/plotly.figure_factory.create_gantt.html) if `plotly` is installed, with args:
    - enable_mlflow: Enable logging to MLflow.
    - enable_plotly: Enable visualization of logged time as a gantt chart.
    - gantt_filepath: File path to save the generated gantt chart.
    - gantt_params: Args fed to:
        https://plotly.github.io/plotly.py-docs/generated/plotly.figure_factory.create_gantt.html
    - metric_name_prefix: Prefix for the metric names. The metric names are
        - `metric_name_prefix` concatenated with the string returned by `task_name_func`.
    - task_name_func: Callable to return the task name using ``kedro.pipeline.node.Node``
        - object.
  
  - [`pipelinex.AddTransformersHook`](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/hooks/add_transformers.py): Adds Kedro transformers such as:
    - [`pipelinex.MLflowIOTimeLoggerTransformer`](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/transformers/mlflow/mlflow_io_time_logger.py): Logs duration time to load and save each dataset with args:
      - enable_mlflow: Enable logging to MLflow.
      - metric_name_prefix: Prefix for the metric names. The metric names are `metric_name_prefix` concatenated with 'load <data_set_name>' or 'save <data_set_name>'

  To use these hooks, please see example projects at [kedro_mlflow](https://github.com/Minyus/kedro_mlflow) or [pipelinex_sklearn](https://github.com/Minyus/pipelinex_sklearn)

<p align="center">
<img src="img/mlflow_ui.png">
Experiment logs in MLflow's UI
</p>


### Additional Kedro data interface sets
  
[pipelinex.extras.datasets](https://github.com/Minyus/pipelinex/tree/master/src/pipelinex/extras/datasets) provides the following data interface sets mainly for Computer Vision applications using OpenCV, Scikit-image, PyTorch/torchvision, and TensorFlow/Keras.

- [pipelinex.ImagesLocalDataSet](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/datasets/pillow/images.py)
  - loads/saves multiple numpy arrays (RGB, BGR, or monochrome image) from/to a folder in local storage using `pillow` package, working like ``kedro.extras.datasets.pillow.ImageDataSet`` and
  ``kedro.io.PartitionedDataSet`` with conversion between numpy arrays and Pillow images.
  - an example project is at [pipelinex_image_processing](https://github.com/Minyus/pipelinex_image_processing)
- [pipelinex.APIDataSet](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/datasets/requests/api_dataset.py)
  - downloads multiple contents (such as images and json) by HTTP requests using `requests` package
  - an example project is at [pipelinex_image_processing](https://github.com/Minyus/pipelinex_image_processing)
- [pipelinex.AsyncAPIDataSet](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/datasets/httpx/async_api_dataset.py)
  - downloads multiple contents (such as images and json) by asynchronous HTTP requests using `httpx` package
  - an example project is at [pipelinex_image_processing](https://github.com/Minyus/pipelinex_image_processing)

- [pipelinex.IterableImagesDataSet](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/datasets/torchvision/iterable_images.py)
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


## Enhanced Kedro context: YAML interface for Kedro pipelines

PipelineX enables you to use Kedro in more convenient ways.
Using [pipelinex.FlexibleContext](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/framework/context/flexible_context.py), you can define the inter-task dependency (DAG) for Kedro pipelines in YAML.

### Here are the options configurable in `parameters.yml`:

#### `HatchDict` features

```yaml
# parameters.yml

model:
  =: sklearn.linear_model.LogisticRegression
  C: 1.23456
  max_iter: 987
  random_state: 42
cols_features: # Columns used as features in the Titanic data table
  - Pclass # The passenger's ticket class
  - Parch # # of parents / children aboard the Titanic
col_target: Survived # Column used as the target: whether the passenger survived or not
```

#### Define Kedro pipelines using `PIPELINES` key
  - Optionally specify the default Python module (path of .py file) if you want to omit the module name
  - Optionally specify the Python function decorator to apply to each node
  - Specify `inputs`, `func`, and `outputs` for each node
    - For sub-pipelines consisting of nodes of only single input and single output, you can optionally use Sequential API similar to PyTorch (`torch.nn.Sequential`) and Keras (`tf.keras.Sequential`)

```yaml
# parameters.yml

PIPELINES:
  __default__:
    =: pipelinex.FlexiblePipeline
    module: # Optionally specify the default Python module so you can omit the module name to which functions belongs
    decorator: # Optionally specify function decorator(s) to apply to each node
    nodes:
      - inputs: ["params:model", train_df, "params:cols_features", "params:col_target"]
        func: sklearn_demo.train_model
        outputs: model

      - inputs: [model, test_df, "params:cols_features"]
        func: sklearn_demo.run_inference
        outputs: pred_df
```

#### Configure Kedro run config using `RUN_CONFIG` key
  - Optionally run nodes in parallel
  - Optionally run only missing nodes (skip tasks which have already been run to resume pipeline using the intermediate data files or databases.)
  Note: You can use Kedro CLI to overwrite these run configs.

```yaml
# parameters.yml

RUN_CONFIG:
  pipeline_name: __default__
  runner: SequentialRunner # Set to "ParallelRunner" to run in parallel
  only_missing: False # Set True to run only missing nodes
  tags: # None
  node_names: # None
  from_nodes: # None
  to_nodes: # None
  from_inputs: # None
  load_versions: # None
```

#### Define Kedro hooks using `HOOKS` key

```yaml
# parameters.yml

HOOKS:
  - =: pipelinex.MLflowBasicLoggerHook # Configure and log duration time for the pipeline 
    enable_mlflow: True # Enable configuring and logging to MLflow
    uri: sqlite:///mlruns/sqlite.db
    experiment_name: experiment_001
    artifact_location: ./mlruns/experiment_001
    offset_hours: 0 # Specify the offset hour (e.g. 0 for UTC/GMT +00:00) to log in MLflow

  - =: pipelinex.MLflowArtifactsLoggerHook # Log artifacts of specified file paths and dataset names
    enable_mlflow: True # Enable logging to MLflow
    filepaths_before_pipeline_run: # Optionally specify the file paths to log before pipeline is run
      - conf/base/parameters.yml
    datasets_after_node_run: # Optionally specify the dataset names to log after the node is run
      - model
    filepaths_after_pipeline_run: # None  # Optionally specify the file paths to log after pipeline is run

  - =: pipelinex.MLflowOutputsLoggerHook # Log output datasets of (list of) float, int, and str classes
    enable_mlflow: True # Enable logging to MLflow

  - =: pipelinex.MLflowTimeLoggerHook # Log duration time to run each node (task)
    enable_mlflow: True # Enable logging to MLflow

  - =: pipelinex.AddTransformersHook # Add transformers
    transformers: 
      =: pipelinex.MLflowIOTimeLoggerTransformer # Log duration time to load and save each dataset
      enable_mlflow: True
```


### Here are the options configurable in `catalog.yml`:

- `HatchDict` features available
- Optionally enable caching using `cached` key set to True if you do not want Kedro to load the data from disk/database which were in the memory. ([`kedro.io.CachedDataSet`](https://kedro.readthedocs.io/en/latest/kedro.io.CachedDataSet.html#kedro.io.CachedDataSet) is used under the hood.)

The complete example project is available [here](https://github.com/Minyus/pipelinex_sklearn).


## Use with PyTorch

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

## Use with PyTorch Ignite

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

## Use with OpenCV

A challenge of image processing is that the parameters and algorithms that work with an image often do not work with another image. You will want to output intermediate images from each image processing pipeline step for visual check during development, but you will not want to output all the intermediate images to save time and disk space in production.

Wrappers of OpenCV and `ImagesLocalDataSet` are the solution. You can concentrate on developping your image processing pipeline for an image (3-D or 2-D numpy array), and it will run for all the images in a folder.

If you are devepping an image processing pipeline consisting of 5 steps and you have 10 images, for example, you can check 10 generated images in each of 5 folders, 50 images in total, during development.

## Use with PyTorch Lightning

(To-do)

## Use with TensorFlow/Keras

(To Do.)

## Use with Docker container

### Build Docker image

```bash
# docker build -t pipelinex:3.7.7-slim -f dockerfiles/dockerfile .
```

### Use with Docker container

```bash
# docker run -it --name pipelinex pipelinex:3.7.7-slim /bin/bash
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

## Contributors wanted!

Please see [CONTRIBUTING.md](https://github.com/Minyus/pipelinex/blob/master/CONTRIBUTING.md) for details.

## Author

Yusuke Minami

- [GitHub](https://github.com/Minyus)
- [Linkedin](https://www.linkedin.com/in/yusukeminami/)
- [Twitter](https://twitter.com/Minyus86)

## Contributors

- [shibuiwilliam](https://github.com/shibuiwilliam)
- [MarchRaBBiT](https://github.com/MarchRaBBiT)
