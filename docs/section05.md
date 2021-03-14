## Introduction to Kedro

### Why the unified data interface framework is needed

Machine Learning projects involves with loading and saving various data in various ways such as:

- files in local/network file system, Hadoop Distributed File System (HDFS), Amazon S3, Google Cloud Storage
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


### Kedro overview

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
- <[Kedro's document](https://kedro.readthedocs.io/en/stable/)>
- <[YouTube playlist: Writing Data Pipelines with Kedro](https://www.youtube.com/playlist?list=PLTU89LAWKRwEdiDKeMOU2ye6yU9Qd4MRo)>
- <[Python Packages for Pipeline/Workflow](https://github.com/Minyus/Python_Packages_for_Pipeline_Workflow)>

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


