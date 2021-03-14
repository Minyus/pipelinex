## Introduction to PipelineX

PipelineX is a Python package designed to make Machine Learning projects efficient with modular, reusable, and easy-to-use features for experimentation.

Please refer [here](https://github.com/Minyus/Python_Packages_for_Pipeline_Workflow) to find out how PipelineX differs from other pipeline/workflow packages: Airflow, Luigi, Gokart, Metaflow, and Kedro.

PipelineX provides the following options which can be used independently or together.

- HatchDict: Python in YAML/JSON

  `HatchDict` provides extension for YAML/JSON summarized as follows.

  - Import-less Python object: Include (nested) Python classes and functions in a YAML/JSON file
  - Anchor-less aliasing: Look up another key in the same YAML/JSON file
  - Python expression in YAML/JSON files

  Note: `HatchDict` can be used with or without Kedro.

- Flex-Kedro: Kedro plugin for flexible config

  - Flex-Kedro-Pipeline: Kedro plugin for quicker pipeline set up 

  - Flex-Kedro-Context: Kedro plugin for YAML lovers

- MLflow-on-Kedro: Kedro plugin for MLflow users

  `MLflow-on-Kedro` provides integration of Kedro with [MLflow](https://github.com/mlflow/mlflow) with Kedro DataSets and Hooks.

  Note: You do not need to install MLflow if you do not use.

- Kedro-Extras: Kedro plugin to use various Python packages 

  `Kedro-Extras` provides Kedro DataSets, decorators, and wrappers to use various Python packages such as: 

  - <[PyTorch](https://github.com/pytorch/pytorch)>
  - <[Ignite](https://github.com/pytorch/ignite)>
  - <[Pandas](https://github.com/pandas-dev/pandas)>
  - <[OpenCV](https://github.com/skvark/opencv-python)>
  - <[Memory Profiler](https://github.com/pythonprofilers/memory_profiler)>
  - <[NVIDIA Management Library](https://github.com/gpuopenanalytics/pynvml)>

  Note: You do not need to install Python packages you do not use.


