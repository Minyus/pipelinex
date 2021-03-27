## PipelineX Overview

PipelineX is a Python package to build ML pipelines for experimentation with Kedro, MLflow, and more

PipelineX provides the following options which can be used independently or together.

- HatchDict: Python in YAML/JSON

  `HatchDict` is a Python dict parser that enables you to include Python objects in YAML/JSON files. 

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

Please refer [here](https://github.com/Minyus/Python_Packages_for_Pipeline_Workflow) to find out how PipelineX differs from other pipeline/workflow packages: Airflow, Luigi, Gokart, Metaflow, and Kedro.


