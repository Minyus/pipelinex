## Introduction

PipelineX is a Python package designed to make Machine Learning projects efficient with modular, reusable, and easy-to-use features for experimentation.

Please refer [here](https://github.com/Minyus/Python_Packages_for_Pipeline_Workflow) to find out how PipelineX differs from other pipeline/workflow packages: Airflow, Luigi, Gokart, Metaflow, and Kedro.

PipelineX provides the following options which can be used separately or together.

- `HatchDict` option which provides [enhancements for YAML/JSON](https://github.com/Minyus/pipelinex#pythonic-enhanced-yamljson) useful for parameter management summarized as follows.

  - Import-less Python object: Include (nested) Python classes and functions in a YAML/JSON file
  - Anchor-less aliasing: Look up another key in the same YAML/JSON file
  - Python expression in YAML/JSON files

- Kedro context to define Kedro pipelines in a YAML file with more options

- Integration of Kedro with [MLflow](https://github.com/mlflow/mlflow) as Kedro DataSets and Hooks. 
  Note: You do not need to install MLflow if you do not use.

- Integration of Kedro with the additional Python packages as Kedro DataSets, Hooks, and wrappers. 
  - <[PyTorch](https://github.com/pytorch/pytorch)>
  - <[Ignite](https://github.com/pytorch/ignite)>
  - <[Pandas](https://github.com/pandas-dev/pandas)>
  - <[OpenCV](https://github.com/skvark/opencv-python)>
  - <[Memory Profiler](https://github.com/pythonprofilers/memory_profiler)>
  - <[NVIDIA Management Library](https://github.com/gpuopenanalytics/pynvml)>
  Note: You do not need to install Python packages you do not use.


