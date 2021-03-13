## Story behind PipelineX

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

