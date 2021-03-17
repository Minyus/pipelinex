## Flex-Kedro: Kedro plugin for flexible config

[API document](https://pipelinex.readthedocs.io/en/latest/pipelinex.flex_kedro.html)

Flex-Kedro provides more options to configure Kedro projects flexibly and thus quickly by KFlex-Kedro-Pipeline and Flex-Kedro-Context features.

### Flex-Kedro-Pipeline: Kedro plugin for quicker pipeline set up 

If you want to define Kedro pipelines quickly, you can consider to use `pipelinex.FlexiblePipeline` instead of `kedro.pipeline.Pipeline`. 
`pipelinex.FlexiblePipeline` adds the following options to `kedro.pipeline.Pipeline`.

#### Dict for nodes

To define each node, dict can be used instead of `kedro.pipeline.node`. 

  Example:

  ```python
  pipelinex.FlexiblePipeline(
      nodes=[dict(func=task_func1, inputs="my_input", outputs="my_output")]
  )
  ```

  will be equivalent to:

  ```python
  kedro.pipeline.Pipeline(
      nodes=[
          kedro.pipeline.node(func=task_func1, inputs="my_input", outputs="my_output")
      ]
  )
  ```

#### Sequential nodes

For sub-pipelines consisting of nodes of only single input and single output, you can optionally use Sequential API similar to PyTorch (`torch.nn.Sequential`) and Keras (`tf.keras.Sequential`)

  Example:

  ```python
  pipelinex.FlexiblePipeline(
      nodes=[
          dict(
              func=[task_func1, task_func2, task_func3],
              inputs="my_input",
              outputs="my_output",
          )
      ]
  )
  ```

  will be equivalent to:

  ```python
  kedro.pipeline.Pipeline(
      nodes=[
          kedro.pipeline.node(
              func=task_func1, inputs="my_input", outputs="my_output__001"
          ),
          kedro.pipeline.node(
              func=task_func2, inputs="my_output__001", outputs="my_output__002"
          ),
          kedro.pipeline.node(
              func=task_func3, inputs="my_output__002", outputs="my_output"
          ),
      ]
  )
  ```

#### Decorators without using the method

- Optionally specify the Python function decorator(s) to apply to multiple nodes under the pipeline using `decorator` argument instead of using [`decorate`](https://kedro.readthedocs.io/en/stable/kedro.pipeline.Pipeline.html#kedro.pipeline.Pipeline.decorate) method of `kedro.pipeline.Pipeline`.

  Example:

  ```python
  pipelinex.FlexiblePipeline(
      nodes=[
          kedro.pipeline.node(func=task_func1, inputs="my_input", outputs="my_output")
      ],
      decorator=[task_deco, task_deco],
  )
  ```

  will be equivalent to:

  ```python
  kedro.pipeline.Pipeline(
      nodes=[
          kedro.pipeline.node(func=task_func1, inputs="my_input", outputs="my_output")
      ]
  ).decorate(task_deco, task_deco)

  ```

- Optionally specify the default python module (path of .py file) if you do not want to repeat the same (deep and/or long) Python module (e.g. `foo.bar.my_task1`, `foo.bar.my_task2`, etc.)


### Flex-Kedro-Context: Kedro plugin for YAML lovers

If you want to take advantage of YAML more than Kedro supports, you can consider to use 
`pipelinex.FlexibleContext` instead of `kedro.framework.context.KedroContext`. 
`pipelinex.FlexibleContext` adds preprocess of `parameters.yml` and `catalog.yml` to `kedro.framework.context.KedroContext` to provide flexibility.
This option is for YAML lovers only. 
If you don't like YAML very much, skip this one.

#### Define Kedro pipelines in `parameters.yml`
  
You can define the inter-task dependency (DAG) for Kedro pipelines in `parameters.yml` using `PIPELINES` key. To define each Kedro pipeline, you can use the `kedro.pipeline.Pipeline` or its variant such as `pipelinex.FlexiblePipeline` as shown below.

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

#### Configure Kedro run config in `parameters.yml`

You can specify the run config in `parameters.yml` using `RUN_CONFIG` key instead of specifying the args for `kedro run` command for every run. 

You can still set the args for `kedro run` to overwrite. 

In addition to the args for `kedro run`, you can opt to run only missing nodes (skip tasks which have already been run to resume pipeline using the intermediate data files or databases.) by `only_missing` key.


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

#### Use `HatchDict` feature in `parameters.yml`

You can use `HatchDict` feature in `parameters.yml`.

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

#### Enable caching for Kedro DataSets in `catalog.yml`

Enable caching using `cached` key set to True if you do not want Kedro to load the data from disk/database which were in the memory. ([`kedro.io.CachedDataSet`](https://kedro.readthedocs.io/en/latest/kedro.io.CachedDataSet.html#kedro.io.CachedDataSet) is used under the hood.)

#### Use `HatchDict` feature in `catalog.yml`

You can use `HatchDict` feature in `catalog.yml`.


