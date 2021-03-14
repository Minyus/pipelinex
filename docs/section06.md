## Flex-Kedro: Kedro plugin for flexible config

[pipelinex.flex_kedro API document](https://pipelinex.readthedocs.io/en/latest/pipelinex.flex_kedro.html)

Flex-Kedro provides more options to configure Kedro projects flexibly and thus quickly by KFlex-Kedro-Pipeline and Flex-Kedro-Context features.

### Flex-Kedro-Pipeline: Kedro plugin for quicker pipeline set up 

If you want to define Kedro pipelines quickly, you can consider to use `pipelinex.FlexiblePipeline` instead of `kedro.pipeline.Pipeline`. `pipelinex.FlexiblePipeline` adds the following options to `kedro.pipeline.Pipeline`. 
- Optionally specify the default Python module (path of .py file) if you want to omit the module name
- Optionally specify the Python function decorator to apply to each node
- For sub-pipelines consisting of nodes of only single input and single output, you can optionally use Sequential API similar to PyTorch (`torch.nn.Sequential`) and Keras (`tf.keras.Sequential`)

An example is available in the Flex-Kedro-Context section.

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

#### Configure Kedro run config using `RUN_CONFIG` key in `parameters.yml`

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


