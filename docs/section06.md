## Kedro context to use YAML files more conveniently

Using [pipelinex.FlexibleContext](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/framework/context/flexible_context.py), you can use the YAML files more conveniently 

### Options available in `parameters.yml`

#### Define Kedro pipelines 
  
  You can define the inter-task dependency (DAG) for Kedro pipelines in `parameters.yml` using `PIPELINES` key.
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

#### `HatchDict` feature

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

### Options available in `catalog.yml`

#### Enable caching

Enable caching using `cached` key set to True if you do not want Kedro to load the data from disk/database which were in the memory. ([`kedro.io.CachedDataSet`](https://kedro.readthedocs.io/en/latest/kedro.io.CachedDataSet.html#kedro.io.CachedDataSet) is used under the hood.)

#### `HatchDict` feature

You can use `HatchDict` feature in `catalog.yml`.

