import warnings


def test_import_pipelinex_exports_target_optional_surfaces():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import pipelinex

    assert hasattr(pipelinex, "FlexiblePipeline")
    assert hasattr(pipelinex, "MLflowDataSet")
    assert hasattr(pipelinex, "DfTrainTestSplit")
    assert all("kedro/mlflow" not in str(w.message) for w in caught)
    assert all("sklearn ops" not in str(w.message) for w in caught)
