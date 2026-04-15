import json
import subprocess
import sys
from importlib import import_module


def test_import_pipelinex_exports_target_optional_surfaces():
    code = r"""
import importlib
import json
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    pipelinex = importlib.import_module("pipelinex")

payload = {
    "exports": {
        "FlexiblePipeline": hasattr(pipelinex, "FlexiblePipeline"),
        "MLflowDataSet": hasattr(pipelinex, "MLflowDataSet"),
        "FixedWidthCSVDataSet": hasattr(pipelinex, "FixedWidthCSVDataSet"),
        "DfTrainTestSplit": hasattr(pipelinex, "DfTrainTestSplit"),
    },
    "warnings": [str(w.message) for w in caught],
}
print(json.dumps(payload))
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout.strip())

    assert payload["exports"] == {
        "FlexiblePipeline": True,
        "MLflowDataSet": True,
        "FixedWidthCSVDataSet": True,
        "DfTrainTestSplit": True,
    }
    assert all("kedro/mlflow" not in warning for warning in payload["warnings"])
    assert all("sklearn ops" not in warning for warning in payload["warnings"])
    assert all("pandas datasets" not in warning for warning in payload["warnings"])


def test_fixed_width_csv_dataset_uses_modern_kedro_dataset_when_available():
    from pipelinex import FixedWidthCSVDataSet

    modern_csv_dataset = import_module("kedro_datasets.pandas").CSVDataset

    assert FixedWidthCSVDataSet.__mro__[1] is modern_csv_dataset
    assert FixedWidthCSVDataSet.__mro__[1].__module__.startswith("kedro_datasets")
