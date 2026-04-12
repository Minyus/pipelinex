from multiprocessing.reduction import ForkingPickler
from pathlib import Path

from pipelinex import MLflowDataSet


def test_pickle():
    data_set = MLflowDataSet(dataset="pkl")
    ForkingPickler.dumps(data_set)


def test_init_dataset_sets_default_filepath_suffix():
    data_set = MLflowDataSet(dataset="pkl", dataset_name="unit_test_artifact")

    data_set._init_dataset()

    assert Path(data_set.filepath).name == "unit_test_artifact.pkl"
    assert data_set._dataset is not None


if __name__ == "__main__":
    test_pickle()
