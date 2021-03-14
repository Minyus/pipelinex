from multiprocessing.reduction import ForkingPickler

from pipelinex import MLflowDataSet


def test_pickle():
    data_set = MLflowDataSet(dataset="pkl")
    ForkingPickler.dumps(data_set)


if __name__ == "__main__":
    test_pickle()
