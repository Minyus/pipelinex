import pandas as pd

from pipelinex import DfTrainTestSplit


def test_df_train_test_split_returns_two_frame_split():
    df = pd.DataFrame({"feature": [1, 2, 3, 4], "target": [0, 1, 0, 1]})

    train_df, test_df = DfTrainTestSplit(test_size=0.5, shuffle=False)(df)

    assert train_df["feature"].tolist() == [1, 2]
    assert test_df["feature"].tolist() == [3, 4]
    assert train_df["target"].tolist() == [0, 1]
    assert test_df["target"].tolist() == [0, 1]
