import pandas as pd
from kedro.extras.datasets.text.text_dataset import TextDataSet

from pipelinex import FixedWidthCSVDataSet


def test_simple():
    df = pd.DataFrame(
        {"col1": [0.123456789, 0.5], "col2": [123, -1], "col3": ["foo", "foobar"]}
    )
    filepath = "/tmp/test.csv"
    data_set = FixedWidthCSVDataSet(filepath=filepath)
    data_set.save(df)
    text_data_set = TextDataSet(filepath=filepath)
    reloaded = text_data_set.load()
    print("\n" + reloaded)
    # assert df.equals(reloaded)
