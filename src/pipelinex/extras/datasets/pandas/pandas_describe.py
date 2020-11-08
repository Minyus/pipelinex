from typing import Any, Dict
import pandas as pd
from .csv_local import CSVLocalDataSet


class PandasDescribeDataSet(CSVLocalDataSet):
    """``PandasDescribeDataSet`` saves output of ``df.describe``."""

    def __init__(self, *args, describe_args: Dict[str, Any] = {}, **kwargs) -> None:
        """Creates a new instance of ``PandasDescribeDataSet`` pointing to a concrete
        filepath.

        Args:
            args: Positional arguments for ``CSVLocalDataSet``
            describe_args: Arguments passed on to ``df.describe``.
                See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html for details.
            kwargs: Keyword arguments for ``CSVLocalDataSet``

        """
        super().__init__(*args, **kwargs)
        self._describe_args = describe_args

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            save_args=self._save_args,
            describe_args=self._describe_args,
            version=self._version,
        )

    def _load(self) -> Any:
        """ loading is not supported. """
        return None

    def _save(self, data: pd.DataFrame) -> None:
        df = data.describe(**self._describe_args)
        df.index.name = "Statistics"

        df.reset_index(inplace=True)
        super()._save(df)
