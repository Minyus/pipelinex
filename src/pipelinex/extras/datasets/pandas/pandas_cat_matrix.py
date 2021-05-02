import logging
from typing import Any, Dict

import pandas as pd

from .csv_local import CSVLocalDataSet

log = logging.getLogger(__name__)


class PandasCatMatrixDataSet(CSVLocalDataSet):
    """``PandasDescribeDataSet`` saves output of ``df.describe``."""

    def __init__(self, *args, describe_args: Dict[str, Any] = {}, **kwargs) -> None:
        """Creates a new instance of ``PandasCatMatrixDataSet`` pointing to a concrete
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
        """loading is not supported."""
        return None

    def _save(self, df: pd.DataFrame) -> None:
        cols = self._save_args.pop("cols", None) or df.columns.to_list()
        df = _get_cat_mat_df(df, cols)
        df.reset_index(inplace=True)
        super()._save(df)


def _get_cat_mat_df(df, cols_analyze):
    cols_analyze = [col for col in df.columns if col in cols_analyze]
    n_cols = len(cols_analyze)
    rg = range(n_cols)

    nunique_list = [df[c0].nunique(dropna=False) for c0 in cols_analyze]
    nunique_sr = pd.Series(nunique_list, name="nunique")
    nunique_df = pd.DataFrame(nunique_sr)
    log.info("nunique_df: {}".format(nunique_df))

    dep_mat = [
        [
            (
                df.groupby(c1)[c0]
                .nunique(dropna=False)
                .mean()
                # 1
                # - df.groupby(c1)[c0].nunique(dropna=False).mean()
                # / df[c0].nunique(dropna=False)
            )
            for c0 in cols_analyze
        ]
        for c1 in cols_analyze
    ]
    dep_mat_df = pd.DataFrame(dep_mat, columns=cols_analyze, index=cols_analyze)
    log.info("dep_mat_df: \n{}".format(dep_mat_df))

    # dep_diff_mat = [
    #     [(abs(dep_mat[i0][i1] - dep_mat[i1][i0]) if (i0 < i1) else 1.0) for i0 in rg]
    #     for i1 in rg
    # ]
    # dep_diff_mat_df = pd.DataFrame(
    #     dep_diff_mat, columns=cols_analyze, index=cols_analyze
    # )
    # log.info("dep_diff_mat_df: \n{}".format(dep_diff_mat_df))

    return dep_mat_df

    # dep_diff_mat_2darr = np.array(dep_diff_mat)
    # indices = np.unravel_index(np.argsort(dep_diff_mat_2darr.ravel()), (n_cols, n_cols))
    # dep_dict = {
    #     dep_diff_mat[i0][i1]: (cols_analyze[i0], cols_analyze[i1])
    #     for i0 in rg
    #     for i1 in rg
    # }
    # pair_list = [dep_dict.get(dd) for dd in sorted(dep_dict.keys()) if dd < 1.0]
    # log.info("dep_diff_sorted_pair_list: {}".format(pair_list))
    #
    # dep_info = dict(
    #     dep_mat_df=dep_mat_df,
    #     # dep_diff_mat_df=dep_diff_mat_df,
    #     # pair_list=pair_list,
    # )
    # return dep_info
