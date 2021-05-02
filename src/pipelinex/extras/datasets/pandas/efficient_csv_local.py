import copy
import logging
from typing import Any, Dict, Union

import numpy as np

from .csv_local import CSVLocalDataSet

log = logging.getLogger(__name__)


class EfficientCSVLocalDataSet(CSVLocalDataSet):
    """ """

    DEFAULT_LOAD_ARGS = dict(
        engine="c", keep_default_na=False, na_values=[""], skiprows=0
    )  # type: Dict[str, Any]
    DEFAULT_PREVIEW_ARGS = dict(nrows=None, low_memory=False)  # type: Dict[str, Any]

    def __init__(
        self,
        *args,
        preview_args: Dict[str, Any] = None,
        margin: float = 100.0,
        verbose: Union[bool, int] = True,
        **kwargs
    ) -> None:
        """Creates a new instance of ``PandasDescribeDataSet`` pointing to a concrete
        filepath.

        Args:
            args: Positional arguments for ``CSVLocalDataSet``
            preview_args: Arguments passed on to ``df.describe``.
                See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html for details.
            kwargs: Keyword arguments for ``CSVLocalDataSet``

        """
        super().__init__(*args, **kwargs)

        self._preview_args = copy.deepcopy(self.DEFAULT_PREVIEW_ARGS)
        if preview_args is not None:
            self._preview_args.update(preview_args)
        self._margin = margin
        self._verbose = verbose

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            preview_args=self._preview_args,
            load_args=self._load_args,
            save_args=self._save_args,
            version=self._version,
        )

    def _load(self) -> Any:

        load_args = self._load_args.copy()
        float_dtype = self._load_args.pop("float_dtype", "float16")
        assert float_dtype in {"float16", "float32", "float64"}
        int_dtype = self._load_args.pop("int_dtype", "int8")
        assert int_dtype in {"int8", "int16", "int32", "int64"}

        nrows = self._preview_args.get("nrows")

        self._load_args.update(self._preview_args)
        df = super()._load()

        dtypes_dict = _get_necessary_dtypes(
            df,
            margin=self._margin,
            float_dtype=float_dtype,
            int_dtype=int_dtype,
            verbose=self._verbose,
        )
        if nrows:
            load_args["dtype"] = dtypes_dict
            self._load_args = load_args
            df = super()._load()
        else:
            df = df.astype(dtype=dtypes_dict)
        return df


def dict_val_replace_except(
    d,  # type: dict
    to_except,  # type: Any
    new_value,  # type: Any
):
    return {k: (new_value if v != to_except else v) for k, v in d.items()}


def dict_string_val_prefix(
    d,  # type: dict
    prefix,  # type: Any
):
    return {k: (prefix + v) for k, v in d.items()}


def _get_necessary_dtypes(
    df, margin=100, float_dtype="float16", int_dtype="int8", verbose=True
):
    dtypes_dict = df.dtypes.to_dict()
    # dtypes_dict = {col: dtype.name for col, dtype in dtypes_dict.items()}

    for col, dtype in dtypes_dict.items():
        if dtype == np.object:
            dtypes_dict[col] = "object"  # np.object
        if dtype == np.float64:
            if float_dtype == "float16" and (
                (df[col].max() + margin) < np.finfo(np.float16).max
                and (df[col].min() - margin) > np.finfo(np.float16).min
            ):
                dtypes_dict[col] = "float16"  # np.float16
            elif float_dtype in {"float32", "float16"} and (
                (df[col].max() + margin) < np.finfo(np.float32).max
                and (df[col].min() - margin) > np.finfo(np.float32).min
            ):
                dtypes_dict[col] = "float32"  # np.float32
            else:
                dtypes_dict[col] = "float64"  # np.float64

        if dtype == np.int64:
            if int_dtype == "int8" and (
                (df[col].max() + margin) < np.iinfo(np.int8).max
                and (df[col].min() - margin) > np.iinfo(np.int8).min
            ):
                dtypes_dict[col] = "int8"  # np.int8
            if int_dtype == "int16" and (
                (df[col].max() + margin) < np.iinfo(np.int16).max
                and (df[col].min() - margin) > np.iinfo(np.int16).min
            ):
                dtypes_dict[col] = "float16"  # np.float16
            elif int_dtype in {"int32", "int16"} and (
                (df[col].max() + margin) < np.iinfo(np.int32).max
                and (df[col].min() - margin) > np.iinfo(np.int32).min
            ):
                dtypes_dict[col] = "int32"  # np.int32
            else:
                dtypes_dict[col] = "int64"  # np.int64

    if verbose:

        for dtype in [
            "float16",
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "object",
        ]:
            log.info(
                (
                    "{} is the minimum dtype for columns: \n{}".format(
                        dtype, [col for col, t in dtypes_dict.items() if t == dtype]
                    )
                )
            )

    return dtypes_dict
