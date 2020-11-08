from functools import wraps
import pandas as pd
from typing import Callable, List, Union

import logging

log = logging.getLogger(__name__)


def log_df_summary(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(df, *args, **kwargs):
        log = logging.getLogger(__name__)
        if isinstance(df, pd.DataFrame):
            shape_before = df.shape
            cols_before = df.columns.to_list()
            log.info(
                "{}".format(
                    dict(
                        DF_shape_before="{}".format(shape_before),
                        func="{}".format(getattr(func, "__qualname__", repr(func))),
                    )
                )
            )
        df = func(df, *args, **kwargs)
        if isinstance(df, pd.DataFrame):
            shape_after = df.shape
            cols_after = df.columns.to_list()
            cols_added = [col for col in cols_after if col not in cols_before]
            cols_removed = [col for col in cols_before if col not in cols_after]
            log.info(
                "{}".format(
                    dict(
                        DF_shape_after="{}".format(shape_after),
                        Columns_added="{}".format(cols_added),
                        Columns_removed="{}".format(cols_removed),
                    )
                )
            )
        return df

    return wrapper


def df_set_index(
    cols: Union[List[str], str],
) -> Callable:
    """ decorator with arguments """
    if not isinstance(cols, list):
        cols = [cols]

    def decorator(func: Callable) -> Callable:
        """ decorator without arguments """

        @wraps(func)
        def wrapper(df, parameters, *args, **kwargs):
            if isinstance(df, pd.DataFrame):
                for col in cols:
                    if col not in df.columns:
                        log.warning("Could not find column: ".format(col))
                cols_ = [col for col in cols if col in df.columns]
                df.set_index(keys=cols_, inplace=True)
            df = func(df, parameters, *args, **kwargs)
            if isinstance(df, pd.DataFrame):
                df.reset_index(inplace=True)
            return df

        return wrapper

    return decorator


def total_seconds_to_datetime(
    cols: Union[List[str], str], origin: str = "1970-01-01"
) -> Callable:
    """ decorator with arguments """
    if not isinstance(cols, list):
        cols = [cols]

    def decorator(func: Callable) -> Callable:
        """ decorator without arguments """

        @wraps(func)
        def wrapper(df, parameters, *args, **kwargs):
            if isinstance(df, pd.DataFrame):
                for col in cols:
                    if col not in df.columns:
                        log.warning("Could not find column: ".format(col))
                cols_ = [col for col in cols if col in df.columns]
                for col in cols_:
                    df.loc[:, col] = pd.to_datetime(
                        df[col], unit="s", origin=pd.Timestamp(origin)
                    )
            df = func(df, parameters, *args, **kwargs)
            if isinstance(df, pd.DataFrame):
                for col in cols_:
                    df.loc[:, col] = (df[col] - pd.Timestamp(origin)).dt.total_seconds()
            return df

        return wrapper

    return decorator
