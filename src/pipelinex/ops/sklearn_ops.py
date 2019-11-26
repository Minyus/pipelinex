import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer
import pandas as pd
import numpy as np


def extract_from_df(df, cols, target_col):
    if isinstance(df, pd.DataFrame):
        cols = cols or df.columns
        target_col = target_col
        X = df[cols].values
        y = df[target_col].values if target_col else None
    else:
        X = df
        y = None
    if y:
        return dict(X=X, y=y)
    else:
        return dict(X=X), cols


class DfBaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, target_col=None, **kwargs):
        super().__init__(**kwargs)
        self.cols = cols
        self.target_col = target_col

    def fit(self, df):
        d, cols = extract_from_df(df, self.cols, self.target_col)
        return super().fit(**d)

    def transform(self, df):
        d, cols = extract_from_df(df, self.cols, self.target_col)
        out_arr = super().transform(**d)
        if isinstance(df, pd.DataFrame):
            df.loc[:, cols] = out_arr
            return df
        else:
            return out_arr

    def fit_transform(self, df):
        d, cols = extract_from_df(df, self.cols, self.target_col)
        out_arr = super().fit_transform(**d)
        if isinstance(df, pd.DataFrame):
            df.loc[:, cols] = out_arr
            return df
        else:
            return out_arr

    def __call__(self, df):
        return self.fit_transform(df), self


class FlexibleQuantileTransformer(QuantileTransformer):
    def __init__(self, zero_to_zero=False, **kwargs):
        self.zero_to_zero = zero_to_zero
        super().__init__(**kwargs)

    def _keep_zero(self, X):
        zeros_arr = np.zeros((1, X.shape[1]))
        offset_arr = super().transform(zeros_arr)
        X = X - offset_arr
        return X

    def transform(self, X):
        out_X = super().transform(X)
        if self.zero_to_zero:
            out_X = self._keep_zero(out_X)
        return out_X

    def fit_transform(self, X):
        out_X = super().fit_transform(X)
        if self.zero_to_zero:
            out_X = self._keep_zero(out_X)
        return out_X


class DfQuantileTransformer(DfBaseTransformer, FlexibleQuantileTransformer):
    pass


class DfTrainTestSplit:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, df, *argsignore, **kwargsignore):
        return sklearn.model_selection.train_test_split(df, **self.kwargs)
