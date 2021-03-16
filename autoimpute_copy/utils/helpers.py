"""Helper functions used throughout other methods in automipute.utils."""

import warnings
import numpy as np
import pandas as pd

def _sq_output(data, cols, square=False):
    """Private method to turn unlabeled data into a DataFrame."""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=cols)
    if square:
        data.index = data.columns
    return data

def _index_output(data, index):
    """Private method to transform data to DataFrame and set the index."""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, index=index)
    return data

def _nan_col_dropper(data):
    """Private method to drop columns w/ all missing values from DataFrame."""
    cb = set(data.columns.tolist())
    data.dropna(axis=1, how='all', inplace=True)
    ca = set(data.columns.tolist())
    cdiff = cb.difference(ca)
    if cdiff:
        wrn = f"{cdiff} dropped from DataFrame because all rows missing."
        warnings.warn(wrn)
    return data, cdiff

def _one_hot_encode(X, fit_x = None):
    """Private method to handle one hot encoding for categoricals."""
    cats = X.select_dtypes(include=(np.object,)).columns.size
    if cats > 0:
        if fit_x is not None:
            N = X.shape[0]

            dum_x = pd.concat([X, fit_x[X.columns]], sort = False, ignore_index = True)
            dum_X = pd.get_dummies(dum_x, drop_first=True)

            X = dum_X.loc[:N-1]
        else:
            X = pd.get_dummies(X, drop_first=True)

    return X
