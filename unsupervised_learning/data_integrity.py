# parsing modules
import json
import re
from urllib.parse import unquote

# Pipeline imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline

import pandas as pd
import numpy as np


class DataIntegrityTransformer(BaseEstimator, TransformerMixin):
    """
    Explicitly convert lists of variables into date, numeric, categorical, and boolean types respectively
    Keyword Args:
        date_vars - List of variables that should be converted to datetime format
        numeric_vars - List of variables that should be converted to numeric format
        categorical_vars - List of variables that should be converted to object
        bool_vars - List of variables that should be converted to boolean

    Returns:
        Original dataframe with conversions applied

    Intent:
        Ensure variables are stored in right format before being operated on by parsing and FE transformers
    """

    # constructor assigns variable lists
    def __init__(self, date_vars, numeric_vars, categorical_vars):
        self.date_vars = date_vars
        self.numeric_vars = numeric_vars
        self.categorical_vars = categorical_vars
        # self.bool_vars = bool_vars

    # fit does nothing
    def fit(self, X, y=None):
        return self

    # transform looks for common columns in provided df (X) and does explicit conversion
    def transform(self, X, y=None):
        # intersection is used to find common columns between list and df, accomodates changes in columns without breaking pipeline
        if self.date_vars:
            good_date_keys = X.columns.intersection(self.date_vars)
            X[good_date_keys] = X[good_date_keys].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S.%f')

        if self.numeric_vars:
            good_num_keys = X.columns.intersection(self.numeric_vars)
            X[good_num_keys] = X[good_num_keys].apply(pd.to_numeric)

        # if self.bool_vars:
        #   good_bool_keys = X.columns.intersection(self.bool_vars)
        #   X[good_bool_keys] = X[good_bool_keys].apply(lambda x: x.astype(bool))

        if self.categorical_vars:
            good_cat_keys = X.columns.intersection(self.categorical_vars)
            X[good_cat_keys] = X[good_cat_keys].apply(lambda x: x.astype(str))

        # good_cat_keys = X.columns.intersection(self.categorical_vars)
        # X[good_cat_keys] = X[good_cat_keys].apply(lambda x: x.astype(str))
        # X.loc[:,good_cat_keys] = X[good_cat_keys].to_string()

        return X
