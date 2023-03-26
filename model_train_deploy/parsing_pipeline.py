# parsing modules
import json
import re
from urllib.parse import unquote
import itertools

# Pipeline imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline

import pandas as pd
import numpy as np


# feature selecting pipeline
class FeatureSelector(BaseEstimator, TransformerMixin):
    # constructor
    def __init__(self, feature_names):
        self.feature_names = feature_names

    """
    Return dataframe with select features only

    Keyword Args:
        feature_names - List of column names that should be selected

    Input:
        X - Dataframe which includes features to be selected

    Returns:
        Dataframe with columns from `feature_names` only

    """

    # fit does nothing
    def fit(self, X, y=None):
        return self

    # transform that describes what FeatureSelector will do (just return columns specified)
    def transform(self, X, y=None):
        good_keys = X.columns.intersection(self.feature_names)
        return X[good_keys]


class FeatureDropper(BaseEstimator, TransformerMixin):
    """
    Removes features from dataframe

    Keyword Args:
        feature_to_drop - List of column names that should be dropped

    Input:
        X - Dataframe which includes features to be dropped

    Returns:
        Dataframe with columns from `feature_to_drop` removed

    """

    # constructor
    def __init__(self, feature_to_drop):
        self.feature_to_drop = feature_to_drop

    # fit does nothing
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        good_keys = X.columns.intersection(self.feature_to_drop)
        X = X.drop(good_keys, axis=1)
        return X


# writing pipeline class that will parse categorical features
class ParsingTransformer(BaseEstimator, TransformerMixin):
    """
    Adds column names to input numpy array, parses tmx variables to remove HTTP markup syntax, and cleans up categorical variables with regex

    Keyword Args:
        df_names - List of column names for initial training set, applied if input is a numpy array
        tmx_unclean_vars - List of TMX variables with unclean data

    Input:
        X - Pandas dataframe or numpy array with all columns

    Returns:
        Pandas dataframe with tmx variables cleaned and column names attached

    """

    def __init__(self, df_names=None, tmx_unclean_vars=None, exact_match_vars=None, regex_match_vars=None):
        self.df_names = df_names
        self.tmx_unclean_vars = tmx_unclean_vars
        self.exact_match_vars = exact_match_vars
        self.regex_match_vars = regex_match_vars

    # cleaning json columns for parsing
    def clean_json(self, obj):
        try:
            obj = obj.replace("\'", "\"")
        except AttributeError:
            obj = "{}"
        json_val = json.loads(obj)

        return json_val

    def parse_json_to_column(self, json_val):
        # a = pd.json_normalize(temp)
        return pd.Series(json_val)

    def parse_dirty_json(self, obj):
        json_val = self.clean_json(obj)
        return self.parse_json_to_column(json_val)

    def parse_url_seps(self, string):
        try:
            value = unquote(string)
        except TypeError:
            value = np.nan

        return value

    # find and replace identical levels represented with different spellings (att communications, at&t communications)
    def find_and_replace_exact(self, df, column_name, dictionary):
        try:
            for k, v in dictionary.items():
                df.loc[df[column_name].str.strip().isin(v), column_name] = k
        except KeyError:
            pass

    # find and replace (with regex) multiple categorical levels that are not useful
    def find_replace_regex(self, string, regex, replacement):
        try:
            replace_bool = bool(re.search(regex, string))
            if replace_bool:
                return replacement
            else:
                return string
        except TypeError:
            return np.nan
        except KeyError:
            pass

    def apply_find_replace(self, df, tuples, replace_type):
        if replace_type == 'exact':
            for element in tuples:
                column_name = element[0]
                dictionary = element[1]
                self.find_and_replace_exact(df=df, column_name=column_name, dictionary=dictionary)

        elif replace_type == 'regex':
            for element in tuples:
                column_name = element[0]
                regex = element[1]
                replacement = element[2]
                df[column_name] = df[column_name].apply(self.find_replace_regex, regex=regex, replacement=replacement)
        else:
            raise Error("failed as replace_type was not regex or exact, the only two values accepted")

    # fit does nothing
    def fit(self, X, y=None):
        return self

    # transformer method that will apply functions defined above
    def transform(self, X, y=None):

        if self.df_names:
            X = pd.DataFrame(X, columns=self.df_names)
        if self.tmx_unclean_vars:
            # parsing tmx columns
            good_tmx_keys = X.columns.intersection(self.tmx_unclean_vars)
            tmx_unclean_df = X.loc[:, good_tmx_keys]
            tmx_clean_df = tmx_unclean_df.applymap(self.parse_url_seps)
            X = X.drop(good_tmx_keys, axis=1)
            X = pd.concat([X, tmx_clean_df], axis=1)

        # good_regex_match_keys = X.columns.intersection(self.regex_match_vars)
        # good_exact_match_keys = X.columns.intersection(self.exact_match_vars)

        # collapsing dirty levels in categorical variables
        # finding and replacing using regex for fsl_source and ea_reason
        self.apply_find_replace(df=X, tuples=self.regex_match_vars, replace_type='regex')

        # finding and replacing using exact match
        self.apply_find_replace(df=X, tuples=self.exact_match_vars, replace_type='exact')

        return X


class MissingHandler(BaseEstimator, TransformerMixin):
    """
    Removes features with too many missing values (user provided thershold) and features that have just ONE value

    Keyword Args:
        missing_threshold - Columns with % of missing values greater than this threshold will be dropped

    Input:
        X - Pandas dataframe with all columns

    Returns:
        Pandas dataframe with variables above missing threshold and variables with a single constant value dropped

    """

    def __init__(self, missing_threshold=1):
        self.missing_threshold = missing_threshold

    # fit learns which variables have a single value only, and which variables have % missing values greater than threshold
    def fit(self, X, y=None):
        self.missing_cols = X.columns[X.isnull().mean() >= self.missing_threshold]
        self.one_unique_cols = [c for c
                                in list(X)
                                if len(X[c].unique()) == 1]

        self.cols_to_drop = list(set([*self.missing_cols, *self.one_unique_cols]))

        return self

    # transform drops variables learned in fit
    def transform(self, X, y=None):
        X = X.drop(self.cols_to_drop, axis=1)
        return X


class DateMissingTransformer(BaseEstimator, TransformerMixin):
    """
    For date variables where missingness has a special meaning (missing equates to value not found), imputes missing values with date application was created

    Keyword Args:
        datemissing_cols - date variables to be imputed
        reference_cols - list of reference columns (for eg., tmx vars ending in _result) that are used as references to differentiate between types of missing values
        reference_dict - mapping variables from datemissing_cols to their appropriate reference column
        imputation_col - column whose values will be used to impute missing values with (usually the date the application was created)


    Input:
        X - Dataframe with date variables to be imputed (only - if additional variables are provided, their missing values
        will be imputed with values from reference_col too)

    Returns:
        Dataframe with imputed date variables

    """

    def __init__(self, datemissing_cols, reference_cols, reference_dict, imputation_col='ca_ft_created_at_tz_adj'):
        self.datemissing_cols = datemissing_cols
        self.reference_cols = reference_cols
        self.reference_dict = reference_dict
        self.imputation_col = imputation_col

    def NullFiller(self, X, X_ref):
        names = X.columns
        X_tr = pd.DataFrame()
        for col in names:
            X_tr[col] = X[col].fillna(value=X[self.imputation_col])
            if col in self.reference_dict:
                reference_col = self.reference_dict[col]
                X_tr[col] = np.where(X_ref[reference_col].isnull(), np.nan, X_tr[col])

        return X_tr

    def fit(self, X, y=None):
        return self

    def get_feature_names(self):
        return self.col_names

    def transform(self, X, y=None):

        good_dm_keys = X.columns.intersection(self.datemissing_cols)
        datemissing_df = X.loc[:, good_dm_keys]

        good_ref_keys = X.columns.intersection(self.reference_cols)
        reference_df = X.loc[:, good_ref_keys]

        output_df = self.NullFiller(datemissing_df, reference_df)  # .reset_index(drop = True)

        cols_to_drop = list(X.columns.intersection(good_dm_keys))
        X = X.drop(cols_to_drop, axis=1)

        output_df = pd.concat([X, output_df], axis=1)
        return output_df







