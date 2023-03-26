# parsing modules
import json
import re
from urllib.parse import unquote
from datetime import datetime, timezone

# Pipeline imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer, MissingIndicator

from sklearn.pipeline import FeatureUnion, Pipeline

import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar


class DateTransformer(BaseEstimator, TransformerMixin):
    """
    Engineers date based features from provided list of date and timestamp variables.
    Transformer requires variables to be in datetime format.
    Keyword Args:
        date_cols - List of variables to engineer features from, that are in date format
        timestamp_cols - List of variables to engineer features from, that include timestamp
        creation_col - Column based on which to calculate 'days since'

    Input:
        X - Dataframe with date variables to be transformed (only - any other variables will be dropped)

    Returns:
        Dataframe with engineered features only (original date columns are removed)

    """

    # constructor uses USFederalHolidayCalendar() to check for holidays
    def __init__(self, date_cols, timestamp_cols, todrop_cols, creation_col='ca_ft_created_at_tz_adj'):
        self.date_cols = date_cols
        self.timestamp_cols = timestamp_cols
        self.todrop_cols = todrop_cols
        self.hols_list = list(USFederalHolidayCalendar().holidays())
        self.creation_col = creation_col

    # fit does nothing
    def fit(self, X, y=None):
        return self

    def days_since(self, date, reference_value):
        reference = reference_value.replace(tzinfo=None)
        date = date.replace(tzinfo=None)
        return (reference - date).days

    def cyclic_seconds_sin(self, value):
        seconds_in_day = 24 * 60 * 60
        seconds_past_midnight = (value.hour * 3600) + (value.minute * 60) + value.second + (
                    value.microsecond / 1000000.0)
        sin_time = np.sin(2 * np.pi * seconds_past_midnight / seconds_in_day)

        return sin_time

    def cyclic_seconds_cos(self, value):
        seconds_in_day = 24 * 60 * 60
        seconds_past_midnight = (value.hour * 3600) + (value.minute * 60) + value.second + (
                    value.microsecond / 1000000.0)
        cos_time = np.cos(2 * np.pi * seconds_past_midnight / seconds_in_day)

        return cos_time

    def cyclic_dow_sin(self, values):
        days_in_week = 7
        days_of_week = (values.astype('datetime64[D]').view('int64') - 4) % 7
        sin_days = np.sin(2 * np.pi * days_of_week / days_in_week)

        return sin_days

    def cyclic_dow_cos(self, values):
        days_in_week = 7
        days_of_week = (values.astype('datetime64[D]').view('int64') - 4) % 7
        cos_days = np.cos(2 * np.pi * days_of_week / days_in_week)

        return cos_days

    def day_of_week(self, value):
        day_num = value.weekday()
        days = {0: 'Mon', 1: 'Tues', 2: 'Weds', 3: 'Thurs', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
        try:
            dow = days[day_num]
        except KeyError:
            dow = np.nan

        return dow

    def is_weekend(self, values):
        day_nums = (values.astype('datetime64[D]').view('int64') - 4) % 7
        weekend_flags = np.where(day_nums < 5, 0, 1)
        return weekend_flags

    def is_holiday(self, values):
        dates = values.astype('datetime64[D]')
        hol_flag = np.where(np.isin(dates, self.hols_list), 1, 0)
        return hol_flag

    # cycle through date variables provided, and engineer multiple features
    def create_date_variables(self, X):
        names = X.columns

        for col in names:
            X['{}_ft_days_since'.format(col)] = list(map(self.days_since, X[col], X[self.creation_col]))
            X['{}_ft_dayofweek_sin'.format(col)] = self.cyclic_dow_sin(X[col].values)
            X['{}_ft_dayofweek_cos'.format(col)] = self.cyclic_dow_cos(X[col].values)
            X['{}_ft_dayofweek'.format(col)] = list(map(self.day_of_week, X[col]))
            X['{}_ft_is_weekend'.format(col)] = self.is_weekend(X[col].values)
            X['{}_ft_is_holiday'.format(col)] = self.is_holiday(X[col].values)

        X = X.drop(names, axis=1)
        return X

    # cycle through timestamp columns, and engineer multiple features
    def create_timestamp_variables(self, X):
        names = X.columns

        for col in names:
            X['{}_ft_timeofday_sin'.format(col)] = list(map(self.cyclic_seconds_sin, X[col]))
            X['{}_ft_timeofday_cos'.format(col)] = list(map(self.cyclic_seconds_cos, X[col]))

        X = X.drop(names, axis=1)
        return X

    def transform(self, X, y=None):
        good_date_keys = X.columns.intersection(self.date_cols)
        good_timestamp_keys = X.columns.intersection(self.timestamp_cols)
        date_df = X.loc[:, good_date_keys]
        date_df = self.create_date_variables(date_df)

        timestamp_df = X.loc[:, good_timestamp_keys]
        timestamp_df = self.create_timestamp_variables(timestamp_df)

        cols_to_drop = list(X.columns.intersection(self.todrop_cols))

        X = X.drop(cols_to_drop, axis=1)
        output_df = pd.concat([X, date_df, timestamp_df], axis=1)

        return output_df


class FeatureConsolidator(BaseEstimator, TransformerMixin):
    """
    Create flag variables for IDA columns with -1, -2, and convert these values to np.nan. Creates match variables based on input tuples provided. Consolidates variables based on consolidation mapping provided.
    Keyword Args:
        ida_vars - Variables that require special treatment for IDA missing values (-1 and -2)
        consolidation_vars - tuples that will be provided as input to consolidate_variables function
        consolidation_mapping - mapping dictionary that will be provided as input to consolidate_variables function
        match_vars - tuples that are provided as input to create_match_variables function

    Input:
        X - Dataframe with variables variables to be transformed (other variables can also be included)

    Returns:
        Dataframe with IDA variable values changed (-1, -2 replaced), and flag variables (ending with _matchstatus) for each IDA variable provided, features engineered and a few dropped as a part of consolidation process, and match variables added for features provided as a part of match_vars

    """

    def __init__(self, ida_vars, consolidation_vars, consolidation_mapping, match_vars):
        self.ida_vars = ida_vars
        self.consolidation_vars = consolidation_vars
        self.consolidation_mapping = consolidation_mapping
        self.match_vars = match_vars

    def return_ida_matchstatus(self, X, col_name):
        conditions = [X[col_name] == -1, X[col_name] == -2, np.isnan(X[col_name])]
        choices = ['Partial Match', 'No Match', 'Missing']
        matchstatus = np.select(conditions, choices, default='Other')

        return matchstatus

    # cycle through each IDA variable and create match status column
    def adjust_ida_variables(self, X):
        names = X.columns
        for col in names:
            X['{}_ft_matchstatus'.format(col)] = self.return_ida_matchstatus(X, col)

        X = X.replace(dict.fromkeys([-1, -2], np.nan))

        return X

    def consolidate_variables(self, X, var_groups, mapping_dict):
        inv_mapping = {v: k for k, v in mapping_dict.items()}

        for col_group in var_groups:
            new_col = col_group[0]
            ref_col_1 = col_group[1]
            ref_col_2 = col_group[2]

            try:
                X[[ref_col_1, ref_col_2]] = X[[ref_col_1, ref_col_2]].applymap(mapping_dict.get)
                X[new_col] = np.nanmax(X[[ref_col_1, ref_col_2]].values, axis=1)

                X.drop([ref_col_1, ref_col_2], axis=1, inplace=True)
                X[new_col] = X[new_col].map(inv_mapping)
            except KeyError:
                pass

    def create_match_variables(self, X, var_groups):
        for col_group in var_groups:
            new_col = col_group[0]
            ref_col_1 = col_group[1]
            ref_col_2 = col_group[2]
            try:
                X[new_col] = np.where(X[ref_col_1] == X[ref_col_2], 1, 0)
            except KeyError:
                pass

    # fit does nothing
    def fit(self, X, y=None):
        return self

    # return features names to add to numpy array
    def get_feature_names(self):
        return self.col_names

    # transform that describes what FeatureSelector will do (just return columns specified)
    def transform(self, X, y=None):
        good_ida_keys = X.columns.intersection(self.ida_vars)
        ida_df = X.loc[:, good_ida_keys]

        output_df = self.adjust_ida_variables(ida_df)
        self.col_names = output_df.columns.tolist()

        cols_to_drop = list(X.columns.intersection(good_ida_keys))
        X = X.drop(cols_to_drop, axis=1)

        output_df = pd.concat([X, output_df], axis=1)

        # consolidating columns
        self.consolidate_variables(X=output_df, var_groups=self.consolidation_vars,
                                   mapping_dict=self.consolidation_mapping)

        # creating match variables (starting with editing existing variables)
        output_df['tmx_raw_time_zone'] = output_df['tmx_raw_time_zone'].div(60).round(4)
        self.create_match_variables(X=output_df, var_groups=self.match_vars)

        return output_df


class MissingIndicatorTransformer(BaseEstimator, TransformerMixin):
    """
    For variables where missingness has a special meaning (missing equates to value not found), create flag variables
    to indicate missingness

    Keyword Args:
        None

    Input:
        missingind_vars - List of variables to create flags for (only - if additional variables are provided,
        flags will be created for them too)

    Returns:
        Dataframe with flag (boolean) variables for each column in input dataframe

    """

    # constructor defines function from sklearn which creates flag variables based on missing data
    def __init__(self, missingind_vars):
        self.missingind_vars = missingind_vars
        self.indicator = MissingIndicator(features='all')

    def fit(self, X, y=None):
        good_mi_keys = X.columns.intersection(self.missingind_vars)
        missingind_df = X.loc[:, good_mi_keys]
        self.indicator.fit(missingind_df)
        return self

    # return features names to add to numpy array
    def get_feature_names(self):
        return self.col_names

    # transform creates indicator variables with suffix '_missing_ind'
    def transform(self, X, y=None):
        good_mi_keys = X.columns.intersection(self.missingind_vars)
        missingind_df = X.loc[:, good_mi_keys]

        X_tr = self.indicator.transform(missingind_df) * 1.0
        output_df = pd.DataFrame(
            X_tr, columns=missingind_df.columns + '_ft_missing_ind'
        ).reset_index(drop=True)

        self.col_names = output_df.columns.tolist()

        output_df = pd.concat([X, output_df], axis=1)
        return output_df




