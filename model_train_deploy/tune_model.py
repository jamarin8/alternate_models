# V3 - A straightforward tuning script that optimizes for AUC-PR in HSBC and TD only

from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil

from platform import python_version

import argparse
import csv
import json
import joblib
import numpy as np
import pandas as pd

# basic modules
import sys
import importlib

# parsing modules
import json
import re
from urllib.parse import unquote
from datetime import datetime, timezone

# custom parsing transfomers
from parsing_pipeline import FeatureSelector, ParsingTransformer, MissingHandler, FeatureDropper, DateMissingTransformer
from data_integrity_pipeline import DataIntegrityTransformer
from feature_engineering import DateTransformer, idaParser, MissingIndicatorTransformer

# custom variable lists
from date_features import all_date_vars
from fe_vars import ida_impute_vars, date_missing_vars, fe_date_vars, fe_timestamp_vars, find_replace_exact_vars, \
    find_replace_regex_vars, date_impute_dict, date_missing_ref_vars
from numeric_features import numeric_vars
from categorical_features import categorical_vars
from bool_features import bool_vars
from tmx_toclean_features import tmx_vars
from drop_vars_list import vars_to_drop, low_fi_vars, ip_vars

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from feature_engine.encoding import RareLabelEncoder, MeanEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
import category_encoders as ce
import xgboost as xgb
# import featuretools as ft

# Modeling Imports
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate

# Pipeline imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

# evaluation imports
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, f1_score, recall_score, precision_score, average_precision_score
from sklearn.model_selection import cross_validate
from model_evaluation_class import ModelEvaluation

# sagemaker imports
# import sagemaker
# import boto3
# from sagemaker import get_execution_role

import re

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)


# inference functions ---------------

def model_fn(model_dir):
    """
    model_fn re-defined to accomodate Sagemaker Python SDK.

    Args:
        model_dir: Default location where model is stored on the instance

    Returns: a pipeline object which can be used to predict with

    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


def predict_fn(input_data, model):
    """
    A modified predict_fn for Scikit-learn. Calls a model on data deserialized in input_fn. Returns probabilities instead of predictions.

    Args:
        input_data: input data (Numpy array) for prediction deserialized by input_fn
        model: Scikit-learn model loaded in memory by model_fn

    Returns: probability of fraud
    """
    pred_prob = model.predict_proba(input_data)
    return pred_prob


if __name__ == '__main__':
    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    # Hyperparameters are described here.

    # model hyperparameters
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--n_estimators', type=int)
    parser.add_argument('--reg_alpha', type=int)
    parser.add_argument('--reg_lambda', type=int)
    parser.add_argument('--gamma', type=int)
    parser.add_argument('--min_child_weight', type=float)
    parser.add_argument('--subsample', type=float)
    parser.add_argument('--colsample_bytree', type=float)
    parser.add_argument('--max_delta_step', type=int)
    parser.add_argument('--colsample_bylevel', type=float)
    parser.add_argument('--colsample_bynode', type=float)
    parser.add_argument('--num_parallel_tree', type=int)

    # pipeline hyperparameters
    parser.add_argument('--num_missing_imputer', type=str)
    parser.add_argument('--percent_missing', type=float)
    parser.add_argument('--cat_missing_imputer', type=str)
    parser.add_argument('--cat_encoder', type=str)
    parser.add_argument('--num_scaler', type=str)
    parser.add_argument('--tol_rare_label', type=float)

    # Data, model, and output directories
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    # parser.add_argument('--raw', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default='complete_ts_transformed_20210924.csv')
    parser.add_argument('--raw-file', type=str, default='X_train_20210908.csv')
    parser.add_argument('--target', type=str,
                        default='dep_var')  # in this script we ask user to explicitly name the target

    args, _ = parser.parse_known_args()

    # reading in data
    print('reading data')

    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    train_df_raw = pd.read_csv(os.path.join(args.test, args.raw_file))
    print('read in data successfully')

    y_train = train_df[args.target]
    X_train = train_df.drop(args.target, axis=1)
    X_train.drop(ip_vars, axis=1, inplace=True)

    # X_train = X_train.drop(low_fi_vars, axis = 1)

    # X_train_sample_1 = X_train.iloc[7900:8000, ]
    # y_train_sample_1 = y_train[7900:8000, ]

    # X_train = X_train.iloc[1000:7000, ]
    # y_train = y_train[1000:7000]

    # dataframe_cols = list(X_train)

    print('separated full data into features and target')

    clf = xgb.XGBClassifier(max_depth=args.max_depth,
                            learning_rate=args.learning_rate,
                            n_estimators=args.n_estimators,
                            reg_alpha=args.reg_alpha,
                            reg_lambda=args.reg_lambda,
                            gamma=args.gamma,
                            min_child_weight=args.min_child_weight,
                            subsample=args.subsample,
                            colsample_bytree=args.colsample_bytree,
                            colsample_bylevel=args.colsample_bylevel,
                            colsample_bynode=args.colsample_bynode,
                            max_delta_step=args.max_delta_step,
                            random_state=42,
                            n_jobs=-1
                            )

    cv_obj_shuffle = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # start cross validation
    print('starting cross validation')
    start_cv_time = time.time()
    me_obj = ModelEvaluation()
    partner_perf_df = me_obj.cv_partner_performance(classifier=clf, cv=cv_obj_shuffle, X=X_train, y=y_train,
                                                    X_raw=train_df_raw)
    print('cross validation done')

    end_cv_time = time.time()

    hsbc_td_aucpr = partner_perf_df.groupby('partner').mean().loc[['hsbc', 'td']]['auc_pr'].mean()

    print('printing final results')
    print('AUC-PR = {}'.format(hsbc_td_aucpr))

    print("Cross Validation Results have been printed. End of Script.")

