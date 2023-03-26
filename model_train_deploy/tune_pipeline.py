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
from fe_vars import ida_impute_vars, date_missing_vars, fe_date_vars, fe_timestamp_vars, find_replace_exact_vars
from fe_vars import find_replace_regex_vars, date_impute_dict, date_missing_ref_vars, consolidation_dict, \
    consolidation_groups, match_groups
from numeric_features import numeric_vars
from categorical_features import categorical_vars
from bool_features import bool_vars
from tmx_toclean_features import tmx_vars
from drop_vars_list import data_error_drop_vars

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from feature_engine.encoding import RareLabelEncoder, MeanEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.model_selection import StratifiedKFold
import category_encoders as ce
import xgboost as xgb
# import featuretools as ft

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

    # pipeline hyperparameters
    parser.add_argument('--num_missing_imputer', type=str)
    parser.add_argument('--percent_missing', type=float)
    parser.add_argument('--cat_missing_imputer', type=str)
    parser.add_argument('--cat_encoder', type=str)
    parser.add_argument('--num_scaler', type=str)
    parser.add_argument('--tol_rare_label', type=float)
    parser.add_argument('--hash_components', type=int)
    parser.add_argument('--n_neighbors', type=int)

    # Data, model, and output directories
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    # parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default='X_train_20210920.csv')
    # parser.add_argument('--test-file', type=str, default='boston_test.csv')
    # parser.add_argument('--features', type=str)  # in this script we ask user to explicitly name features
    parser.add_argument('--target', type=str,
                        default='dep_var')  # in this script we ask user to explicitly name the target

    args, _ = parser.parse_known_args()

    # reading in data
    print('reading data')

    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    print('read in data successfully')

    y_train = train_df[args.target]
    X_train = train_df.drop(args.target, axis=1)
    # X_train = X_train.drop(ip_vars, axis = 1)

    # X_train = X_train.iloc[1000:7000, ]
    # y_train = y_train[1000:7000]

    dataframe_cols = list(X_train)

    print('separated full data into features and target')
    print('starting parsing and feature engineering')

    pipeline_mapping = {
        'knn': KNNImputer(n_neighbors=args.n_neighbors, weights="uniform"),
        'mean': SimpleImputer(strategy='mean'),
        'num_constant': SimpleImputer(strategy='constant', fill_value=99999999),
        'cat_constant': CategoricalImputer(imputation_method='missing', fill_value='Missing'),
        'target_enc': ce.TargetEncoder(),
        'woe_enc': ce.woe.WOEEncoder(),
        'hash_enc': ce.hashing.HashingEncoder(n_components=args.hash_components),
        'james_stein_enc': ce.james_stein.JamesSteinEncoder(),
        'catboost_enc': ce.cat_boost.CatBoostEncoder(),
        'standard': StandardScaler(),
        'clipping': MinMaxScaler(feature_range=(0.01, 0.99)),
        'robust': RobustScaler()
    }

    # assigning categorical hyperparameter values
    num_missing_imputer = pipeline_mapping[args.num_missing_imputer]
    percent_missing = args.percent_missing
    cat_missing_imputer = pipeline_mapping['cat_constant']
    cat_encoder = pipeline_mapping[args.cat_encoder]
    num_scaler = pipeline_mapping[args.num_scaler]
    tol_rare_label = args.tol_rare_label

    # Features that will be analysed and used in initial model training
    all_features = list(set([*numeric_vars, *all_date_vars, *categorical_vars]))
    # cleaning data and ensuring formats of variables
    parsing_pipeline = Pipeline(steps=[('cleaner', ParsingTransformer(df_names=dataframe_cols,
                                                                      tmx_unclean_vars=tmx_vars,
                                                                      exact_match_vars=find_replace_exact_vars,
                                                                      regex_match_vars=find_replace_regex_vars)),
                                       ('date_impute', DateMissingTransformer(datemissing_cols=date_missing_vars,
                                                                              reference_cols=date_missing_ref_vars,
                                                                              reference_dict=date_impute_dict)),
                                       ('feature_selector', FeatureSelector(all_features)),
                                       ('data_integrity', DataIntegrityTransformer(all_date_vars,
                                                                                   numeric_vars,
                                                                                   categorical_vars))
                                       ])

    # creating features
    fe_pipeline = Pipeline(steps=[("ida_fe", idaParser(ida_vars=ida_impute_vars,
                                                       consolidation_vars=consolidation_groups,
                                                       consolidation_mapping=consolidation_dict,
                                                       match_vars=match_groups)),
                                  ('dates_fe', DateTransformer(fe_date_vars,
                                                               fe_timestamp_vars,
                                                               all_date_vars)),
                                  ('missing_dealer', MissingHandler(percent_missing)),
                                  ('feature_dropper', FeatureDropper(data_error_drop_vars))
                                  ])

    # categorical and numeric transformers
    categorical_transformer = Pipeline(steps=[('cat_imputer', cat_missing_imputer),
                                              ('rare_label_encoder',
                                               RareLabelEncoder(tol=tol_rare_label, n_categories=3,
                                                                replace_with='Others')),
                                              ('cat_encoder', cat_encoder)
                                              ])

    numeric_transformer = Pipeline(steps=[('num_imputer', num_missing_imputer),
                                          ('scaler', num_scaler)
                                          ])

    # combining final cat and num transformer pipelines
    preprocessor = ColumnTransformer(transformers=[
        ("num_t", numeric_transformer, make_column_selector(dtype_exclude="object")),
        ("cat_t", categorical_transformer, make_column_selector(dtype_include="object"))])

    model = xgb.XGBClassifier(max_depth=3,
                              random_state=42
                              )

    clf = Pipeline(steps=[('parser', parsing_pipeline),
                          ('feature creation', fe_pipeline),
                          ('preprocessor', preprocessor),
                          ('classifier', model)])

    # start cross validation
    print('starting cross validation')
    start_cv_time = time.time()

    cv_obj_shuffle = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_results = cross_validate(clf, X_train, y_train, cv=cv_obj_shuffle,
                                scoring=('average_precision', 'roc_auc'), n_jobs=-1)

    for trial_run in range(len(cv_results['test_average_precision'])):
        print('Showing results for round {}'.format(trial_run))

        auc_pr = cv_results['test_average_precision'][trial_run]
        print('AUC-PR = {}'.format(auc_pr))

        auc_roc = cv_results['test_roc_auc'][trial_run]
        print('AUC-ROC = {}'.format(auc_roc))

        fit_time = cv_results['fit_time'][trial_run]
        print('Fit-Time = {}'.format(fit_time))

        score_time = cv_results['score_time'][trial_run]
        print('Score-Time = {}'.format(score_time))

    print('cross validation done')

    end_cv_time = time.time()

    print('printing final results')
    auc_pr_mean = cv_results['test_average_precision'].mean()
    print('AUC-PR = {}'.format(auc_pr_mean))

    auc_roc_mean = cv_results['test_roc_auc'].mean()
    print('AUC-ROC = {}'.format(auc_roc_mean))

    end_model_time = time.time()

    print('CV-Time = {}'.format(end_cv_time - start_cv_time))
    print('Model-Time = {}'.format(end_model_time - end_cv_time))
    print('Total-Time = {}'.format(end_model_time - start_cv_time))

    # persist model (dumping entire pipeline)
    # print('training done!')

    # print('testing model for prediction')

    # test_probs_array = clf.predict_proba(X_train_sample_1)
    # print('predict works!')

    # path = os.path.join(args.model_dir, "model.joblib")
    # joblib.dump(clf, path)

    # print('model persisted at ' + path)
    # print("saved model!")
    print("Cross Validation Results have been printed. End of Script.")

