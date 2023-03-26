# This is the training script
from __future__ import print_function
import os
import sys
import subprocess

# #adding base directory to root to import custom scripts
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(BASE_DIR)


# module to install libraries in main
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install("feature_engine==1.0.2")
install("xgboost==1.4.2")
install("category_encoders==2.2.2")

import time
import logging
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
from sklearn.model_selection import StratifiedKFold

# evaluation imports
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, f1_score, recall_score, precision_score, average_precision_score
from sklearn.model_selection import cross_validate

# custom parsing transfomers
from parsing_pipeline import FeatureSelector, ParsingTransformer, MissingHandler, FeatureDropper, DateMissingTransformer
from data_integrity_pipeline import DataIntegrityTransformer
from feature_engineering import DateTransformer, FeatureConsolidator, MissingIndicatorTransformer

# custom variable lists
from date_features import all_date_vars
from fe_vars import ida_impute_vars, date_missing_vars, fe_date_vars, fe_timestamp_vars, find_replace_exact_vars
from fe_vars import final_raw_features, final_transformed_features, final_transformed_noida_features, \
    final_raw_noida_features
from fe_vars import find_replace_regex_vars, date_impute_dict, date_missing_ref_vars, consolidation_dict, \
    consolidation_groups, match_groups, missingind_vars
from numeric_features import numeric_vars
from categorical_features import categorical_vars
from tmx_toclean_features import tmx_vars

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

# inference functions

"""
The functions defined below (input_fn, model_fn, and predict_fn are used by the container during serving to shape the input payload, load the saved model and return a prediction respectively)
"""


def input_fn(input_data, content_type):
    """An input function that passes along values with string content"""

    #     print("Checking input_fn. CAN ANYBODY SEE THIS?!")
    #     print("The content type passed is {}".format(content_type))
    #     print("The payload looks like this {}".format(input_data))
    np_array = encoders.decode(input_data, content_type)
    return np_array


def model_fn(model_dir):
    """
    model_fn re-defined to accomodate Sagemaker Python SDK.
    Args:
        model_dir: Default location where model is stored on the instance

    Returns: a pipeline object which can be used to predict with
    """
    model = joblib.load(os.path.join(model_dir, "model_pipeline.joblib"))
    return model


def predict_fn(input_data, model):
    """
    A modified predict_fn for Scikit-learn. Calls a model on data deserialized in input_fn. Returns probabilities instead of predictions.

    Args:
        input_data: input data (Numpy array) for prediction deserialized by input_fn
        model: Scikit-learn model loaded in memory by model_fn

    Returns: probability of fraud
    """
    #     print("testing out that model control is now inside the predict_fn. CAN ANYBODY SEE THIS?!. Next print statement is how the input data looks")
    #     print("But before that, the type of the data sent is {}".format(type(input_data)))
    #     print("it's shape is {}".format(input_data.shape))
    #     print("it's shape is {} after reshaping".format(input_data.reshape(1, -1).shape))
    #     print("Lets look at model object")
    #     print(model)

    #     print(input_data)

    # reshaping during prediction to convert a 1D array to 2D array (xgb can't predict on 1d array, needs additional column axis)
    if len(input_data.shape) == 1:
        pred_prob = model.predict_proba(input_data.reshape(1, -1))
    else:
        pred_prob = model.predict_proba(input_data)
    return pred_prob


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()

    # pipeline hyperparameters
    parser.add_argument('--num_missing_imputer', type=str)
    parser.add_argument('--percent_missing', type=float)
    parser.add_argument('--cat_missing_imputer', type=str)
    parser.add_argument('--cat_encoder', type=str)
    parser.add_argument('--num_scaler', type=str)
    parser.add_argument('--tol_rare_label', type=float)

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

    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--train-file', type=str, default='X_train_payload_features_20211122.csv')

    parser.add_argument('--target', type=str, default='dep_var')

    args, _ = parser.parse_known_args()

    # reading in data
    logger.debug('reading data')

    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    logger.debug('read in data successfully')

    y_train = train_df[args.target]
    X_train = train_df.drop(args.target, axis=1)

    dataframe_cols = list(X_train)

    print('separated full data into features and target')
    print('starting parsing and feature engineering')

    pipeline_mapping = {
        'mean': SimpleImputer(strategy='mean'),
        'cat_constant': CategoricalImputer(imputation_method='missing', fill_value='Missing'),
        'woe_enc': ce.woe.WOEEncoder(),
        'standard': StandardScaler()
    }

    # assigning categorical hyperparameter values
    num_missing_imputer = pipeline_mapping[args.num_missing_imputer]
    percent_missing = args.percent_missing
    cat_missing_imputer = pipeline_mapping[args.cat_missing_imputer]
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
                                       ('data_integrity', DataIntegrityTransformer(all_date_vars,
                                                                                   numeric_vars,
                                                                                   categorical_vars))
                                       ])

    # creating features
    fe_pipeline = Pipeline(steps=[("ida_fe", FeatureConsolidator(ida_vars=ida_impute_vars,
                                                                 consolidation_vars=consolidation_groups,
                                                                 consolidation_mapping=consolidation_dict,
                                                                 match_vars=match_groups)),
                                  ('dates_fe', DateTransformer(fe_date_vars,
                                                               fe_timestamp_vars,
                                                               all_date_vars)),
                                  ('feature_selector', FeatureSelector(final_transformed_features)),
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

    #     #start cross validation
    #     print('starting cross validation')
    #     start_cv_time = time.time()

    #     cv_obj_shuffle = StratifiedKFold(n_splits=10, shuffle = True, random_state = 42)

    #     cv_results = cross_validate(model_pipeline, X_train, y_train, cv=cv_obj_shuffle,
    #                                scoring=('average_precision', 'roc_auc'),
    #                                n_jobs = -1)

    #     print('cross validation done')

    #     end_cv_time = time.time()

    #     print('printing final results')
    #     auc_pr_mean = cv_results['test_average_precision'].mean()
    #     print('AUC-PR = {}'.format(auc_pr_mean))

    #     auc_roc_mean = cv_results['test_roc_auc'].mean()
    #     print('AUC-ROC = {}'.format(auc_roc_mean))

    model = xgb.XGBClassifier(max_depth=args.max_depth + 4,
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

    model_pipeline = Pipeline(steps=[('parser', parsing_pipeline),
                                     ('feature creation', fe_pipeline),
                                     ('preprocessor', preprocessor),
                                     ('classifier', model)])

    # train model
    print('starting model training')
    model_pipeline.fit(X_train, y_train)
    # model.fit(X_train, y_train)

    # persist model (dumping entire pipeline)
    print('training done! Copying over training files')

    import shutil
    from distutils.dir_util import copy_tree

    # path to source directory
    src_dir = '/opt/ml/code'
    # path to destination directory
    # dest_dir_1 = os.path.join(args.model_dir, "code/")
    dest_dir_1 = args.model_dir

    # getting all the files in the source directory
    files = os.listdir(src_dir)

    # copying files over
    copy_tree(src_dir, dest_dir_1)

    path = os.path.join(args.model_dir, "model_pipeline.joblib")
    joblib.dump(model_pipeline, path)

    print('model persisted at ' + path)

    """ DEBUG ZONE. These code snippets are used to debug location and storage of inference script during deployment"""
    print("THIS IS THE FOLDER WHERE THE SCRIPT IS STORED: {}".format(dest_dir_1))
    print("THE OTHER FILES IN SCRIPT DIR ARE: {}".format(os.listdir(dest_dir_1)))

    print("saved model!")



