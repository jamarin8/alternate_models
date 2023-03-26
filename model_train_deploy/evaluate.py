"""Evaluation script for measuring AUC PR for fraud model"""

import logging
import pickle

import os
import json
import subprocess
import sys
import numpy as np
import pathlib
import tarfile
import joblib

# adding base directory to root to import custom scripts
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(BASE_DIR)


# module to install libraries in main
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install("feature_engine==1.0.2")
install("xgboost==1.4.2")
install("category_encoders==2.2.2")

import numpy as np
import pandas as pd

# ML Imports
from sklearn.metrics import accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, \
    precision_recall_curve
import category_encoders as ce
import xgboost as xgb
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from feature_engine.encoding import RareLabelEncoder, MeanEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.model_selection import StratifiedKFold

# custom parsing transfomers
from parsing_pipeline import FeatureSelector, ParsingTransformer, MissingHandler, FeatureDropper, DateMissingTransformer
from data_integrity_pipeline import DataIntegrityTransformer
from feature_engineering import DateTransformer, FeatureConsolidator, MissingIndicatorTransformer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path, "r:gz") as tar:
        tar.extractall("/opt/ml/processing/model")

    model_path = "/opt/ml/processing/model"
    print(os.listdir(model_path))

    logger.debug("Loading xgboost model.")
    clf = joblib.load(os.path.join(model_path, "model.joblib"))

    logger.debug("Reading test data.")
    # test_path = "/opt/ml/processing/test/X_test_payload_features_20211005.csv"
    test_path = "/opt/ml/processing/test/X_test_transformed_payload_features_20211028.csv"

    X_test = pd.read_csv(test_path)

    logger.debug("Reading test data.")
    y_test = X_test['dep_var']
    X_test.drop('dep_var', axis=1, inplace=True)

    logger.info("Performing predictions against test data.")
    prediction_scores = clf.predict_proba(X_test)
    y_scores = prediction_scores[:, 1]

    logger.debug("Calculating AUC PR")
    auc_pr = average_precision_score(y_true=y_test, y_score=y_scores)
    xg_precision, xg_recall, _ = precision_recall_curve(y_test, y_scores)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    report_dict = {
        "binary_classification_metrics": {
            "auc": {
                "value": auc_pr,
                "standard_deviation": None
            },
            "receiver_operating_characteristic_curve": {
                "false_positive_rates": list(fpr),
                "true_positive_rates": list(tpr),
            },
            "precision_recall_curve": {
                "precisions": list(xg_precision),
                "recalls": list(xg_recall)
            },

        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with aucpr: %f", auc_pr)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))


