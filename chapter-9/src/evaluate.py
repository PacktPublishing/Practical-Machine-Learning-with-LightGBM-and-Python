
import logging
import pathlib
import pickle
import tarfile
import argparse
import os
import subprocess
import sys
import joblib
import json

def install(package):
    subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", package])
    
install("lightgbm")

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


def prepare_data(df):
    categorical_features = list(
        df.loc[:, df.dtypes == "object"].columns.values
    )
    for f in categorical_features:
        df[f] = df[f].astype("category")
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    logger.debug("Loading LightGBM model.")
    model = joblib.load(open("lightgbm-model", "rb"))

    logger.debug("Reading test data.")
    test_local_path = "/opt/ml/processing/test/test.csv"
    
    test_df = pd.read_csv(test_local_path)
    
    X_test, y_test = prepare_data(test_df)
    test_f1 = f1_score(y_test, model.predict(X_test))
    
    logger.debug("Calculating F1 score.")
    metric_dict = {
        "classification_metrics": {"f1": {"value": test_f1}}
    }
    
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing evaluation report with F1: %f", test_f1)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(metric_dict))
