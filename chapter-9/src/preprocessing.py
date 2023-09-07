
import argparse
import pathlib
import boto3
import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args, _ = parser.parse_known_args()
    logger.info("Received arguments {}".format(args))
    
    # Set local path prefix in the processing container
    local_dir = "/opt/ml/processing"    
    
    input_data_path = os.path.join("/opt/ml/processing/census-income", "census-income.csv")
    
    logger.info("Reading claims data from {}".format(input_data_path))
    df = pd.read_csv(input_data_path)
    
    df = df.replace("unknown", np.nan)

    categorical_features = list(
        df.loc[:, df.dtypes == "object"].columns.values
    )
    for f in categorical_features:
        df[f] = df[f].astype("category")
    categorical_features = [c for c in categorical_features if not c == "Class"]

    for f in df.columns:
        if f in categorical_features:
            df[f].fillna(df[f].mode()[0], inplace=True)
        else:
            df[f].fillna(df[f].median(), inplace=True)

    df = pd.get_dummies(df, columns=categorical_features)

    X = df.drop(columns=["Class"], axis=1)
    y = df["Class"]
    
    train_ratio = args.train_ratio
    val_ratio = args.validation_ratio
    test_ratio = args.test_ratio
    
    logger.debug("Splitting data into train, validation, and test sets")

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_ratio)
    
    X_train["Class"] = y_train
    X_val["Class"] = y_val
    X_test["Class"] = y_test
    X["Class"] = y
    
    logger.info("Train data shape after preprocessing: {}".format(X_train.shape))
    logger.info("Validation data shape after preprocessing: {}".format(X_val.shape))
    logger.info("Test data shape after preprocessing: {}".format(X_test.shape))
    
    # Save processed datasets to the local paths in the processing container.
    # SageMaker will upload the contents of these paths to S3 bucket
    logger.debug("Writing processed datasets to container local path.")
    train_output_path = os.path.join(f"{local_dir}/train", "train.csv")
    validation_output_path = os.path.join(f"{local_dir}/val", "validation.csv")
    test_output_path = os.path.join(f"{local_dir}/test", "test.csv")
    full_processed_output_path = os.path.join(f"{local_dir}/full", "dataset.csv")

    logger.info("Saving train data to {}".format(train_output_path))
    X_train.to_csv(train_output_path, index=False)
    
    logger.info("Saving validation data to {}".format(validation_output_path))
    X_val.to_csv(validation_output_path, index=False)

    logger.info("Saving test data to {}".format(test_output_path))
    X_test.to_csv(test_output_path, index=False)
    
    logger.info("Saving full processed data to {}".format(full_processed_output_path))
    X.to_csv(full_processed_output_path, index=False)
