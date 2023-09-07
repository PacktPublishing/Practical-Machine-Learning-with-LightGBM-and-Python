
import argparse
import os
import subprocess
import sys
import joblib
import json
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

def install(package):
    subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", package])
    
install("lightgbm")

import lightgbm as lgb


def prepare_data(df):
    categorical_features = list(
        df.loc[:, df.dtypes == "object"].columns.values
    )
    for f in categorical_features:
        df[f] = df[f].astype("category")
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--boosting_type", type=str, default="gbdt")
    parser.add_argument("--objective", type=str, default="binary")

    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_leaves", type=int, default=30)
    parser.add_argument("--max_bin", type=int, default=300)

    # SageMaker
    parser.add_argument("--train_data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation_data_dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))

    args = parser.parse_args()

    train_df = pd.read_csv(f"{args.train_data_dir}/train.csv")
    val_df = pd.read_csv(f"{args.validation_data_dir}/validation.csv")

    params = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_bin": args.max_bin,
    }

    X, y = prepare_data(train_df)
    model = lgb.LGBMClassifier(**params)

    scores = cross_val_score(model, X, y, scoring="f1_macro")
    train_f1 = scores.mean()
    model = model.fit(X, y)

    X_test, y_test = prepare_data(val_df)
    test_f1 = f1_score(y_test, model.predict(X_test))

    print(f"[0]#011train-f1:{train_f1:.2f}")
    print(f"[0]#011validation-f1:{test_f1:.2f}")

    metrics_data = {"hyperparameters": params,
                    "binary_classification_metrics": {"validation:f1": {"value": test_f1},
                                                      "train:f1": {"value": train_f1}
                                                      }
                    }

    # Save the evaluation metrics to the location specified by output_data_dir
    metrics_location = args.output_data_dir + "/metrics.json"

    # Save the model to the location specified by model_dir
    model_location = args.model_dir + "/lightgbm-model"

    with open(metrics_location, "w") as f:
        json.dump(metrics_data, f)

    with open(model_location, "wb") as f:
        joblib.dump(model, f)
