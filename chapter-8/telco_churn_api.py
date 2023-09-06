import os
import secrets
from typing import Annotated
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

app = FastAPI()

model = joblib.load("churn_pipeline.pkl")

security = HTTPBasic()
USER = bytes(os.getenv("CHURN_USER"), "utf-8")
PASSWORD = bytes(os.getenv("CHURN_PASSWORD"), "utf-8")


def authenticate(username: bytes, password: bytes):
    valid_user = secrets.compare_digest(
        username, USER
    )
    valid_password = secrets.compare_digest(
        password, PASSWORD
    )
    if not (valid_user and valid_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return username


@app.post('/predict')
def predict_instances(
        credentials: Annotated[HTTPBasicCredentials, Depends(security)],
        instances: list[dict[str, str]]
):
    authenticate(credentials.username.encode("utf-8"), credentials.password.encode("utf-8"))

    instance_frame = pd.DataFrame(instances)
    predictions = model.predict_proba(instance_frame)

    results = {}
    for i, row in enumerate(predictions):
        prediction = model.classes_[np.argmax(row)]
        probability = np.amax(row)
        results[i] = {"prediction": prediction, "probability": probability}
    return results
