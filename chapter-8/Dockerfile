    FROM python:3.10-slim

    RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
    RUN apt-get -y install curl
    RUN apt-get install libgomp1

    WORKDIR /usr/src/app

    COPY requirements.txt ./
    RUN pip install --no-cache-dir -r requirements.txt
    COPY . .

    CMD [ "uvicorn", "telco_churn_api:app", "--host", "0.0.0.0", "--port", "8080" ]