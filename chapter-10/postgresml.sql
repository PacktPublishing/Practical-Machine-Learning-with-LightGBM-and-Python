DROP TABLE pgml.telco_churn CASCADE;
CREATE TABLE pgml.telco_churn
(
    customerid       VARCHAR(100),
    gender           VARCHAR(100),
    seniorcitizen    BOOLEAN,
    partner          VARCHAR(10),
    dependents       VARCHAR(10),
    tenure           REAL,
    phoneservice     VARCHAR(10),
    multiplelines    VARCHAR(30),
    internetservice  VARCHAR(30),
    onlinesecurity   VARCHAR(30),
    onlinebackup     VARCHAR(30),
    deviceprotection VARCHAR(30),
    techsupport      VARCHAR(30),
    streamingtv      VARCHAR(30),
    streamingmovies  VARCHAR(30),
    contract         VARCHAR(30),
    paperlessbilling VARCHAR(30),
    paymentmethod    VARCHAR(30),
    monthlycharges   VARCHAR(50),
    totalcharges     VARCHAR(50),
    churn            VARCHAR(10)
);

COPY pgml.telco_churn (customerid,
                       gender,
                       seniorcitizen,
                       partner,
                       dependents,
                       tenure,
                       phoneservice,
                       multiplelines,
                       internetservice,
                       onlinesecurity,
                       onlinebackup,
                       deviceprotection,
                       techsupport,
                       streamingtv,
                       streamingmovies,
                       contract,
                       paperlessbilling,
                       paymentmethod,
                       monthlycharges,
                       totalcharges,
                       churn
    ) FROM '/tmp/telco-churn.csv'
    DELIMITER ','
    CSV HEADER;

SELECT *
FROM pgml.telco_churn;

UPDATE pgml.telco_churn
SET totalcharges = NULL
WHERE totalcharges = ' ';

DROP VIEW pgml.telco_churn_data;
CREATE VIEW pgml.telco_churn_data AS
SELECT gender,
       seniorcitizen,
       CAST(CASE partner
                WHEN 'Yes' THEN true
                WHEN 'No' THEN false
           END AS BOOLEAN) AS partner,
       CAST(CASE dependents
                WHEN 'Yes' THEN true
                WHEN 'No' THEN false
           END AS BOOLEAN) AS dependents,
       tenure,
       CAST(CASE phoneservice
                WHEN 'Yes' THEN true
                WHEN 'No' THEN false
           END AS BOOLEAN) AS phoneservice,
       multiplelines,
       internetservice,
       onlinesecurity,
       onlinebackup,
       deviceprotection,
       techsupport,
       streamingtv,
       streamingmovies,
       contract,
       paperlessbilling,
       paymentmethod,
       CAST(monthlycharges AS REAL),
       CAST(totalcharges AS REAL),
       CAST(CASE churn
                WHEN 'Yes' THEN true
                WHEN 'No' THEN false
           END AS BOOLEAN) AS churn
FROM pgml.telco_churn;


SELECT *
FROM pgml.train('Telco Churn',
                task => 'classification',
                relation_name => 'pgml.telco_churn_data',
                y_column_name => 'churn',
                algorithm => 'lightgbm',
                preprocess => '{"totalcharges": {"impute": "mean"} }',
                search => 'random',
                search_args => '{"n_iter": 500 }',
                search_params => '{
                        "num_leaves": [2, 4, 8, 16, 32, 64, 128],
                        "max_bin": [32, 64, 128, 256, 512],
                        "learning_rate": [0.0001, 0.001, 0.1, 0.5],
                        "n_estimators": [20, 40, 80, 100, 200, 400]
                }'
    );

SELECT metrics, hyperparams
FROM pgml.models m
LEFT OUTER JOIN pgml.projects p on p.id = m.project_id
WHERE p.name = 'Telco Churn';

SELECT pgml.predict(
               'Telco Churn',
               ROW (
                   CAST('Male' AS VARCHAR(30)),
                   1,
                   1,
                   0,
                   0,
                   1,
                   CAST('No phone service' AS VARCHAR(30)),
                   CAST('Fiber optic' AS VARCHAR(30)),
                   CAST('No' AS VARCHAR(30)),
                   CAST('Yes' AS VARCHAR(30)),
                   CAST('No' AS VARCHAR(30)),
                   CAST('No' AS VARCHAR(30)),
                   CAST('Yes' AS VARCHAR(30)),
                   CAST('No' AS VARCHAR(30)),
                   CAST('Month-to-month' AS VARCHAR(30)),
                   CAST('Yes' AS VARCHAR(30)),
                   CAST('Electronic check' AS VARCHAR(30)),
                   CAST(20.25 AS REAL),
                   CAST(4107.25 AS REAL)
                   )
           ) AS prediction;

SELECT *,
       pgml.predict(
               'Telco Churn',
               ROW (
                   gender,
                   seniorcitizen,
                   CAST(CASE partner
                            WHEN 'Yes' THEN true
                            WHEN 'No' THEN false
                       END AS BOOLEAN),
                   CAST(CASE dependents
                            WHEN 'Yes' THEN true
                            WHEN 'No' THEN false
                       END AS BOOLEAN),
                   tenure,
                   CAST(CASE phoneservice
                            WHEN 'Yes' THEN true
                            WHEN 'No' THEN false
                       END AS BOOLEAN),
                   multiplelines,
                   internetservice,
                   onlinesecurity,
                   onlinebackup,
                   deviceprotection,
                   techsupport,
                   streamingtv,
                   streamingmovies,
                   contract,
                   paperlessbilling,
                   paymentmethod,
                   CAST(monthlycharges AS REAL),
                   CAST(totalcharges AS REAL)
                   )
           ) AS prediction
FROM pgml.telco_churn;