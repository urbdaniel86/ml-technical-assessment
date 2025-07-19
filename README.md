# ML Technical Assessment: Sportsbook Churn Prediction

## Overview

This project explores, models, and productionizes churn prediction for sportsbook customers, as part of a Senior ML Engineer technical assessment.

## Repository Structure

```
├── data/
│   ├── clean_data.csv        # Cleaned CSV files for modeling
│   └── sample_data__technical_assessment_1.xlsx # Original Excel dataset
├── notebooks/                # Jupyter notebooks for EDA and modeling
│   ├── 01_eda_and_data_processing.ipynb
│   ├── 02_model1_baseline.ipynb
│   └── 03_model2_early_churn.ipynb
├── src/                      # Python scripts (mockups)
├── outputs/                  # Saved models and results artifacts (mockups)
├── environment.yml           # Conda environment specification
└── README.md                 # Project overview and instructions
```

## Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone <repo_url>
   cd ml-technical-assessment
   ```
2. **Create and activate the Conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate sportsbook-churn-env
   ```

## Section 1: Data Exploration (notebooks/01_eda_and_data_processing.ipynb)
Main findings:
- **Missingness & Cleaning**:
    - Dropped ~40% of the data due to no qp_date/ftd_date, yet they have deposits/handles/ngr (inconsistency)
    - No duplicated records were found (after cleaning)
- **Feature Analysis**:
    - Categorical features described and plotted frequency of appearance
    - Numerical features plotted. Highly skewed data with outliers (e.g. few players deposit/bet a high amounts). Log scaling for better visualization
    - Other analyses: active months per user, registration to betting funnel
- **Assumptions**:
    - There must be a business logic that explains deposits/handles/ngr with no qp_date/ftd_date, like promotions, credits or bonuses to attract new players before they make any deposits themselves
    - total_deposit tracks how much money they deposit on activity_month but not a cummulative deposit (i.e. they could have deposited in the past and they used the money later to place bets)
- **Conclusions**:
    - This is a limited dataset meant as an example. More data, a longer timespan, and talks with a business team would be needed to interpret and process the data accordingly

## Section 2: Predictive Modeling

### Model 1: Baseline Churn (02_model1_baseline.ipynb)

- **Objective**: Predict churn probability for months 0-60
- **Features**:
  - Time: `months_active`
  - Monetary: `total_deposit`, `total_handle`, `total_ngr`
  - Encoded categoricals: `brand_id`,`ben_login_id`,`player_req_product`,`tracker_id`
- **Algorithm**: XGBoost (powerful, straightforward, tunable)
- **Evaluation**: accuracy, confusion matrix, retention curve comparison (actual vs. predicted), SHAP
- **Results**:
    - Accuracy: 0.7468354430379747 (Ok)
    - ROC AUC: 0.6402782423360844 (Poor)
    - SHAP: tracker_id was the most important variable, followed by total_ngr and total_handle
- **Next steps**:
    - Hyperparameter tuning (GridSearch, Optuna) to improve metrics
    - Cross-validation
    - Production/MLOps

### Model 2: Early‑Churn Prediction (03_model2_early_churn.ipynb)

- **Objective**: Identify users who will churn quickly
- **Threshold**: churn if max active month ≤ 2 (i.e., no activity after month 2). Enough time to get data + early enough to predict quickly
- **Features**:
  - Time: `months_active`
  - Monetary:`total_deposit_2m`, `total_handle_2m`, `total_ngr_2m`
  - Encoded categoricals: `brand_id`,`ben_login_id`,`player_req_product`,`tracker_id`
- **Algorithm**: XGBoost (powerful, straightforward, tunable)
- **Evaluation**: ROC AUC, precision, recall, accuracy, confusion matrix, SHAP
- **Results**:
    - ROC AUC: 0.559814376868502 (Very Poor)
    - Precision: 0.8210227272727273 (Good)
    - Recall: 0.9413680781758957 (Great)
    - Accuracy: 0.7868421052631579 (Ok)
    - SHAP: again, tracker_id was the most important variable, followed by total_ngr and total_handle
- **Next steps**:
    - Hyperparameter tuning (GridSearch, Optuna) to improve metrics
    - Further feature engineering (e.g. new features based on time series)
    - Cross-validation
    - Production/MLOps

### Results Summary

| Model             | Accuracy | ROC AUC | Precision | Recall |
| ----------------- | -------- | ------- | --------- | ------ |
| 1. Baseline Churn | 0.75     | 0.64    |  —        |  —     |
| 2. Early‑Churn    | 0.79     | 0.56    | 0.82      | 0.94   |

## Section 3: Production

### 1. Model Evaluation
To evaluate the effectiveness of the model over time as new data continues to come in, I'd continually:
- Log and track metrics (ROC AUC, precision, recall, churn rate) over time in a dashboard (e.g. Grafana)
- Retrain when performance drops below a threshold (e.g. AUC drops 5% from baseline) or on a fixed period (e.g. monthly)
- Compare new vs. old model performance before swapping in production
Alternatives that I'd consider with more time/data:
- As mentioned above, I'd perform several rounds of hyperparameter tuning, feature engineering, and cross-validation
- We could also explore deep learning models for sequential data (e.g. RNN)

### 2. Production & Scale
I'd approach the fact that the model needs to be repeated ~2000 times from data engineering + mlops perspective:
- Customize ETLs for each source, depending on data structure
- Parameterize via YAML/JSON config files for each source (e.g. partner, geography, etc.)
- Use orchestration (e.g. Dagster) to cycle over configs

For saving the outputs of the models:
- Register trained models in an ML registry (e.g. MLFlow) to keep track of versioned artifacts
- Save the batch outputs in a cloud bucket (e.g. S3) including metadata, predictions, and metrics
- The format could be JSON or Parquet, depending on the size

For the OOP model as an artifact, I would:
- Create a Python script for model training (see src/churn_model_mockup.py)
- Call via console. For example:
```bash
   python src/churn_model_mockup.py --train --data data/clean_data.csv --out outputs/churn_model_artifact_mockup.pkl
   ```
