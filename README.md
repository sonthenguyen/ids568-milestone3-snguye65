# IDS568 Milestone 3: Workflow Automation & Experiment Tracking

## Overview
This project implements a complete automated ML workflow using Airflow, MLflow, and GitHub Actions CI/CD for the Iris classification dataset.

## Architecture
```
Data → Preprocess → Train → Register → Staging → Production
         (Airflow)   (Airflow)  (MLflow)   (MLflow Registry)
```

## Project Structure
```
├── dags/train_pipeline.py        # Airflow DAG
├── src/preprocess.py             # Data preprocessing
├── src/train.py                  # Model training with MLflow logging
├── src/register.py               # MLflow model registration
├── run_experiments.py            # Runs 5 experiments with different hyperparameters
├── register_best_model.py        # Registers best model to MLflow registry
├── promote_model.py              # Promotes model to production
├── model_validation.py           # CI quality gate script
└── .github/workflows/            # GitHub Actions CI/CD
```

## Setup Instructions

### 1. Install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start MLflow server
```bash
mlflow server --host 0.0.0.0 --port 5000
```

### 3. Initialize Airflow
```bash
export AIRFLOW_HOME=$(pwd)/airflow_home
airflow db init
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
```

### 4. Start Airflow
```bash
export AIRFLOW_HOME=$(pwd)/airflow_home
airflow webserver --port 8080
airflow scheduler
```

## How to Run the Pipeline

### Option 1: Run experiments manually
```bash
python3 run_experiments.py        # Run 5 experiments
python3 register_best_model.py    # Register best model
python3 promote_model.py          # Promote to production
```

### Option 2: Run via Airflow DAG
```bash
airflow dags unpause iris_train_pipeline
airflow dags trigger iris_train_pipeline
```

## DAG Design
- **preprocess_data**: Loads Iris dataset, splits into train/test, saves CSVs and metadata
- **train_model**: Trains RandomForest with MLflow logging, pushes run_id via XCom
- **register_model**: Registers best model to MLflow registry if accuracy > 0.85
- **Idempotency**: Tasks use fixed random seeds and versioned output paths
- **Retry config**: 2 retries with 2-minute delay on failure

## CI/CD Model Governance
- Every push to main triggers automated training and validation
- Quality gates: accuracy > 0.85, f1 > 0.84
- Pipeline fails if thresholds not met, preventing bad models from being registered

## Experiment Tracking
- All runs tracked in MLflow with hyperparameters, metrics, and model artifacts
- Best model selected based on accuracy and registered to MLflow registry
- Model progresses through None → Staging → Production aliases

## Rollback Procedure
1. Identify last good version in MLflow registry
2. Re-assign production alias to previous version:
```bash
python3 -c "
import mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.set_registered_model_alias('iris_classifier', 'production', 'PREVIOUS_VERSION')
"
```

## Monitoring Recommendations
- Track accuracy drift over time using MLflow metrics
- Alert if accuracy drops below 0.85 threshold
- Monitor prediction latency and data distribution shifts
