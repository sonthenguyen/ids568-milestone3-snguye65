from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import preprocess_data
from src.train import train_model
from src.register import register_model

TRACKING_URI = "http://localhost:5000"

default_args = {
    "owner": "mlops-student",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "on_failure_callback": lambda ctx: print(f"❌ Task failed: {ctx['task_instance'].task_id}"),
}

def task_preprocess(**kwargs):
    preprocess_data(output_dir="/tmp/iris_data")

def task_train(**kwargs):
    run_id, acc, f1 = train_model(data_dir="/tmp/iris_data",
        n_estimators=100, max_depth=3, tracking_uri=TRACKING_URI)
    kwargs["ti"].xcom_push(key="run_id", value=run_id)

def task_register(**kwargs):
    run_id = kwargs["ti"].xcom_pull(key="run_id", task_ids="train_model")
    register_model(run_id, model_name="iris_classifier", tracking_uri=TRACKING_URI)

with DAG(
    dag_id="iris_train_pipeline",
    default_args=default_args,
    description="Preprocess → Train → Register",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "milestone3"],
) as dag:
    t1 = PythonOperator(task_id="preprocess_data", python_callable=task_preprocess)
    t2 = PythonOperator(task_id="train_model",     python_callable=task_train)
    t3 = PythonOperator(task_id="register_model",  python_callable=task_register)
    t1 >> t2 >> t3