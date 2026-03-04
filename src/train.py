import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import json

def train_model(data_dir="data", n_estimators=100, max_depth=3,
                min_samples_split=2, tracking_uri="http://localhost:5000"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("iris_classification")
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df  = pd.read_csv(f"{data_dir}/test.csv")
    feature_cols = [c for c in train_df.columns if c != "target"]
    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test,  y_test  = test_df[feature_cols],  test_df["target"]
    with open(f"{data_dir}/meta.json") as f:
        data_meta = json.load(f)
    with mlflow.start_run() as run:
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("data_version", "v1")
        mlflow.log_param("train_rows", data_meta["rows_train"])
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average="weighted")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")
        print(f"✅ Run {run.info.run_id[:8]} | acc={acc:.4f} | f1={f1:.4f}")
        return run.info.run_id, acc, f1

if __name__ == "__main__":
    train_model()
