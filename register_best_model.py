from src.register import register_model
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

exp = client.get_experiment_by_name("iris_classification")
runs = client.search_runs(exp.experiment_id, order_by=["metrics.accuracy DESC"], max_results=1)
best_run = runs[0]
best_run_id = best_run.info.run_id
acc = best_run.data.metrics["accuracy"]

print(f"Best run: {best_run_id[:8]} | accuracy={acc:.4f}")

# Step 1: Register
register_model(best_run_id)

# Step 2: Set staging alias
client.set_registered_model_alias("iris_classifier", "staging", "1")
print("✅ iris_classifier v1 → Staging")
