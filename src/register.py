import mlflow
from mlflow.tracking import MlflowClient

def register_model(run_id, model_name="iris_classifier",
                   accuracy_threshold=0.85, tracking_uri="http://localhost:5000"):
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy", 0)
    if accuracy < accuracy_threshold:
        raise ValueError(f"❌ Accuracy {accuracy:.4f} below threshold {accuracy_threshold}")
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    version = result.version
    client.transition_model_version_stage(name=model_name, version=version, stage="Staging")
    client.update_model_version(name=model_name, version=version,
        description=f"Auto-registered. Accuracy={accuracy:.4f}")
    print(f"✅ Registered {model_name} v{version} → Staging (acc={accuracy:.4f})")
    return version

if __name__ == "__main__":
    pass