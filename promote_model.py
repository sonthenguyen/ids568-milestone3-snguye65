import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

# Step 3: Promote to production
client.set_registered_model_alias("iris_classifier", "production", "1")
print("✅ iris_classifier v1 → Production")

model = client.get_registered_model("iris_classifier")
print(f"Current aliases: {model.aliases}")
