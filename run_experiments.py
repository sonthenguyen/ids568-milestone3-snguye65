from src.preprocess import preprocess_data
from src.train import train_model

preprocess_data(output_dir="data")

experiments = [
    {"n_estimators": 50,  "max_depth": 2, "min_samples_split": 2},
    {"n_estimators": 100, "max_depth": 3, "min_samples_split": 2},
    {"n_estimators": 150, "max_depth": 5, "min_samples_split": 3},
    {"n_estimators": 200, "max_depth": 7, "min_samples_split": 4},
    {"n_estimators": 100, "max_depth": None, "min_samples_split": 5},
]

for i, params in enumerate(experiments, 1):
    print(f"\n--- Run {i}/5 ---")
    train_model(data_dir="data", **params)