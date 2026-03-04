import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os, json

def preprocess_data(output_dir="data", test_size=0.2, random_state=42):
    os.makedirs(output_dir, exist_ok=True)
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
    meta = {"rows_train": len(train_df), "rows_test": len(test_df), "features": list(iris.feature_names)}
    with open(f"{output_dir}/meta.json", "w") as f:
        json.dump(meta, f)
    print(f"✅ Preprocessed: {len(train_df)} train, {len(test_df)} test rows")
    return output_dir

if __name__ == "__main__":
    preprocess_data()