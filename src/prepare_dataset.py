# src/prepare_dataset.py
import os
import pandas as pd

os.makedirs("data/raw", exist_ok=True)
train_path = os.path.join("data", "raw", "train.csv")

if not os.path.exists(train_path):
    print("train.csv not found -> writing a small dummy dataset to data/raw/train.csv")
    df = pd.DataFrame({
        "Survived": [0, 1, 1, 0],
        "Pclass":   [3, 1, 3, 2],
        "Sex":      ["male", "female", "female", "male"],
        "Age":      [22, 38, 26, 35],
        "Fare":     [7.25, 71.83, 7.92, 8.05],
        "Embarked": ["S", "C", "Q", "S"]
    })
    df.to_csv(train_path, index=False)
    print(f"wrote dummy train.csv ({len(df)} rows) to {train_path}")
else:
    print(f"train.csv already present at {train_path} (skipping creation)")
