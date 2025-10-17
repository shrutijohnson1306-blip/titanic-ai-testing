# tests/test_model_regression.py
import os
import pandas as pd
from src.data_cleaning import clean_data
from src.features import encode_and_scale
from src.model_training import train_model

def test_model_accuracy_above_threshold():
    data_path = os.path.join(os.getcwd(), "data", "raw", "train.csv")
    assert os.path.exists(data_path), f"train.csv not found at {data_path}"
    df = pd.read_csv(data_path)
    df = clean_data(df)
    df = encode_and_scale(df)
    y = df['Survived']
    X = df.drop(columns=['Survived'], errors='ignore')
    model, acc = train_model(X, y)
    assert acc >= 0.75, f"Model accuracy dropped! Expected ≥ 0.75, got {acc:.2f}"
