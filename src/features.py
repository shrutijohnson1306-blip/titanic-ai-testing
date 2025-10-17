# src/features.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def encode_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # One-hot encode some categorical columns if present
    cat_cols = [c for c in ['Sex', 'Embarked'] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    # Scale numeric columns if present
    num_cols = [c for c in ['Age', 'Fare'] if c in df.columns]
    if num_cols:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    return df
