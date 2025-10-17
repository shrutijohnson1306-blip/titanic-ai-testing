# src/data_cleaning.py
import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Fill Age with median
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())
    # Fill Embarked with mode
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    # Fill Fare with median (useful for test set)
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    # Drop Cabin due to many missing values
    if 'Cabin' in df.columns:
        df = df.drop(columns=['Cabin'])
    return df
