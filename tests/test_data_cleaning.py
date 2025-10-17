# tests/test_data_cleaning.py
import pandas as pd
from src.data_cleaning import clean_data

def test_clean_data_fills_missing_values():
    df = pd.DataFrame({
        'Age': [22, None, 30],
        'Embarked': ['S', None, 'C'],
        'Fare': [7.25, None, 8.05],
        'Cabin': ['B45', None, None]
    })
    cleaned = clean_data(df)
    assert cleaned['Age'].isnull().sum() == 0
    assert cleaned['Embarked'].isnull().sum() == 0
    assert cleaned['Fare'].isnull().sum() == 0
    assert 'Cabin' not in cleaned.columns
