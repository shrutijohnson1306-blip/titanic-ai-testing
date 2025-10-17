# tests/test_integration_pipeline.py
import pandas as pd
from src.data_cleaning import clean_data
from src.features import encode_and_scale

def test_cleaning_and_encoding_pipeline():
    df = pd.DataFrame({
        'Age': [22, None, 30],
        'Fare': [7.25, 71.83, None],
        'Sex': ['male', 'female', 'female'],
        'Embarked': ['S', None, 'C'],
        'Cabin': [None, 'B45', None]
    })
    cleaned = clean_data(df)
    processed = encode_and_scale(cleaned)
    assert processed.isnull().sum().sum() == 0
    assert any(col.startswith('Sex_') for col in processed.columns)
    assert any(col.startswith('Embarked_') for col in processed.columns)
    assert 'Cabin' not in processed.columns
