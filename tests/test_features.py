# tests/test_features.py
import pandas as pd
import numpy as np
from src.features import encode_and_scale

def test_encode_and_scale_creates_new_columns_and_scales_values():
    df = pd.DataFrame({
        'Age': [22, 38],
        'Fare': [7.25, 71.83],
        'Sex': ['male', 'female'],
        'Embarked': ['S', 'C']
    })
    processed = encode_and_scale(df)
    assert any(col.startswith('Sex_') for col in processed.columns)
    assert any(col.startswith('Embarked_') for col in processed.columns)
    np.testing.assert_almost_equal(processed['Age'].mean(), 0, decimal=1)
    np.testing.assert_almost_equal(processed['Fare'].mean(), 0, decimal=1)
    assert processed.isnull().sum().sum() == 0
