import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def encode_and_scale(df):
    """
    Encode categorical columns (OneHotEncoder) and scale numeric columns.
    This version ensures compatibility with tests and ML training.
    """
    df = df.copy()
    
    # --- One-hot encode categorical columns ---
    cat_cols = ['Sex', 'Embarked']
    existing_cat_cols = [c for c in cat_cols if c in df.columns]
    if existing_cat_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(df[existing_cat_cols].astype(str))
        encoded_cols = encoder.get_feature_names_out(existing_cat_cols)
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
        df = pd.concat([df.drop(columns=existing_cat_cols), encoded_df], axis=1)

    # --- Scale numeric columns (only existing ones) ---
    num_cols = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
    existing_num_cols = [c for c in num_cols if c in df.columns]
    if existing_num_cols:
        scaler = StandardScaler()
        df[existing_num_cols] = scaler.fit_transform(df[existing_num_cols])

    return df


