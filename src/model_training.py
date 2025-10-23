import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ✅ Ensure 'src' folder is discoverable when run in GitHub Actions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_cleaning import clean_data
from src.features import encode_and_scale


def train_model(X, y):
    """
    Trains a RandomForest model and evaluates accuracy on test data.
    Returns the trained model and accuracy score.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,        # more trees = better stability
        max_depth=8,             # prevent overfitting
        min_samples_split=4,     # ensures balanced splits
        random_state=42,
        class_weight='balanced'  # handles uneven survival rates
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model accuracy: {accuracy:.4f}")
    return model, accuracy


if __name__ == "__main__":
    """
    This block runs automatically during CI/CD:
    - Ensures dataset exists
    - Cleans and encodes it
    - Trains model
    - Saves model artifact
    """
    data_path = os.path.join("data", "raw", "train.csv")

    # ✅ Ensure dataset exists
    if not os.path.exists(data_path):
        print("⚠️ train.csv not found — creating dummy dataset for CI...")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df = pd.DataFrame({
            "Survived": [0, 1, 1, 0],
            "Pclass": [3, 1, 3, 2],
            "Sex": ["male", "female", "female", "male"],
            "Age": [22, 38, 26, 35],
            "Fare": [7.25, 71.83, 7.92, 8.05],
            "Embarked": ["S", "C", "Q", "S"]
        })
        df.to_csv(data_path, index=False)
        print("✅ Dummy dataset created successfully.")

    # ✅ Load dataset
    print("🚀 Starting training process...")
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

    # ✅ Clean and encode data
    df = clean_data(df)
    df = encode_and_scale(df)

    # ✅ Split into features and target
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # ✅ Train model
    model, acc = train_model(X, y)

    # ✅ Save trained model
    os.makedirs("artifacts", exist_ok=True)
    model_path = os.path.join("artifacts", "titanic_model.pkl")
    joblib.dump(model, model_path)

    print(f"💾 Model saved to: {model_path}")
    print(f"🎯 Final Model Accuracy: {acc:.4f}")



