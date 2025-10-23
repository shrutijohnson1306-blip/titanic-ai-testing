import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.data_cleaning import clean_data
from src.features import encode_and_scale


def train_model(X, y):
    """
    Trains a RandomForest model and evaluates accuracy on test data.
    Returns the trained model and accuracy score.
    """
    # Split into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ Improved Random Forest configuration
    model = RandomForestClassifier(
        n_estimators=200,        # more trees = better stability
        max_depth=8,             # prevent overfitting
        min_samples_split=4,     # ensures balanced splits
        random_state=42,
        class_weight='balanced'  # handles uneven survival rates
    )

    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model accuracy: {accuracy:.4f}")

    return model, accuracy


if __name__ == "__main__":
    """
    This block runs automatically during CI/CD:
    - Loads Titanic dataset
    - Cleans and encodes it
    - Trains model
    - Saves model as artifact
    """
    data_path = os.path.join(os.getcwd(), "data", "raw", "train.csv")

    if os.path.exists(data_path):
        print("🚀 Starting training process...")
        df = pd.read_csv(data_path)

        # Clean and encode data
        df = clean_data(df)
        df = encode_and_scale(df)

        # Separate features and target
        X = df.drop(columns=["Survived"])
        y = df["Survived"]

        # Train model
        model, acc = train_model(X, y)

        # Save trained model
        os.makedirs("artifacts", exist_ok=True)
        model_path = os.path.join("artifacts", "titanic_model.pkl")
        joblib.dump(model, model_path)

        print(f"💾 Model saved to: {model_path}")
        print(f"🎯 Final Model Accuracy: {acc:.4f}")
    else:
        print("⚠️ train.csv not found — skipping model training.")


