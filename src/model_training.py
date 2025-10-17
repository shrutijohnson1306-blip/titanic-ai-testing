# src/model_training.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(X, y):
    # Drop irrelevant columns if they exist
    drop_cols = [c for c in ['PassengerId', 'Name', 'Ticket'] if c in X.columns]
    X = X.drop(columns=drop_cols, errors='ignore')

    # Keep only numeric columns
    X = X.select_dtypes(include=['int64', 'float64'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

