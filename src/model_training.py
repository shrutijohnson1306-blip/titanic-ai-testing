import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(X, y):
    # Split into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Improved Random Forest configuration
    model = RandomForestClassifier(
        n_estimators=200,       # more trees = better stability
        max_depth=8,            # prevent overfitting
        min_samples_split=4,    # ensures balanced splits
        random_state=42,
        class_weight='balanced' # helps with uneven survival rates
    )

    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    return model, accuracy

