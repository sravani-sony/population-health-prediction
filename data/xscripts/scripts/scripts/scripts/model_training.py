from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_model(X, y):
    """
    Trains a Random Forest model and prints classification metrics.

    Parameters:
    - X: Features dataframe
    - y: Target values

    Returns:
    - model: Trained Random Forest model
    """

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    # Evaluation
    print("Model Performance:")
    print(classification_report(y_test, preds))

    return model
