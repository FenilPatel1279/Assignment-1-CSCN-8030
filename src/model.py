
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import os

def _ensure_dir(path):
    """Ensure output directories exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

def train_model(data):
    """
    Train a Logistic Regression model on wildfire dataset.
    Returns: trained model, X_test, y_test, predictions, probabilities
    """
    X = data[["Temperature", "Humidity", "SmokeLevel", "SatelliteHeat"]]
    y = data["Wildfire"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    return model, X_test, y_test, y_pred, y_prob


def evaluate_model(y_test, y_pred, model, X_test, save_path="../outputs/charts/confusion_matrix.png"):
    """Evaluate model with accuracy, classification report, and confusion matrix."""
    _ensure_dir(save_path)
    
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.show()


def show_sample_predictions(X_test, y_test, y_pred, y_prob, n=10):
    """Show sample predictions with probabilities for interpretation."""
    results = X_test.copy()
    results["Actual"] = y_test.values
    results["Predicted"] = y_pred
    results["Fire_Probability"] = y_prob
    print("\nSample Predictions:\n", results.head(n))
    return results.head(n)

