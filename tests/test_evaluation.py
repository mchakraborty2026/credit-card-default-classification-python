from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from evaluation import calculate_specificity, evaluate_model


class DummyModel:
    """Simple model that returns predefined predictions."""

    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, X):
        return self.predictions


def test_calculate_specificity():
    """Check that specificity is calculated correctly."""

    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 0, 1, 1, 1]

    specificity = calculate_specificity(y_true, y_pred)

    assert specificity == 0.75


def test_evaluate_model():
    """Check that classification metrics are calculated correctly."""

    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    predictions = [0, 0, 0, 1, 0, 1, 1, 1]

    model = DummyModel(predictions)

    X_test = pd.DataFrame(
        {"feature": range(8)}
    )

    results = evaluate_model(
        "Dummy Model",
        model,
        X_test,
        y_true,
    )

    assert results["Model"] == "Dummy Model"
    assert results["Accuracy"] == 0.75
    assert results["Recall"] == 0.75
    assert results["Specificity"] == 0.75
    assert results["Precision"] == 0.75
    assert results["F1"] == 0.75
