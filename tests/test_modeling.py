from pathlib import Path
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from modeling import build_models, create_pipeline


def create_sample_features():
    """Create sample numeric and categorical predictors."""

    return pd.DataFrame(
        {
            "LIMIT_BAL": [20000, 50000, 100000, 150000],
            "AGE": [24, 35, 42, 50],
            "SEX": ["Female", "Male", "Female", "Male"],
            "EDUCATION": [
                "University",
                "Graduate_School",
                "High_School",
                "University",
            ],
        }
    )


def test_build_models():
    """Check that all expected models are created."""

    X = create_sample_features()

    models = build_models(X)

    expected_models = {
        "L1 Logistic Regression",
        "Bagging",
        "Random Forest",
        "Gradient Boosting",
    }

    assert set(models.keys()) == expected_models

    for model in models.values():
        assert isinstance(model, Pipeline)


def test_create_pipeline_can_fit():
    """Check that a preprocessing/model pipeline can fit and predict."""

    X = create_sample_features()
    y = [0, 0, 1, 1]

    pipeline = create_pipeline(
        X,
        LogisticRegression(max_iter=1000),
    )

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})
