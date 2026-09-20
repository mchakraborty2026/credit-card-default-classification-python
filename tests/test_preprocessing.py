from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from preprocessing import clean_data, split_data


def create_sample_data():
    """Create a small dataset for preprocessing tests."""

    return pd.DataFrame(
        {
            "ID": range(1, 11),
            "LIMIT_BAL": [
                20000,
                120000,
                90000,
                50000,
                50000,
                50000,
                500000,
                100000,
                140000,
                20000,
            ],
            "SEX": [2, 2, 2, 2, 1, 1, 1, 2, 2, 1],
            "EDUCATION": [2, 2, 2, 2, 2, 1, 1, 2, 3, 3],
            "MARRIAGE": [1, 2, 2, 1, 1, 2, 2, 2, 1, 2],
            "default.payment.next.month": [
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
        }
    )


def test_clean_data():
    """Check that cleaning performs the expected transformations."""

    raw_data = create_sample_data()

    cleaned = clean_data(raw_data)

    assert "ID" not in cleaned.columns
    assert "Default" in cleaned.columns
    assert "default.payment.next.month" not in cleaned.columns

    assert cleaned.loc[0, "SEX"] == "Female"
    assert cleaned.loc[4, "SEX"] == "Male"

    assert cleaned.loc[0, "EDUCATION"] == "University"
    assert cleaned.loc[5, "EDUCATION"] == "Graduate_School"

    assert cleaned.loc[0, "MARRIAGE"] == "Married"
    assert cleaned.loc[1, "MARRIAGE"] == "Single"


def test_split_data():
    """Check that the train/test split returns the expected sizes."""

    raw_data = create_sample_data()
    cleaned = clean_data(raw_data)

    X_train, X_test, y_train, y_test = split_data(
        cleaned,
        test_size=0.20,
        random_state=123,
    )

    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2

    assert "Default" not in X_train.columns
    assert set(y_train.unique()).issubset({0, 1})
