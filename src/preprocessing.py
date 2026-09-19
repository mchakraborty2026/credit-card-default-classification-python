from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "Default"

REQUIRED_COLUMNS = [
    "ID",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "default.payment.next.month",
]


def load_data(file_path):
    """
    Load the UCI Credit Card Default dataset from a CSV file.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    return pd.read_csv(file_path)


def validate_columns(data):
    """
    Check that the dataset contains the columns required
    for preprocessing and modeling.
    """
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def clean_data(data):
    """
    Prepare the raw credit-card dataset for analysis.

    Steps:
    - Rename the target variable.
    - Remove the customer ID.
    - Recode demographic categorical variables.
    """

    validate_columns(data)

    cleaned = data.copy()

    cleaned = cleaned.rename(
        columns={
            "default.payment.next.month": TARGET_COLUMN
        }
    )

    cleaned = cleaned.drop(columns=["ID"])

    cleaned["SEX"] = cleaned["SEX"].map(
        {
            1: "Male",
            2: "Female",
        }
    )

    cleaned["EDUCATION"] = cleaned["EDUCATION"].replace(
        {
            1: "Graduate_School",
            2: "University",
            3: "High_School",
            0: "Other",
            4: "Other",
            5: "Other",
            6: "Other",
        }
    )

    cleaned["MARRIAGE"] = cleaned["MARRIAGE"].replace(
        {
            1: "Married",
            2: "Single",
            0: "Other",
            3: "Other",
        }
    )

    return cleaned


def split_data(
    data,
    test_size=0.30,
    random_state=123,
):
    """
    Create a stratified training and test split.

    Stratification preserves the proportion of default
    and non-default cases in both datasets.
    """

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
