from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 123


def build_preprocessor(X):
    """
    Create preprocessing steps for numeric and categorical predictors.

    Numeric variables are standardized.
    Categorical variables are converted to dummy variables using
    one-hot encoding.
    """

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_columns = X.select_dtypes(
        exclude=["object", "category"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_columns,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_columns,
            ),
        ],
        sparse_threshold=0.0,
    )

    return preprocessor


def create_pipeline(X, model):
    """
    Combine preprocessing and a machine-learning model
    into one reproducible pipeline.
    """

    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor(X)),
            ("model", model),
        ]
    )


def build_models(X):
    """
    Create the classification models used in this project.
    """

    models = {
        "L1 Logistic Regression": create_pipeline(
            X,
            LogisticRegressionCV(
                Cs=10,
                cv=10,
                penalty="l1",
                solver="liblinear",
                scoring="accuracy",
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),

        "Bagging": create_pipeline(
            X,
            BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    random_state=RANDOM_STATE
                ),
                n_estimators=500,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),

        "Random Forest": create_pipeline(
            X,
            RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),

        "Gradient Boosting": create_pipeline(
            X,
            GradientBoostingClassifier(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=3,
                min_samples_leaf=10,
                random_state=RANDOM_STATE,
            ),
        ),
    }

    return models


def train_models(models, X_train, y_train):
    """
    Fit each model on the training dataset.
    """

    trained_models = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[model_name] = model

    return trained_models
