import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_specificity(y_true, y_pred):
    """
    Calculate specificity:

    Specificity = TN / (TN + FP)
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    return tn / (tn + fp)


def evaluate_model(
    model_name,
    model,
    X_test,
    y_test,
):
    """
    Generate predictions and calculate classification metrics
    for one trained model.
    """

    predictions = model.predict(X_test)

    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "Recall": recall_score(
            y_test,
            predictions,
        ),
        "Specificity": calculate_specificity(
            y_test,
            predictions,
        ),
        "Precision": precision_score(
            y_test,
            predictions,
        ),
        "F1": f1_score(
            y_test,
            predictions,
        ),
    }

    return results


def compare_models(
    trained_models,
    X_test,
    y_test,
):
    """
    Evaluate every trained model and return
    a comparison table.
    """

    results = []

    for model_name, model in trained_models.items():

        model_results = evaluate_model(
            model_name,
            model,
            X_test,
            y_test,
        )

        results.append(model_results)

    comparison = pd.DataFrame(results)

    metric_columns = [
        "Accuracy",
        "Recall",
        "Specificity",
        "Precision",
        "F1",
    ]

    comparison[metric_columns] = (
        comparison[metric_columns].round(4)
    )

    return comparison


def get_confusion_matrix(
    model,
    X_test,
    y_test,
):
    """
    Return the confusion matrix for a trained model.
    """

    predictions = model.predict(X_test)

    return confusion_matrix(
        y_test,
        predictions,
    )
