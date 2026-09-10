import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def calculate_metrics(
    y_true,
    y_pred
):
    """
    Calculate binary classification metrics.

    Labels:
        0 = Normal
        1 = Attack
    """

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    if (tn + fp) > 0:
        false_positive_rate = (
            fp / (fp + tn)
        )
    else:
        false_positive_rate = 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }


def print_metrics(
    name,
    metrics
):
    """
    Display metrics in a readable format.
    """

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print(
        f"Accuracy          : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision         : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall            : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score          : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"False Positive Rate: "
        f"{metrics['false_positive_rate']:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        f"  TN: {metrics['true_negatives']}"
        f"    FP: {metrics['false_positives']}"
    )

    print(
        f"  FN: {metrics['false_negatives']}"
        f"    TP: {metrics['true_positives']}"
    )