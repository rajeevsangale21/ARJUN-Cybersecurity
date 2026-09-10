import numpy as np


def calculate_mse(y_true, y_pred):
    """
    Calculate mean squared error.
    """

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    return float(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )


def calculate_mae(y_true, y_pred):
    """
    Calculate mean absolute error.
    """

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    return float(
        np.mean(
            np.abs(y_true - y_pred)
        )
    )


def calculate_rmse(y_true, y_pred):
    """
    Calculate root mean squared error.
    """

    mse = calculate_mse(
        y_true,
        y_pred
    )

    return float(
        np.sqrt(mse)
    )


def per_feature_mae(
    y_true,
    y_pred,
    feature_names
):
    """
    Calculate MAE for every state feature.
    """

    y_true = np.asarray(
        y_true,
        dtype=np.float32
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float32
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have "
            "the same shape."
        )

    if y_true.ndim != 2:
        raise ValueError(
            "Expected arrays with shape "
            "(samples, features)."
        )

    if y_true.shape[1] != len(feature_names):
        raise ValueError(
            "Number of feature names does not "
            "match state dimension."
        )

    errors = np.mean(
        np.abs(y_true - y_pred),
        axis=0
    )

    result = {}

    for name, error in zip(
        feature_names,
        errors
    ):
        result[name] = float(error)

    return result


def persistence_baseline(
    previous_states
):
    """
    Persistence baseline:
    predict that the next state equals
    the current state.
    """

    previous_states = np.asarray(
        previous_states,
        dtype=np.float32
    )

    if previous_states.ndim != 2:
        raise ValueError(
            "previous_states must have "
            "shape (samples, features)."
        )

    return previous_states.copy()


def evaluate_world_model(
    y_true,
    y_pred,
    baseline_pred=None,
    feature_names=None
):
    """
    Generate complete World Model evaluation.
    """

    result = {
        "mse": calculate_mse(
            y_true,
            y_pred
        ),

        "mae": calculate_mae(
            y_true,
            y_pred
        ),

        "rmse": calculate_rmse(
            y_true,
            y_pred
        )
    }

    if baseline_pred is not None:

        baseline_mse = calculate_mse(
            y_true,
            baseline_pred
        )

        baseline_mae = calculate_mae(
            y_true,
            baseline_pred
        )

        result["baseline_mse"] = baseline_mse

        result["baseline_mae"] = baseline_mae

        if baseline_mse > 0:

            result["mse_improvement_percent"] = (
                (baseline_mse - result["mse"])
                / baseline_mse
                * 100.0
            )

        else:

            result[
                "mse_improvement_percent"
            ] = 0.0

    if feature_names is not None:

        result[
            "per_feature_mae"
        ] = per_feature_mae(
            y_true,
            y_pred,
            feature_names
        )

    return result