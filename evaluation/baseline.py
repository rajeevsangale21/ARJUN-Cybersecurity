"""
ARJUN - Logistic Regression Baseline

Fair baseline for the ARJUN World Model.

The baseline predicts whether the next network state
is an attack state.

Features:
    - Latest observed state
    - Short-term state movement

Evaluation:
    - Precision
    - Recall
    - F1
    - False Positive Rate (FPR)

The class interface is compatible with evaluate.py:
    baseline.fit(X, y)
    baseline.predict(X)
    baseline.predict_proba(X)
    baseline.evaluate(X, y)
"""

from pathlib import Path
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


class LogisticRegressionBaseline:
    """
    Logistic Regression baseline for ARJUN.

    Accepts either:

        X = (samples, features)

    or:

        X = (samples, sequence_length, features)

    For sequences, the latest state and the delta between
    the latest two states are used.
    """

    def __init__(
        self,
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ):
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

        self.scaler = StandardScaler()

        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
        )

        self.feature_names = None
        self.is_fitted = False

    # ============================================================
    # DATA PREPARATION
    # ============================================================

    @staticmethod
    def _prepare_X(X):
        """
        Convert input into a 2D feature matrix.

        3D sequence:
            (samples, sequence_length, features)

        becomes:
            latest state + latest delta

        2D input is used directly.
        """

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        if X.ndim == 3:

            if X.shape[1] < 2:
                raise ValueError(
                    "Sequence input requires at least "
                    "two time steps."
                )

            latest = X[:, -1, :]
            previous = X[:, -2, :]

            delta = latest - previous

            X = np.concatenate(
                [
                    latest,
                    delta,
                ],
                axis=1,
            )

        elif X.ndim != 2:

            raise ValueError(
                "X must have shape "
                "(samples, features) or "
                "(samples, sequence, features). "
                f"Received: {X.shape}"
            )

        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return X.astype(np.float32)

    @staticmethod
    def _prepare_y(y):

        y = np.asarray(y)

        if y.ndim > 1:
            y = y.reshape(-1)

        result = []

        for value in y:

            if isinstance(value, str):

                text = value.strip().lower()

                normal_values = {
                    "benign",
                    "normal",
                    "0",
                    "false",
                    "none",
                }

                result.append(
                    0
                    if text in normal_values
                    else 1
                )

            else:

                try:

                    result.append(
                        0
                        if float(value) == 0
                        else 1
                    )

                except (
                    ValueError,
                    TypeError,
                ):

                    result.append(1)

        return np.asarray(
            result,
            dtype=np.int32,
        )

    # ============================================================
    # TRAINING
    # ============================================================

    def fit(
        self,
        X,
        y,
        feature_names=None,
        validation_data=None,
    ):

        X = self._prepare_X(X)
        y = self._prepare_y(y)

        if len(X) == 0:
            raise ValueError(
                "Training data is empty."
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y have different "
                "numbers of samples."
            )

        unique_classes = np.unique(y)

        if len(unique_classes) < 2:
            raise ValueError(
                "Logistic Regression requires both "
                "benign and attack samples. "
                f"Found classes: {unique_classes.tolist()}"
            )

        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else None
        )

        # IMPORTANT:
        # Fit scaler ONLY on training data.
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(
            X_scaled,
            y,
        )

        self.is_fitted = True

        return self

    # ============================================================
    # PREDICTION
    # ============================================================

    def predict_proba(self, X):

        if not self.is_fitted:
            raise RuntimeError(
                "Classifier has not been fitted."
            )

        X = self._prepare_X(X)

        X_scaled = self.scaler.transform(X)

        probabilities = self.model.predict_proba(
            X_scaled
        )

        return probabilities[:, 1]

    def predict(
        self,
        X,
        threshold=0.5,
    ):

        probabilities = self.predict_proba(X)

        return (
            probabilities >= threshold
        ).astype(int)

    # ============================================================
    # EVALUATION
    # ============================================================

    def evaluate(
        self,
        X,
        y,
        threshold=0.5,
    ):

        y_true = self._prepare_y(y)

        y_pred = self.predict(
            X,
            threshold=threshold,
        )

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        return {
            "accuracy": float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
            "precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "fpr": float(fpr),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }

    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================

    def feature_importance(self):

        if not self.is_fitted:
            raise RuntimeError(
                "Classifier has not been fitted."
            )

        coefficients = self.model.coef_[0]

        if self.feature_names is None:

            names = [
                f"feature_{i}"
                for i in range(len(coefficients))
            ]

        else:

            names = self.feature_names

            if len(names) != len(coefficients):

                names = [
                    f"feature_{i}"
                    for i in range(len(coefficients))
                ]

        pairs = list(
            zip(
                names,
                coefficients,
            )
        )

        pairs.sort(
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        return [
            {
                "feature": name,
                "coefficient": float(value),
                "importance": float(abs(value)),
            }
            for name, value in pairs
        ]

    # ============================================================
    # SAVE / LOAD
    # ============================================================

    def save(self, path):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "is_fitted": self.is_fitted,
            },
            path,
        )

    def load(self, path):

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Classifier not found: {path}"
            )

        checkpoint = joblib.load(path)

        if isinstance(checkpoint, dict):

            self.model = checkpoint["model"]

            self.scaler = checkpoint.get(
                "scaler",
                StandardScaler(),
            )

            self.feature_names = checkpoint.get(
                "feature_names"
            )

            self.is_fitted = checkpoint.get(
                "is_fitted",
                True,
            )

        else:

            self.model = checkpoint
            self.is_fitted = True

        return self

    @classmethod
    def from_file(cls, path):

        instance = cls()

        instance.load(path)

        return instance