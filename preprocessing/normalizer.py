from pathlib import Path

import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler


class NetworkStateNormalizer:
    """
    Normalizes ARJUN network-state vectors.

    IMPORTANT:
    The scaler must be fitted ONLY on training data.

    After fitting, the same scaler is reused for:
        - validation data
        - test data
        - forecasting/inference data
    """

    def __init__(self):

        self.scaler = StandardScaler()

        self.is_fitted = False


    # ==================================================
    # FIT
    # ==================================================

    def fit(self, states):
        """
        Fit the normalizer using training states only.

        Parameters
        ----------
        states : numpy.ndarray
            Training network states.

        Returns
        -------
        self
        """

        states = self._validate_states(
            states
        )

        self.scaler.fit(
            states
        )

        self.is_fitted = True

        return self


    # ==================================================
    # TRANSFORM
    # ==================================================

    def transform(self, states):
        """
        Normalize network states using the
        already-fitted training scaler.
        """

        if not self.is_fitted:

            raise RuntimeError(
                "Normalizer has not been fitted. "
                "Call fit() using training data first."
            )

        states = self._validate_states(
            states
        )

        normalized = (
            self.scaler.transform(
                states
            )
        )

        normalized = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return normalized.astype(
            np.float32
        )


    # ==================================================
    # FIT + TRANSFORM
    # ==================================================

    def fit_transform(self, states):
        """
        Fit the scaler on training states and
        immediately transform them.

        Use this ONLY for training data.
        """

        states = self._validate_states(
            states
        )

        normalized = (
            self.scaler.fit_transform(
                states
            )
        )

        self.is_fitted = True

        normalized = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return normalized.astype(
            np.float32
        )


    # ==================================================
    # INVERSE TRANSFORM
    # ==================================================

    def inverse_transform(self, states):
        """
        Convert normalized states back into
        the original feature scale.
        """

        if not self.is_fitted:

            raise RuntimeError(
                "Normalizer has not been fitted."
            )

        states = self._validate_states(
            states
        )

        original = (
            self.scaler.inverse_transform(
                states
            )
        )

        original = np.nan_to_num(
            original,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return original.astype(
            np.float32
        )


    # ==================================================
    # SAVE
    # ==================================================

    def save(self, path):
        """
        Save the fitted scaler.
        """

        if not self.is_fitted:

            raise RuntimeError(
                "Cannot save an unfitted normalizer."
            )

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            self.scaler,
            path
        )


    # ==================================================
    # LOAD
    # ==================================================

    def load(self, path):
        """
        Load a previously fitted scaler.
        """

        path = Path(
            path
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Normalizer not found: {path}"
            )

        self.scaler = joblib.load(
            path
        )

        self.is_fitted = True

        return self


    # ==================================================
    # VALIDATION
    # ==================================================

    @staticmethod
    def _validate_states(states):
        """
        Validate state matrix shape and values.
        """

        states = np.asarray(
            states,
            dtype=np.float32
        )

        if states.ndim != 2:

            raise ValueError(
                "Network states must be a "
                "2-dimensional array."
            )

        if states.shape[0] == 0:

            raise ValueError(
                "Network state array is empty."
            )

        states = np.nan_to_num(
            states,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return states