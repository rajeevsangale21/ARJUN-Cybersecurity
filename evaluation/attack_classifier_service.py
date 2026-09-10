import numpy as np


class AttackClassifierService:
    """
    Service wrapper around the ARJUN attack classifier.

    Provides a simple interface for:
        - attack probability
        - binary prediction
        - feature importance
    """

    def __init__(self, classifier):
        if classifier is None:
            raise ValueError(
                "A trained classifier is required."
            )

        self.classifier = classifier

    def predict_probability(self, state):
        """
        Return P(attack | state).

        Parameters
        ----------
        state : array-like
            One network state vector.

        Returns
        -------
        float
            Attack probability between 0 and 1.
        """

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        if state.ndim == 1:
            state = state.reshape(1, -1)

        if state.ndim != 2:
            raise ValueError(
                "State must be a 1D or 2D array."
            )

        probabilities = (
            self.classifier.predict_proba(state)
        )

        probabilities = np.asarray(
            probabilities,
            dtype=np.float32,
        ).reshape(-1)

        if len(probabilities) == 0:
            return 0.0

        return float(
            np.clip(
                probabilities[0],
                0.0,
                1.0,
            )
        )

    def predict(self, state, threshold=0.5):
        """
        Return binary attack prediction.
        """

        probability = (
            self.predict_probability(state)
        )

        return int(
            probability >= threshold
        )

    def feature_importance(self):
        """
        Return classifier feature importance.
        """

        if not hasattr(
            self.classifier,
            "feature_importance",
        ):
            return []

        return self.classifier.feature_importance()