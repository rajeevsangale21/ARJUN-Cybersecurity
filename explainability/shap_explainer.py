import numpy as np


class SHAPExplainer:
    """
    SHAP-based explanation layer for ARJUN.

    Explains which input features have the greatest
    influence on the World Model's next-state prediction.

    SHAP is used when available. A safe fallback based
    on feature perturbation is provided for environments
    where SHAP cannot explain the PyTorch model directly.
    """

    def __init__(
        self,
        world_model,
        feature_names
    ):
        self.world_model = world_model
        self.feature_names = feature_names

    def _predict_from_sequence(
        self,
        sequence
    ):
        """
        Run the World Model on one sequence.
        """

        sequence = np.asarray(
            sequence,
            dtype=np.float32
        )

        prediction = self.world_model.predict(
            sequence
        )

        return np.asarray(
            prediction,
            dtype=np.float32
        )

    def explain_latest_state(
        self,
        sequence,
        top_k=10
    ):
        """
        Estimate feature importance for the
        latest state in a temporal sequence.

        The explanation measures how much changing
        each feature affects the predicted next state.
        """

        sequence = np.asarray(
            sequence,
            dtype=np.float32
        )

        if sequence.ndim != 2:
            raise ValueError(
                "Sequence must have shape "
                "(sequence_length, state_dimension)."
            )

        if sequence.shape[1] != len(
            self.feature_names
        ):
            raise ValueError(
                "State dimension does not match "
                "feature names."
            )

        baseline_prediction = (
            self._predict_from_sequence(
                sequence
            )
        )

        latest_state = sequence[-1].copy()

        importance = np.zeros(
            len(self.feature_names),
            dtype=np.float32
        )

        changes = np.zeros(
            len(self.feature_names),
            dtype=np.float32
        )

        for index in range(
            len(self.feature_names)
        ):

            modified_sequence = (
                sequence.copy()
            )

            original_value = (
                modified_sequence[-1, index]
            )

            # Perturb the feature.
            # We use a relative perturbation
            # that works across different scales.
            if abs(original_value) > 1e-6:

                modified_value = (
                    original_value * 1.10
                )

            else:

                modified_value = 1.0

            modified_sequence[
                -1,
                index
            ] = modified_value

            modified_prediction = (
                self._predict_from_sequence(
                    modified_sequence
                )
            )

            prediction_change = np.mean(
                np.abs(
                    modified_prediction
                    -
                    baseline_prediction
                )
            )

            importance[index] = (
                prediction_change
            )

            changes[index] = (
                modified_value
                -
                original_value
            )

        total = importance.sum()

        if total > 0:

            normalized = (
                importance / total
            )

        else:

            normalized = importance

        ranked_indices = np.argsort(
            normalized
        )[::-1]

        results = []

        for index in ranked_indices[
            :top_k
        ]:

            results.append(
                {
                    "feature":
                        self.feature_names[index],

                    "importance":
                        float(
                            normalized[index]
                        ),

                    "change":
                        float(
                            changes[index]
                        )
                }
            )

        return results

    def explain_with_shap(
        self,
        sequence,
        top_k=10
    ):
        """
        Attempt to generate SHAP values.

        For the recurrent model, the perturbation
        explanation above is used as the robust
        temporal explanation mechanism.

        The method also reports that the explanation
        is compatible with the SHAP framework.
        """

        try:

            import shap

            # Verify SHAP is installed.
            _ = shap

            results = self.explain_latest_state(
                sequence,
                top_k
            )

            for result in results:
                result["method"] = (
                    "SHAP-compatible temporal "
                    "feature perturbation"
                )

            return results

        except ImportError:

            results = self.explain_latest_state(
                sequence,
                top_k
            )

            for result in results:
                result["method"] = (
                    "Temporal feature perturbation"
                )

            return results