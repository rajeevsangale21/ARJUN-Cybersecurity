import numpy as np


class UnifiedFeatureContributionAnalyzer:

    def __init__(self, feature_names):

        self.feature_names = list(
            feature_names
        )

    def calculate(
        self,
        current_state,
        predicted_state,
        top_k=5
    ):

        current_state = np.asarray(
            current_state,
            dtype=np.float32
        )

        predicted_state = np.asarray(
            predicted_state,
            dtype=np.float32
        )

        if current_state.shape != predicted_state.shape:

            raise ValueError(
                "Current and predicted states "
                "must have the same shape."
            )

        if len(current_state) != len(
            self.feature_names
        ):

            raise ValueError(
                "State dimension does not match "
                "feature names."
            )

        changes = np.abs(
            predicted_state
            - current_state
        )

        # Relative change with numerical stability
        relative_changes = (
            changes
            / (
                np.abs(current_state)
                + 1.0
            )
        )

        ranking = []

        for feature, change in zip(
            self.feature_names,
            relative_changes
        ):

            ranking.append(
                {
                    "feature": feature,
                    "contribution": float(
                        change
                    )
                }
            )

        ranking.sort(
            key=lambda x: x["contribution"],
            reverse=True
        )

        return ranking[:top_k]

    def calculate_timeline(
        self,
        current_state,
        predicted_states,
        top_k=5
    ):

        results = []

        reference_state = np.asarray(
            current_state,
            dtype=np.float32
        )

        for step, state in enumerate(
            predicted_states,
            start=1
        ):

            contributions = self.calculate(
                reference_state,
                state,
                top_k
            )

            results.append(
                {
                    "step": step,
                    "features": contributions
                }
            )

        return results