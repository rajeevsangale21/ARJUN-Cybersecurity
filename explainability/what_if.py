import numpy as np


class WhatIfAnalyzer:
    """
    Performs counterfactual what-if analysis
    on the current network state.

    Example:

        Reduce SYN activity by 50%

    Then observe how the predicted state changes.
    """

    def __init__(
        self,
        world_model,
        feature_names
    ):
        self.world_model = world_model
        self.feature_names = feature_names

        self.feature_index = {
            name: index
            for index, name in enumerate(
                feature_names
            )
        }

    def modify_feature(
        self,
        state,
        feature_name,
        factor
    ):
        """
        Modify one feature by a multiplicative factor.

        factor = 0.5
            reduces feature by 50%

        factor = 1.5
            increases feature by 50%
        """

        state = np.asarray(
            state,
            dtype=np.float32
        ).copy()

        if feature_name not in (
            self.feature_index
        ):
            raise ValueError(
                f"Unknown feature: "
                f"{feature_name}"
            )

        index = self.feature_index[
            feature_name
        ]

        state[index] *= factor

        return state

    def analyze(
        self,
        sequence,
        feature_name,
        factor
    ):
        """
        Compare original and counterfactual
        World Model predictions.
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

        baseline_prediction = (
            self.world_model.predict(
                sequence
            )
        )

        modified_sequence = (
            sequence.copy()
        )

        modified_sequence[-1] = (
            self.modify_feature(
                modified_sequence[-1],
                feature_name,
                factor
            )
        )

        counterfactual_prediction = (
            self.world_model.predict(
                modified_sequence
            )
        )

        prediction_change = (
            counterfactual_prediction
            -
            baseline_prediction
        )

        return {
            "feature":
                feature_name,

            "factor":
                float(factor),

            "baseline_prediction":
                baseline_prediction,

            "counterfactual_prediction":
                counterfactual_prediction,

            "prediction_change":
                prediction_change
        }