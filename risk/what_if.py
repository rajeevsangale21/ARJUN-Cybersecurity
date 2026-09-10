import numpy as np


class ModelBasedWhatIf:

    def __init__(
        self,
        forecaster,
        classifier,
        normalizer,
        feature_names,
    ):

        self.forecaster = forecaster
        self.classifier = classifier
        self.normalizer = normalizer
        self.feature_names = list(
            feature_names
        )

    def _predict_probabilities(
        self,
        predicted_states,
    ):

        probabilities = []

        for state in predicted_states:

            probability = (
                self.classifier.attack_probability(
                    state
                )
            )

            probabilities.append(
                float(probability)
            )

        return np.asarray(
            probabilities,
            dtype=np.float32,
        )

    def _forecast(
        self,
        normalized_sequence,
        graph_sequence,
        steps,
    ):

        predicted_states = (
            self.forecaster.forecast(
                normalized_sequence,
                graph_sequence,
                steps=steps,
            )
        )

        predicted_states = np.asarray(
            predicted_states,
            dtype=np.float32,
        )

        probabilities = (
            self._predict_probabilities(
                predicted_states
            )
        )

        return (
            predicted_states,
            probabilities,
        )

    def simulate(
        self,
        normalized_sequence,
        graph_sequence,
        feature_name,
        change_percent,
        steps=10,
    ):
        """
        Perform a genuine model-based what-if simulation.

        Example:

            feature_name = "syn_count"
            change_percent = -50

        This means:

            SYN count is reduced by 50%

        The modified state is then passed through the
        World Model again.
        """

        if feature_name not in self.feature_names:

            raise ValueError(
                f"Unknown feature: {feature_name}"
            )

        normalized_sequence = np.asarray(
            normalized_sequence,
            dtype=np.float32,
        ).copy()

        if normalized_sequence.ndim != 2:

            raise ValueError(
                "normalized_sequence must be 2D."
            )

        if len(normalized_sequence) == 0:

            raise ValueError(
                "normalized_sequence is empty."
            )

        feature_index = (
            self.feature_names.index(
                feature_name
            )
        )

        # --------------------------------------------------
        # Baseline forecast
        # --------------------------------------------------

        baseline_states, baseline_probs = (
            self._forecast(
                normalized_sequence,
                graph_sequence,
                steps,
            )
        )

        # --------------------------------------------------
        # Convert current state to original units
        # --------------------------------------------------

        current_normalized_state = (
            normalized_sequence[-1]
        )

        current_original_state = (
            self.normalizer.inverse_transform(
                current_normalized_state.reshape(
                    1, -1
                )
            )[0]
        )

        # --------------------------------------------------
        # Apply what-if modification
        # --------------------------------------------------

        modified_original_state = (
            current_original_state.copy()
        )

        original_value = float(
            modified_original_state[
                feature_index
            ]
        )

        change_fraction = (
            float(change_percent) / 100.0
        )

        modified_value = (
            original_value
            * (1.0 + change_fraction)
        )

        # Counts and quantities should not become
        # negative because of the simulation.

        if modified_value < 0:

            modified_value = 0.0

        modified_original_state[
            feature_index
        ] = modified_value

        # --------------------------------------------------
        # Convert modified state back to model scale
        # --------------------------------------------------

        modified_normalized_state = (
            self.normalizer.transform(
                modified_original_state.reshape(
                    1, -1
                )
            )[0]
        )

        counterfactual_sequence = (
            normalized_sequence.copy()
        )

        counterfactual_sequence[-1] = (
            modified_normalized_state
        )

        # --------------------------------------------------
        # Counterfactual forecast
        # --------------------------------------------------

        counterfactual_states, counterfactual_probs = (
            self._forecast(
                counterfactual_sequence,
                graph_sequence,
                steps,
            )
        )

        # --------------------------------------------------
        # Probability change
        # --------------------------------------------------

        probability_change = (
            counterfactual_probs
            - baseline_probs
        )

        probability_reduction = (
            baseline_probs
            - counterfactual_probs
        )

        return {

            "feature": feature_name,

            "change_percent":
                float(change_percent),

            "original_value":
                original_value,

            "modified_value":
                float(modified_value),

            "baseline_states":
                baseline_states,

            "counterfactual_states":
                counterfactual_states,

            "baseline_probabilities":
                baseline_probs,

            "counterfactual_probabilities":
                counterfactual_probs,

            "probability_change":
                probability_change,

            "probability_reduction":
                probability_reduction,

            "baseline_max_probability":
                float(
                    np.max(
                        baseline_probs
                    )
                ),

            "counterfactual_max_probability":
                float(
                    np.max(
                        counterfactual_probs
                    )
                ),

            "maximum_probability_reduction":
                float(
                    np.max(
                        probability_reduction
                    )
                ),
        }