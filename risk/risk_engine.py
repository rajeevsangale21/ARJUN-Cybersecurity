import numpy as np

from risk.risk_levels import get_risk_level


class RiskEngine:
    """
    Calculates future cyber-risk from predicted
    network-state trajectories.

    The risk score represents the likelihood that
    the predicted trajectory contains increasingly
    suspicious behavior.

    Score range:
        0.0 -> 1.0
    """

    def __init__(self, feature_names):

        self.feature_names = feature_names

        self.feature_index = {
            name: index
            for index, name in enumerate(
                feature_names
            )
        }

    def get_feature(
        self,
        state,
        feature_name
    ):
        """
        Safely retrieve a feature from a state.
        """

        index = self.feature_index.get(
            feature_name
        )

        if index is None:
            return 0.0

        return float(state[index])

    def calculate_state_risk(
        self,
        state,
        progression,
        stage_prediction
    ):
        """
        Calculate risk for one predicted state.
        """

        score = 0.0

        # --------------------------------
        # Behavioral progression
        # --------------------------------

        progression_score = float(
            progression.get(
                "progression_score",
                0.0
            )
        )

        score += (
            progression_score * 0.40
        )

        # --------------------------------
        # MITRE stage confidence
        # --------------------------------

        stage_confidence = float(
            stage_prediction.get(
                "confidence",
                0.0
            )
        )

        stage_name = stage_prediction.get(
            "stage",
            ""
        )

        if stage_name != "No strong attack stage":

            score += (
                stage_confidence * 0.20
            )

        # --------------------------------
        # Port scanning
        # --------------------------------

        if progression.get(
            "port_scan",
            False
        ):
            score += 0.15

        # --------------------------------
        # SYN activity
        # --------------------------------

        if progression.get(
            "syn_activity",
            False
        ):
            score += 0.10

        # --------------------------------
        # High connection volume
        # --------------------------------

        if progression.get(
            "high_connection_volume",
            False
        ):
            score += 0.10

        # --------------------------------
        # Traffic growth
        # --------------------------------

        traffic_growth = float(
            progression.get(
                "traffic_growth",
                0.0
            )
        )

        if traffic_growth > 0:

            growth_score = min(
                traffic_growth / 5.0,
                1.0
            )

            score += (
                growth_score * 0.05
            )

        score = max(
            0.0,
            min(1.0, score)
        )

        return score

    def calculate_forecast_risk(
        self,
        predicted_states,
        progression_results,
        mitre_predictions
    ):
        """
        Calculate risk for every future step.
        """

        if not (
            len(predicted_states)
            ==
            len(progression_results)
            ==
            len(mitre_predictions)
        ):
            raise ValueError(
                "Predicted states, progression "
                "results and MITRE predictions "
                "must have the same length."
            )

        timeline = []

        for index in range(
            len(predicted_states)
        ):

            state = predicted_states[
                index
            ]

            progression = (
                progression_results[index]
            )

            stage_prediction = (
                mitre_predictions[index]
            )

            risk_score = (
                self.calculate_state_risk(
                    state,
                    progression,
                    stage_prediction
                )
            )

            risk_level = get_risk_level(
                risk_score
            )

            timeline.append(
                {
                    "step":
                        index + 1,

                    "risk_score":
                        risk_score,

                    "risk_level":
                        risk_level,

                    "mitre_stage":
                        stage_prediction.get(
                            "stage",
                            "Unknown"
                        ),

                    "stage_confidence":
                        float(
                            stage_prediction.get(
                                "confidence",
                                0.0
                            )
                        )
                }
            )

        return timeline

    def calculate_infiltration_probability(
        self,
        risk_timeline
    ):
        """
        Convert forecast risk into an
        infiltration probability timeline.

        This is deliberately smoothed so that
        one anomalous prediction does not cause
        an unrealistic probability jump.
        """

        probabilities = []

        previous_probability = 0.0

        for item in risk_timeline:

            risk = float(
                item["risk_score"]
            )

            # Smooth transition
            probability = (
                0.7 * risk
                +
                0.3 * previous_probability
            )

            probability = max(
                0.0,
                min(1.0, probability)
            )

            probabilities.append(
                probability
            )

            previous_probability = (
                probability
            )

        return probabilities

    def build_forecast_report(
        self,
        predicted_states,
        progression_results,
        mitre_predictions
    ):
        """
        Create the complete Phase 5
        forecast report.
        """

        risk_timeline = (
            self.calculate_forecast_risk(
                predicted_states,
                progression_results,
                mitre_predictions
            )
        )

        probabilities = (
            self.calculate_infiltration_probability(
                risk_timeline
            )
        )

        for index, probability in enumerate(
            probabilities
        ):
            risk_timeline[index][
                "infiltration_probability"
            ] = probability

        return risk_timeline