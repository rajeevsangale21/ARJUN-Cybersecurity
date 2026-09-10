import numpy as np


class UnifiedRiskEngine:
    """
    Combines current attack probability, predicted future risk,
    future classifier probability and MITRE confidence.
    """

    def __init__(
        self,
        current_weight=0.20,
        forecast_weight=0.35,
        future_probability_weight=0.25,
        mitre_weight=0.20,
    ):

        total = (
            current_weight
            + forecast_weight
            + future_probability_weight
            + mitre_weight
        )

        if not np.isclose(total, 1.0):
            raise ValueError(
                "Risk weights must sum to 1.0."
            )

        self.current_weight = current_weight
        self.forecast_weight = forecast_weight
        self.future_probability_weight = (
            future_probability_weight
        )
        self.mitre_weight = mitre_weight

    @staticmethod
    def risk_level(score):

        score = float(
            np.clip(score, 0.0, 1.0)
        )

        if score < 0.20:
            return "LOW"

        if score < 0.40:
            return "MEDIUM"

        if score < 0.70:
            return "HIGH"

        return "CRITICAL"

    def combine(
        self,
        current_attack_probability,
        forecast_risk,
        mitre_confidence,
        future_attack_probability=None,
    ):
        """
        Calculate unified risk.

        future_attack_probability is optional for backward
        compatibility with older ARJUN code.
        """

        current_attack_probability = float(
            np.clip(
                current_attack_probability,
                0.0,
                1.0,
            )
        )

        forecast_risk = float(
            np.clip(
                forecast_risk,
                0.0,
                1.0,
            )
        )

        mitre_confidence = float(
            np.clip(
                mitre_confidence,
                0.0,
                1.0,
            )
        )

        if future_attack_probability is None:

            # Backward-compatible calculation.
            score = (
                0.35 * current_attack_probability
                + 0.45 * forecast_risk
                + 0.20 * mitre_confidence
            )

        else:

            future_attack_probability = float(
                np.clip(
                    future_attack_probability,
                    0.0,
                    1.0,
                )
            )

            score = (
                self.current_weight
                * current_attack_probability
                + self.forecast_weight
                * forecast_risk
                + self.future_probability_weight
                * future_attack_probability
                + self.mitre_weight
                * mitre_confidence
            )

        score = float(
            np.clip(score, 0.0, 1.0)
        )

        return {
            "risk_score": score,
            "risk_level": self.risk_level(score),
        }

    def build_timeline(
        self,
        current_attack_probability,
        forecast_risks,
        mitre_results,
        future_attack_probabilities=None,
    ):
        """
        Build the complete future risk timeline.
        """

        forecast_risks = np.asarray(
            forecast_risks,
            dtype=np.float32,
        )

        if future_attack_probabilities is not None:

            future_attack_probabilities = np.asarray(
                future_attack_probabilities,
                dtype=np.float32,
            )

            if len(future_attack_probabilities) != len(
                forecast_risks
            ):
                raise ValueError(
                    "Forecast risk and future probability "
                    "lengths must match."
                )

        timeline = []

        for index, forecast_risk in enumerate(
            forecast_risks
        ):

            mitre = mitre_results[index]

            mitre_stage = mitre.get(
                "stage",
                "Unknown",
            )

            mitre_confidence = float(
                mitre.get(
                    "confidence",
                    0.0,
                )
            )

            if future_attack_probabilities is not None:

                future_probability = float(
                    future_attack_probabilities[
                        index
                    ]
                )

            else:

                future_probability = None

            result = self.combine(
                current_attack_probability=(
                    current_attack_probability
                ),
                forecast_risk=float(
                    forecast_risk
                ),
                mitre_confidence=mitre_confidence,
                future_attack_probability=(
                    future_probability
                ),
            )

            timeline.append(
                {
                    "step": index + 1,

                    "current_attack_probability":
                        float(
                            current_attack_probability
                        ),

                    "future_attack_probability":
                        future_probability,

                    "forecast_risk":
                        float(forecast_risk),

                    "mitre_stage":
                        mitre_stage,

                    "mitre_confidence":
                        mitre_confidence,

                    "risk_score":
                        result["risk_score"],

                    "risk_level":
                        result["risk_level"],
                }
            )

        return timeline