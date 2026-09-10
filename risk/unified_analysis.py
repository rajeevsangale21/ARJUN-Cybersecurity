import numpy as np

from risk.attack_progression import (
    AttackProgressionAnalyzer,
)

from forecasting.mitre_mapper import (
    MitreStageMapper,
)

from risk.unified_risk import (
    UnifiedRiskEngine,
)

from risk.unified_contributions import (
    UnifiedFeatureContributionAnalyzer,
)


class UnifiedAnalysisService:
    """
    Combines all ARJUN intelligence outputs:

        Attack progression
        MITRE ATT&CK stage
        Future attack probability
        Unified risk
        Feature contributions
    """

    def __init__(self, feature_names):

        self.feature_names = list(
            feature_names
        )

        if not self.feature_names:
            raise ValueError(
                "feature_names cannot be empty."
            )

        self.progression_analyzer = (
            AttackProgressionAnalyzer()
        )

        self.mitre_mapper = (
            MitreStageMapper()
        )

        self.risk_engine = (
            UnifiedRiskEngine()
        )

        self.contribution_analyzer = (
            UnifiedFeatureContributionAnalyzer(
                self.feature_names
            )
        )

    # =========================================================
    # Validation
    # =========================================================

    def _validate_state(
        self,
        state,
        name="state",
    ):

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        if state.ndim != 1:
            raise ValueError(
                f"{name} must be a 1D state vector."
            )

        if len(state) != len(
            self.feature_names
        ):
            raise ValueError(
                f"{name} dimension "
                f"{len(state)} does not match "
                f"feature dimension "
                f"{len(self.feature_names)}."
            )

        return np.nan_to_num(
            state,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    # =========================================================
    # Main analysis
    # =========================================================

    def analyze(
        self,
        current_state,
        future_states,
        current_attack_probability,
        future_attack_probabilities=None,
    ):
        """
        Perform complete ARJUN forecast analysis.

        Parameters
        ----------
        current_state:
            Current network state.

        future_states:
            K predicted future network states.

        current_attack_probability:
            Current attack probability from the
            attack classifier.

        future_attack_probabilities:
            Optional classifier probabilities evaluated
            against predicted future states.
        """

        current_state = self._validate_state(
            current_state,
            "current_state",
        )

        future_states = np.asarray(
            future_states,
            dtype=np.float32,
        )

        if future_states.ndim == 1:
            future_states = (
                future_states.reshape(
                    1,
                    -1,
                )
            )

        if future_states.ndim != 2:
            raise ValueError(
                "future_states must be a 2D array."
            )

        if future_states.shape[1] != len(
            self.feature_names
        ):
            raise ValueError(
                "Future state dimension does not "
                "match feature names."
            )

        future_states = np.nan_to_num(
            future_states,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        current_attack_probability = float(
            np.clip(
                current_attack_probability,
                0.0,
                1.0,
            )
        )

        # =====================================================
        # Future probability validation
        # =====================================================

        if future_attack_probabilities is not None:

            future_attack_probabilities = np.asarray(
                future_attack_probabilities,
                dtype=np.float32,
            ).reshape(-1)

            if len(
                future_attack_probabilities
            ) != len(future_states):

                raise ValueError(
                    "future_attack_probabilities must "
                    "contain exactly one probability "
                    "for each future state."
                )

            future_attack_probabilities = np.clip(
                future_attack_probabilities,
                0.0,
                1.0,
            )

        # =====================================================
        # 1. ATTACK PROGRESSION
        # =====================================================

        progression_scores = (
            self.progression_analyzer.analyze(
                future_states,
                self.feature_names,
            )
        )

        progression_scores = np.asarray(
            progression_scores,
            dtype=np.float32,
        )

        # =====================================================
        # 2. MITRE ATT&CK STAGES
        # =====================================================

        mitre_results = (
            self.mitre_mapper.map_sequence(
                future_states,
                self.feature_names,
            )
        )

        # =====================================================
        # 3. UNIFIED RISK TIMELINE
        # =====================================================

        timeline = (
            self.risk_engine.build_timeline(
                current_attack_probability=(
                    current_attack_probability
                ),
                forecast_risks=progression_scores,
                mitre_results=mitre_results,
                future_attack_probabilities=(
                    future_attack_probabilities
                ),
            )
        )

        # =====================================================
        # 4. FEATURE CONTRIBUTIONS
        # =====================================================

        contributions = (
            self.contribution_analyzer.calculate_timeline(
                current_state,
                future_states,
                top_k=5,
            )
        )

        # =====================================================
        # 5. HIGHEST RISK
        # =====================================================

        if timeline:

            highest_risk = max(
                timeline,
                key=lambda item:
                float(
                    item.get(
                        "risk_score",
                        0.0,
                    )
                ),
            )

        else:

            highest_risk = {
                "step": 0,
                "risk_score": 0.0,
                "risk_level": "LOW",
                "mitre_stage": "Unknown",
                "mitre_confidence": 0.0,
            }

        # =====================================================
        # 6. MAXIMUM FUTURE PROBABILITY
        # =====================================================

        if future_attack_probabilities is not None:

            maximum_future_probability = float(
                np.max(
                    future_attack_probabilities
                )
            )

        else:

            maximum_future_probability = None

        # =====================================================
        # 7. ATTACK STAGE TIMELINE
        # =====================================================

        stage_timeline = []

        for index, result in enumerate(
            mitre_results,
            start=1,
        ):

            stage_timeline.append(
                {
                    "step": index,
                    "stage": result.get(
                        "stage",
                        "Unknown",
                    ),
                    "confidence": float(
                        result.get(
                            "confidence",
                            0.0,
                        )
                    ),
                }
            )

        # =====================================================
        # 8. SUMMARY
        # =====================================================

        summary = {

            "current_attack_probability":
                current_attack_probability,

            "maximum_forecast_risk":
                float(
                    max(
                        (
                            item.get(
                                "risk_score",
                                0.0,
                            )
                            for item in timeline
                        ),
                        default=0.0,
                    )
                ),

            "maximum_future_attack_probability":
                maximum_future_probability,

            "maximum_risk_level":
                highest_risk.get(
                    "risk_level",
                    "LOW",
                ),

            "highest_risk_step":
                int(
                    highest_risk.get(
                        "step",
                        0,
                    )
                ),

            "highest_risk_mitre_stage":
                highest_risk.get(
                    "mitre_stage",
                    "Unknown",
                ),

            "highest_risk_mitre_confidence":
                float(
                    highest_risk.get(
                        "mitre_confidence",
                        0.0,
                    )
                ),

            "forecast_steps":
                int(
                    len(future_states)
                ),

            "state_dimension":
                int(
                    future_states.shape[1]
                ),
        }

        # =====================================================
        # RESULT
        # =====================================================

        return {

            "timeline":
                timeline,

            "progression_scores":
                progression_scores.tolist(),

            "mitre_results":
                mitre_results,

            "stage_timeline":
                stage_timeline,

            "feature_contributions":
                contributions,

            "summary":
                summary,
        }