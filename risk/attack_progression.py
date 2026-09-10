import numpy as np


class AttackProgressionAnalyzer:

    def __init__(self):

        self.feature_weights = {

            "port_scan_count": 0.30,

            "syn_count": 0.15,

            "unique_dst_ports": 0.15,

            "unique_dst_ips": 0.10,

            "total_bytes": 0.10,

            "total_packets": 0.10,

            "retransmission_count": 0.05,

            "rst_count": 0.05
        }

    def calculate_score(
        self,
        state,
        feature_names
    ):
        """
        Calculate an attack progression score
        from a network state.
        """

        state = np.asarray(
            state,
            dtype=np.float32
        )

        if len(state) != len(feature_names):

            raise ValueError(
                "State dimension does not match "
                "feature names."
            )

        values = dict(
            zip(
                feature_names,
                state
            )
        )

        score = 0.0

        for feature, weight in (
            self.feature_weights.items()
        ):

            if feature not in values:
                continue

            value = float(
                values[feature]
            )

            # Bounded contribution
            normalized = (
                abs(value)
                / (1.0 + abs(value))
            )

            score += (
                normalized
                * weight
            )

        return float(
            np.clip(
                score,
                0.0,
                1.0
            )
        )

    def analyze(
        self,
        states,
        feature_names
    ):

        states = np.asarray(
            states,
            dtype=np.float32
        )

        scores = []

        for state in states:

            scores.append(
                self.calculate_score(
                    state,
                    feature_names
                )
            )

        return np.asarray(
            scores,
            dtype=np.float32
        )

    @staticmethod
    def probability_from_score(
        score
    ):
        """
        Convert progression score into a bounded
        risk-oriented probability.

        This is a heuristic risk probability,
        not a statistically calibrated probability.
        """

        return float(
            np.clip(
                score,
                0.0,
                1.0
            )
        )