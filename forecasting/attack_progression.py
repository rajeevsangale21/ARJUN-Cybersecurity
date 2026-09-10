import numpy as np


class AttackProgressionAnalyzer:
    """
    Analyzes the predicted evolution of network states.

    This provides behavioral indicators that can
    later be mapped to MITRE ATT&CK stages.
    """

    def __init__(self, feature_names):

        self.feature_names = feature_names

        self.index = {
            name: index
            for index, name in enumerate(
                feature_names
            )
        }

    def get_value(self, state, feature):
        index = self.index.get(feature)

        if index is None:
            return 0.0

        return float(state[index])

    def analyze_state(
        self,
        state,
        previous_state=None
    ):
        """
        Analyze one predicted state.
        """

        flow_count = self.get_value(
            state,
            "flow_count"
        )

        unique_dst_ports = self.get_value(
            state,
            "unique_dst_ports"
        )

        unique_dst_hosts = self.get_value(
            state,
            "unique_dst_hosts"
        )

        syn_count = self.get_value(
            state,
            "syn_count"
        )

        rst_count = self.get_value(
            state,
            "rst_count"
        )

        total_bytes = self.get_value(
            state,
            "total_bytes"
        )

        indicators = {}

        # -------------------------
        # Port scanning indicator
        # -------------------------

        indicators["port_scan"] = (
            unique_dst_ports >= 20
            or unique_dst_hosts >= 10
        )

        # -------------------------
        # SYN scanning indicator
        # -------------------------

        indicators["syn_activity"] = (
            syn_count >= 20
        )

        # -------------------------
        # High connection activity
        # -------------------------

        indicators["high_connection_volume"] = (
            flow_count >= 100
        )

        # -------------------------
        # RST activity
        # -------------------------

        indicators["reset_activity"] = (
            rst_count >= 20
        )

        # -------------------------
        # Traffic growth
        # -------------------------

        traffic_growth = 0.0

        if previous_state is not None:

            previous_bytes = self.get_value(
                previous_state,
                "total_bytes"
            )

            if previous_bytes > 0:

                traffic_growth = (
                    total_bytes -
                    previous_bytes
                ) / previous_bytes

        indicators["traffic_growth"] = (
            traffic_growth
        )

        # -------------------------
        # Overall activity score
        # -------------------------

        score = 0.0

        if indicators["port_scan"]:
            score += 0.35

        if indicators["syn_activity"]:
            score += 0.20

        if indicators["high_connection_volume"]:
            score += 0.20

        if indicators["reset_activity"]:
            score += 0.10

        if traffic_growth > 1.0:
            score += 0.15

        score = min(
            max(score, 0.0),
            1.0
        )

        indicators["progression_score"] = score

        return indicators

    def analyze_forecast(
        self,
        states
    ):
        """
        Analyze the entire predicted timeline.
        """

        states = np.asarray(
            states,
            dtype=np.float32
        )

        results = []

        previous_state = None

        for step, state in enumerate(states, 1):

            analysis = self.analyze_state(
                state,
                previous_state
            )

            analysis["step"] = step

            results.append(
                analysis
            )

            previous_state = state

        return results