"""
ARJUN - MITRE ATT&CK Stage Mapper

Maps observed/predicted network behaviour to high-level
MITRE ATT&CK attack progression stages.

This is intentionally a transparent rule-based layer.
The World Model remains responsible for predicting future
network states; this module interprets those states.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


MITRE_STAGES = [
    "Reconnaissance",
    "Initial Access",
    "Lateral Movement",
    "Command & Control",
    "Exfiltration",
]


def _safe_value(state: np.ndarray, index: int) -> float:
    """Safely read a feature from a state vector."""
    if state is None:
        return 0.0

    if index >= len(state):
        return 0.0

    value = float(state[index])

    if not np.isfinite(value):
        return 0.0

    return value


def _relative_change(current: np.ndarray, future: np.ndarray, index: int) -> float:
    """Calculate a stable relative feature change."""
    a = abs(_safe_value(current, index))
    b = abs(_safe_value(future, index))

    denominator = max(a, 1.0)

    return (b - a) / denominator


def score_stage(
    current_state: np.ndarray,
    future_state: np.ndarray,
) -> Dict[str, float]:
    """
    Score each high-level attack progression stage.

    Feature indices correspond to ARJUN's 33-feature state representation.
    """

    # Feature indices
    unique_dst_ports = 3
    total_bytes = 4
    total_packets = 5
    avg_duration = 6
    syn_count = 12
    ack_count = 13
    rst_count = 15
    retransmissions = 19
    avg_payload = 20
    avg_iat = 23
    avg_packets_per_second = 28
    avg_bytes_per_second = 29
    avg_unique_dst_ports = 30
    avg_unique_dst_hosts = 31
    port_scan_count = 32

    scores = {
        "Reconnaissance": 0.0,
        "Initial Access": 0.0,
        "Lateral Movement": 0.0,
        "Command & Control": 0.0,
        "Exfiltration": 0.0,
    }

    # ---------------------------------------------------------
    # Reconnaissance
    # ---------------------------------------------------------

    scores["Reconnaissance"] += min(
        abs(_relative_change(current_state, future_state, port_scan_count)),
        1.0,
    ) * 0.40

    scores["Reconnaissance"] += min(
        abs(_relative_change(current_state, future_state, unique_dst_ports)),
        1.0,
    ) * 0.20

    scores["Reconnaissance"] += min(
        abs(_relative_change(current_state, future_state, avg_unique_dst_ports)),
        1.0,
    ) * 0.20

    scores["Reconnaissance"] += min(
        abs(_relative_change(current_state, future_state, avg_unique_dst_hosts)),
        1.0,
    ) * 0.20

    # ---------------------------------------------------------
    # Initial Access
    # ---------------------------------------------------------

    syn_change = _relative_change(current_state, future_state, syn_count)
    rst_change = _relative_change(current_state, future_state, rst_count)

    scores["Initial Access"] += min(abs(syn_change), 1.0) * 0.35
    scores["Initial Access"] += min(abs(rst_change), 1.0) * 0.15

    if _safe_value(future_state, syn_count) > 0:
        scores["Initial Access"] += 0.25

    if _safe_value(future_state, ack_count) > 0:
        scores["Initial Access"] += 0.25

    # ---------------------------------------------------------
    # Lateral Movement
    # ---------------------------------------------------------

    host_change = _relative_change(
        current_state,
        future_state,
        avg_unique_dst_hosts,
    )

    port_change = _relative_change(
        current_state,
        future_state,
        avg_unique_dst_ports,
    )

    packet_change = _relative_change(
        current_state,
        future_state,
        total_packets,
    )

    scores["Lateral Movement"] += min(abs(host_change), 1.0) * 0.40
    scores["Lateral Movement"] += min(abs(port_change), 1.0) * 0.25
    scores["Lateral Movement"] += min(abs(packet_change), 1.0) * 0.15

    if _safe_value(future_state, avg_unique_dst_hosts) > 1:
        scores["Lateral Movement"] += 0.20

    # ---------------------------------------------------------
    # Command & Control
    # ---------------------------------------------------------

    iat = abs(_safe_value(future_state, avg_iat))
    duration = abs(_safe_value(future_state, avg_duration))
    retrans = abs(_safe_value(future_state, retransmissions))

    # Periodic / persistent communication often manifests
    # through timing and long-lived flows.
    if iat > 0:
        scores["Command & Control"] += 0.20

    if duration > 1:
        scores["Command & Control"] += 0.25

    if retrans > 0:
        scores["Command & Control"] += 0.20

    scores["Command & Control"] += min(
        abs(_relative_change(
            current_state,
            future_state,
            avg_iat,
        )),
        1.0,
    ) * 0.15

    scores["Command & Control"] += min(
        abs(_relative_change(
            current_state,
            future_state,
            avg_packets_per_second,
        )),
        1.0,
    ) * 0.10

    scores["Command & Control"] += min(
        abs(_relative_change(
            current_state,
            future_state,
            avg_payload,
        )),
        1.0,
    ) * 0.10

    # ---------------------------------------------------------
    # Exfiltration
    # ---------------------------------------------------------

    byte_rate_change = _relative_change(
        current_state,
        future_state,
        avg_bytes_per_second,
    )

    total_byte_change = _relative_change(
        current_state,
        future_state,
        total_bytes,
    )

    payload_change = _relative_change(
        current_state,
        future_state,
        avg_payload,
    )

    scores["Exfiltration"] += min(abs(byte_rate_change), 1.0) * 0.40
    scores["Exfiltration"] += min(abs(total_byte_change), 1.0) * 0.30
    scores["Exfiltration"] += min(abs(payload_change), 1.0) * 0.20

    if _safe_value(future_state, avg_bytes_per_second) > 0:
        scores["Exfiltration"] += 0.10

    # Normalize to 0..1
    for stage in scores:
        scores[stage] = float(np.clip(scores[stage], 0.0, 1.0))

    return scores


def predict_stage(
    current_state: np.ndarray,
    future_state: np.ndarray,
) -> Tuple[str, Dict[str, float]]:
    """Return the highest-scoring MITRE stage."""
    scores = score_stage(current_state, future_state)

    stage = max(scores, key=scores.get)

    return stage, scores


def progression(
    states: np.ndarray,
) -> List[Dict]:
    """
    Produce MITRE progression for a sequence of predicted states.
    """

    states = np.asarray(states, dtype=np.float32)

    results = []

    if len(states) == 0:
        return results

    current = states[0]

    for step in range(1, len(states)):
        future = states[step]

        stage, scores = predict_stage(
            current,
            future,
        )

        results.append(
            {
                "step": step,
                "stage": stage,
                "stage_scores": scores,
            }
        )

        current = future

    return results


def stage_to_tactic(stage: str) -> str:
    """Map high-level stage to ATT&CK tactic terminology."""

    mapping = {
        "Reconnaissance": "Reconnaissance",
        "Initial Access": "Initial Access",
        "Lateral Movement": "Lateral Movement",
        "Command & Control": "Command and Control",
        "Exfiltration": "Exfiltration",
    }

    return mapping.get(stage, "Unknown")


class MitreStageMapper:
    """
    Object-oriented wrapper around MITRE ATT&CK stage mapping.
    Provides map_sequence expected by UnifiedAnalysisService.
    """

    def __init__(self):
        pass

    def map_sequence(
        self,
        future_states: np.ndarray,
        feature_names=None,
    ) -> List[Dict]:
        future_states = np.asarray(future_states, dtype=np.float32)
        results = []
        if len(future_states) == 0:
            return results

        prev = np.zeros_like(future_states[0])
        for idx, current in enumerate(future_states):
            scores = score_stage(prev, current)
            stage = max(scores, key=scores.get)
            max_score = float(scores.get(stage, 0.0))
            confidence = float(np.clip(max_score, 0.50, 0.99)) if max_score > 0 else 0.75
            results.append({
                "step": idx + 1,
                "stage": stage,
                "tactic": stage_to_tactic(stage),
                "confidence": confidence,
                "scores": scores,
            })
            prev = current

        return results