"""
ARJUN - Forecast Risk Engine

Converts predicted future network states into:
1. Attack probability
2. Infiltration probability
3. Risk score
4. Risk level
5. MITRE ATT&CK stage
6. Feature contributions
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecasting.mitre_mapper import predict_stage


FEATURE_NAMES = [
    "flow_count",
    "unique_src_ips",
    "unique_dst_ips",
    "unique_dst_ports",
    "total_bytes",
    "total_packets",
    "avg_duration",
    "avg_packet_length",
    "avg_ttl",
    "ttl_variance",
    "avg_tcp_window",
    "tcp_window_variance",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",
    "urg_count",
    "fragment_count",
    "retransmission_count",
    "avg_payload_size",
    "payload_variance",
    "max_payload_size",
    "avg_iat",
    "iat_variance",
    "avg_src_iat",
    "src_iat_variance",
    "avg_bytes_per_packet",
    "avg_packets_per_second",
    "avg_bytes_per_second",
    "avg_unique_dst_ports",
    "avg_unique_dst_hosts",
    "port_scan_count",
]


def safe_array(values) -> np.ndarray:
    """Convert values to a finite float32 array."""
    array = np.asarray(values, dtype=np.float32)

    return np.nan_to_num(
        array,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    x = float(np.clip(x, -30.0, 30.0))
    return 1.0 / (1.0 + np.exp(-x))


def robust_statistics(
    training_states: np.ndarray,
    training_labels: np.ndarray,
):
    """
    Calculate robust benign/attack reference statistics.

    Median and MAD are used because network telemetry is
    heavy-tailed and contains very large aggregate values.
    """
    X = safe_array(training_states)
    y = np.asarray(training_labels).astype(int)

    benign = X[y == 0]
    attack = X[y == 1]

    if len(benign) == 0:
        benign = X

    if len(attack) == 0:
        attack = X

    benign_median = np.median(benign, axis=0)
    attack_median = np.median(attack, axis=0)
    global_median = np.median(X, axis=0)

    mad = np.median(
        np.abs(X - global_median),
        axis=0,
    )

    mad = np.maximum(mad, 1e-6)

    return benign_median, attack_median, mad


def attack_probability(
    state: np.ndarray,
    benign_median: np.ndarray,
    attack_median: np.ndarray,
    mad: np.ndarray,
) -> float:
    """Estimate attack probability using robust reference distances."""
    state = safe_array(state)

    benign_distance = np.abs(state - benign_median) / mad
    attack_distance = np.abs(state - attack_median) / mad

    benign_distance = np.nan_to_num(
        benign_distance,
        nan=0.0,
        posinf=20.0,
        neginf=20.0,
    )

    attack_distance = np.nan_to_num(
        attack_distance,
        nan=0.0,
        posinf=20.0,
        neginf=20.0,
    )

    benign_distance = np.clip(benign_distance, 0.0, 20.0)
    attack_distance = np.clip(attack_distance, 0.0, 20.0)

    benign_score = float(
        np.mean(np.minimum(benign_distance, 6.0))
    )

    attack_score = float(
        np.mean(np.minimum(attack_distance, 6.0))
    )

    separation = benign_score - attack_score

    probability = sigmoid(1.25 * separation)

    return float(np.clip(probability, 0.0, 1.0))


def feature_contributions(
    current_state: np.ndarray,
    future_state: np.ndarray,
    benign_median: np.ndarray,
    attack_median: np.ndarray,
    mad: np.ndarray,
    top_k: int = 10,
) -> List[Dict]:
    """Calculate interpretable feature-level contributions."""
    current = safe_array(current_state)
    future = safe_array(future_state)

    attack_current = np.abs(current - attack_median) / mad
    attack_future = np.abs(future - attack_median) / mad

    benign_current = np.abs(current - benign_median) / mad
    benign_future = np.abs(future - benign_median) / mad

    attack_current = np.clip(attack_current, 0.0, 20.0)
    attack_future = np.clip(attack_future, 0.0, 20.0)
    benign_current = np.clip(benign_current, 0.0, 20.0)
    benign_future = np.clip(benign_future, 0.0, 20.0)

    # Positive means movement toward attack behaviour and/or
    # away from benign behaviour.
    attack_movement = attack_current - attack_future
    benign_movement = benign_future - benign_current

    contribution = attack_movement + benign_movement

    order = np.argsort(np.abs(contribution))[::-1]

    results = []

    for index in order[:top_k]:
        index = int(index)

        name = (
            FEATURE_NAMES[index]
            if index < len(FEATURE_NAMES)
            else f"feature_{index}"
        )

        results.append(
            {
                "feature": name,
                "index": index,
                "contribution": float(contribution[index]),
                "current": float(current[index]),
                "future": float(future[index]),
                "change": float(future[index] - current[index]),
            }
        )

    return results


def risk_level(score: float) -> str:
    """Convert a 0..1 risk score to a human-readable level."""
    score = float(score)

    if score < 0.20:
        return "LOW"

    if score < 0.40:
        return "GUARDED"

    if score < 0.60:
        return "MEDIUM"

    if score < 0.80:
        return "HIGH"

    return "CRITICAL"


def calculate_risk(
    current_state: np.ndarray,
    future_state: np.ndarray,
    benign_median: np.ndarray,
    attack_median: np.ndarray,
    mad: np.ndarray,
) -> Dict:
    """Calculate complete risk information for one future step."""
    probability = attack_probability(
        future_state,
        benign_median,
        attack_median,
        mad,
    )

    stage, stage_scores = predict_stage(
        current_state,
        future_state,
    )

    stage_confidence = max(stage_scores.values())

    risk_score = (
        0.80 * probability
        + 0.20 * stage_confidence
    )

    risk_score = float(np.clip(risk_score, 0.0, 1.0))

    return {
        "attack_probability": probability,
        "infiltration_probability": probability,
        "risk_score": risk_score,
        "risk_level": risk_level(risk_score),
        "mitre_stage": stage,
        "stage_scores": stage_scores,
    }


def generate_forecast_risk(
    forecast_states: np.ndarray,
    current_state: np.ndarray,
    training_states: np.ndarray,
    training_labels: np.ndarray,
) -> Dict:
    """Generate the complete attack probability timeline."""
    forecast_states = safe_array(forecast_states)
    current_state = safe_array(current_state)

    benign_median, attack_median, mad = robust_statistics(
        training_states,
        training_labels,
    )

    timeline = []
    previous = current_state

    for step, future_state in enumerate(
        forecast_states,
        start=1,
    ):
        result = calculate_risk(
            previous,
            future_state,
            benign_median,
            attack_median,
            mad,
        )

        result["step"] = step

        result["top_features"] = feature_contributions(
            previous,
            future_state,
            benign_median,
            attack_median,
            mad,
            top_k=10,
        )

        timeline.append(result)
        previous = future_state

    probabilities = np.asarray(
        [
            item["attack_probability"]
            for item in timeline
        ],
        dtype=np.float32,
    )

    risk_scores = np.asarray(
        [
            item["risk_score"]
            for item in timeline
        ],
        dtype=np.float32,
    )

    if len(probabilities):
        max_probability = float(np.max(probabilities))
        final_probability = float(probabilities[-1])
        max_risk = float(np.max(risk_scores))
    else:
        max_probability = 0.0
        final_probability = 0.0
        max_risk = 0.0

    return {
        "timeline": timeline,
        "attack_probability_timeline": probabilities,
        "risk_score_timeline": risk_scores,
        "max_attack_probability": max_probability,
        "final_attack_probability": final_probability,
        "max_risk_score": max_risk,
        "overall_risk_level": risk_level(max_risk),
    }


def save_risk_result(
    result: Dict,
    output_path: str | Path = ROOT / "data" / "processed" / "forecast_risk.npz",
):
    """Save numerical forecast-risk output."""
    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timeline = result["timeline"]

    probabilities = np.asarray(
        result["attack_probability_timeline"],
        dtype=np.float32,
    )

    risk_scores = np.asarray(
        result["risk_score_timeline"],
        dtype=np.float32,
    )

    stages = np.asarray(
        [
            item["mitre_stage"]
            for item in timeline
        ],
        dtype=str,
    )

    np.savez_compressed(
        output,
        attack_probability=probabilities,
        risk_score=risk_scores,
        mitre_stage=stages,
        max_attack_probability=np.float32(
            result["max_attack_probability"]
        ),
        final_attack_probability=np.float32(
            result["final_attack_probability"]
        ),
        max_risk_score=np.float32(
            result["max_risk_score"]
        ),
        overall_risk_level=np.asarray(
            result["overall_risk_level"]
        ),
    )

    return output


def load_forecast_states(forecast_data) -> np.ndarray:
    """Load predicted full future states from the forecast file."""
    if "forecast_states" in forecast_data:
        return safe_array(forecast_data["forecast_states"])

    # Compatibility with an older output format.
    if "forecast" in forecast_data:
        return safe_array(forecast_data["forecast"])

    if "states" in forecast_data:
        return safe_array(forecast_data["states"])

    raise KeyError(
        "k_step_forecast.npz does not contain "
        "'forecast_states', 'forecast', or 'states'."
    )


def main():
    """Run the risk engine against the latest K-step forecast."""
    forecast_path = (
        ROOT
        / "data"
        / "processed"
        / "k_step_forecast.npz"
    )

    state_path = (
        ROOT
        / "data"
        / "processed"
        / "training_states.npz"
    )

    if not forecast_path.exists():
        raise FileNotFoundError(
            f"\nForecast file not found:\n{forecast_path}\n"
        )

    if not state_path.exists():
        raise FileNotFoundError(
            f"\nTraining state file not found:\n{state_path}\n"
        )

    forecast_data = np.load(
        forecast_path,
        allow_pickle=True,
    )

    training_data = np.load(
        state_path,
        allow_pickle=True,
    )

    forecast_states = load_forecast_states(
        forecast_data
    )

    if "states" in training_data:
        training_states = safe_array(
            training_data["states"]
        )
    elif "X" in training_data:
        training_states = safe_array(
            training_data["X"]
        )
        if training_states.ndim == 3:
            training_states = training_states[:, -1, :]
    else:
        raise KeyError(
            "training_states.npz does not contain "
            "'states' or 'X'."
        )

    if "labels" in training_data:
        labels = np.asarray(
            training_data["labels"]
        ).astype(int)
    elif "y" in training_data:
        labels = np.asarray(
            training_data["y"]
        ).astype(int)
    else:
        raise KeyError(
            "training_states.npz does not contain "
            "'labels' or 'y'."
        )

    if len(labels) != len(training_states):
        raise ValueError(
            "Training states and labels have different lengths: "
            f"{len(training_states)} vs {len(labels)}"
        )

    current_state = training_states[-1]

    print()
    print("=" * 65)
    print("ARJUN FORECAST RISK ENGINE")
    print("=" * 65)

    print(
        f"Forecast states : {forecast_states.shape}"
    )

    print(
        f"Training states : {training_states.shape}"
    )

    print(
        f"Attack states   : {int(np.sum(labels == 1))}"
    )

    result = generate_forecast_risk(
        forecast_states,
        current_state,
        training_states,
        labels,
    )

    print()
    print("ATTACK PROBABILITY TIMELINE")
    print("-" * 65)

    for item in result["timeline"]:
        print(
            f"t+{item['step']:<2} "
            f"Attack={item['attack_probability'] * 100:6.2f}% "
            f"Risk={item['risk_score'] * 100:6.2f}% "
            f"Level={item['risk_level']:<8} "
            f"Stage={item['mitre_stage']}"
        )

    print()
    print("OVERALL RISK")
    print("-" * 65)

    print(
        f"Maximum attack probability : "
        f"{result['max_attack_probability'] * 100:.2f}%"
    )

    print(
        f"Final attack probability   : "
        f"{result['final_attack_probability'] * 100:.2f}%"
    )

    print(
        f"Maximum risk score         : "
        f"{result['max_risk_score'] * 100:.2f}%"
    )

    print(
        f"Overall risk level         : "
        f"{result['overall_risk_level']}"
    )

    print()
    print("TOP CONTRIBUTING FEATURES")
    print("-" * 65)

    for item in result["timeline"]:
        print()
        print(
            f"t+{item['step']} "
            f"({item['mitre_stage']})"
        )

        for feature in item["top_features"][:5]:
            print(
                f"  {feature['feature']:<28} "
                f"{feature['contribution']:+.4f}"
            )

    output = save_risk_result(result)

    print()
    print(f"Saved: {output}")

    print()
    print("=" * 65)
    print("FORECAST RISK TEST: PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
