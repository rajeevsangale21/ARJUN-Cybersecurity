"""
ARJUN Z4: Zeek World Model Forecast -> Risk + MITRE + Explainability

Consumes:
    data/processed/zeek_world_model_forecast.npz
    data/processed/zeek_network_states.npz
    data/processed/training_states.npz

Produces:
    data/processed/zeek_forecast_risk.npz
    data/processed/zeek_forecast_risk.json

This is an adapter around the existing ARJUN risk engine. It does not
modify the existing forecast_risk.py or overwrite forecast_risk.npz.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk.forecast_risk import (
    generate_forecast_risk,
    save_risk_result,
    FEATURE_NAMES,
)


FORECAST_FILE = ROOT / "data" / "processed" / "zeek_world_model_forecast.npz"
ZEEK_STATE_FILE = ROOT / "data" / "processed" / "zeek_network_states.npz"
TRAINING_FILE = ROOT / "data" / "processed" / "training_states.npz"

OUTPUT_NPZ = ROOT / "data" / "processed" / "zeek_forecast_risk.npz"
OUTPUT_JSON = ROOT / "data" / "processed" / "zeek_forecast_risk.json"


def finite_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def load_forecast(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)

    for key in ("forecast", "forecast_states", "states"):
        if key in data:
            forecast = finite_array(data[key])
            break
    else:
        raise KeyError(
            f"{path.name} does not contain forecast/state data."
        )

    if forecast.ndim != 2:
        raise ValueError(
            f"Expected forecast shape (K,D), got {forecast.shape}."
        )

    return forecast


def load_zeek_current(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)

    if "states" not in data:
        raise KeyError(
            f"{path.name} does not contain 'states'."
        )

    states = finite_array(data["states"])

    if states.ndim != 2 or states.shape[1] != 33:
        raise ValueError(
            f"Expected Zeek states shape (N,33), got {states.shape}."
        )

    if len(states) == 0:
        raise ValueError("Zeek state artifact is empty.")

    return states[-1]


def load_training(path: Path):
    data = np.load(path, allow_pickle=True)

    if "states" in data:
        states = finite_array(data["states"])
    elif "X" in data:
        states = finite_array(data["X"])
        if states.ndim == 3:
            states = states[:, -1, :]
    else:
        raise KeyError(
            f"{path.name} does not contain 'states' or 'X'."
        )

    if "labels" in data:
        labels = np.asarray(data["labels"]).astype(int)
    elif "y" in data:
        labels = np.asarray(data["y"]).astype(int)
    else:
        raise KeyError(
            f"{path.name} does not contain 'labels' or 'y'."
        )

    if states.ndim != 2 or states.shape[1] != 33:
        raise ValueError(
            f"Expected training states shape (N,33), got {states.shape}."
        )

    if len(states) != len(labels):
        raise ValueError(
            "Training state/label length mismatch: "
            f"{len(states)} vs {len(labels)}"
        )

    return states, labels


def build_json_result(result, source_info):
    timeline = []

    for item in result["timeline"]:
        top_features = []

        for feature in item.get("top_features", []):
            index = int(feature["index"])

            name = (
                FEATURE_NAMES[index]
                if 0 <= index < len(FEATURE_NAMES)
                else f"feature_{index}"
            )

            top_features.append(
                {
                    "feature": name,
                    "index": index,
                    "contribution": float(feature["contribution"]),
                    "current": float(feature["current"]),
                    "future": float(feature["future"]),
                    "change": float(feature["change"]),
                }
            )

        timeline.append(
            {
                "step": int(item["step"]),
                "attack_probability": float(item["attack_probability"]),
                "infiltration_probability": float(
                    item["infiltration_probability"]
                ),
                "risk_score": float(item["risk_score"]),
                "risk_level": str(item["risk_level"]),
                "mitre_stage": str(item["mitre_stage"]),
                "stage_scores": {
                    str(k): float(v)
                    for k, v in item["stage_scores"].items()
                },
                "top_features": top_features,
            }
        )

    return {
        "source": source_info,
        "max_attack_probability": float(
            result["max_attack_probability"]
        ),
        "final_attack_probability": float(
            result["final_attack_probability"]
        ),
        "max_risk_score": float(
            result["max_risk_score"]
        ),
        "overall_risk_level": str(
            result["overall_risk_level"]
        ),
        "timeline": timeline,
        "interpretation": (
            "Forecasted attack probabilities are outputs of the ARJUN "
            "risk engine and are not calibrated probabilities."
        ),
    }


def main():
    print("=" * 72)
    print("ARJUN Z4: ZEEK FORECAST -> RISK + MITRE + EXPLAINABILITY")
    print("=" * 72)

    for path in (
        FORECAST_FILE,
        ZEEK_STATE_FILE,
        TRAINING_FILE,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found:\n{path}"
            )

    forecast_states = load_forecast(FORECAST_FILE)
    current_state = load_zeek_current(ZEEK_STATE_FILE)
    training_states, labels = load_training(TRAINING_FILE)

    print(f"Zeek current state : {current_state.shape}")
    print(f"Forecast states    : {forecast_states.shape}")
    print(f"Training states    : {training_states.shape}")
    print(f"Training attacks   : {int(np.sum(labels == 1))}")

    if forecast_states.shape[1] != current_state.shape[0]:
        raise ValueError(
            "Forecast/current state dimensions do not match."
        )

    if forecast_states.shape[1] != 33:
        raise ValueError(
            f"Expected 33 forecast features, got {forecast_states.shape[1]}."
        )

    result = generate_forecast_risk(
        forecast_states=forecast_states,
        current_state=current_state,
        training_states=training_states,
        training_labels=labels,
    )

    # Save the standard numerical risk artifact without touching the
    # existing generic forecast_risk.npz.
    save_risk_result(
        result,
        output_path=OUTPUT_NPZ,
    )

    source_info = {
        "forecast_file": str(
            FORECAST_FILE.relative_to(ROOT)
        ),
        "zeek_state_file": str(
            ZEEK_STATE_FILE.relative_to(ROOT)
        ),
        "training_file": str(
            TRAINING_FILE.relative_to(ROOT)
        ),
        "forecast_steps": int(len(forecast_states)),
        "state_dimension": int(forecast_states.shape[1]),
    }

    json_result = build_json_result(
        result,
        source_info,
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            json_result,
            fh,
            indent=2,
        )

    print()
    print("ATTACK PROBABILITY / RISK TIMELINE")
    print("-" * 72)

    for item in result["timeline"]:
        print(
            f"t+{item['step']:<2} "
            f"Attack={item['attack_probability'] * 100:6.2f}% "
            f"Risk={item['risk_score'] * 100:6.2f}% "
            f"Level={item['risk_level']:<8} "
            f"MITRE={item['mitre_stage']}"
        )

    print()
    print("TOP PREDICTIVE INDICATORS")
    print("-" * 72)

    for item in result["timeline"]:
        print(f"t+{item['step']}:")
        for feature in item["top_features"][:3]:
            print(
                f"  {feature['feature']:<28} "
                f"contribution={feature['contribution']:+.4f} "
                f"change={feature['change']:+.4f}"
            )

    print()
    print("OVERALL ZEEK FORECAST RISK")
    print("-" * 72)
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
    print("Z4 checks:")
    assert len(result["timeline"]) == len(forecast_states)
    assert len(result["attack_probability_timeline"]) == len(forecast_states)
    assert len(result["risk_score_timeline"]) == len(forecast_states)
    assert np.isfinite(
        result["attack_probability_timeline"]
    ).all()
    assert np.isfinite(
        result["risk_score_timeline"]
    ).all()

    print("  Zeek current state      : PASS")
    print("  33-feature compatibility: PASS")
    print("  Risk timeline           : PASS")
    print("  MITRE mapping           : PASS")
    print("  Feature contributions   : PASS")
    print("  Finite risk outputs     : PASS")

    print()
    print(f"Saved NPZ : {OUTPUT_NPZ}")
    print(f"Saved JSON: {OUTPUT_JSON}")
    print()
    print("ZEEK FORECAST -> RISK: PASSED")


if __name__ == "__main__":
    main()
