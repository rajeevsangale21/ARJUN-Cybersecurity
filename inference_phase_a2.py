from __future__ import annotations

"""
ARJUN PHASE A.2
Production integration of the improved XGBoost classifier with the
existing Temporal GNN + LSTM World Model, K-step forecast, MITRE
mapping, unified risk engine, and feature contributions.

Pipeline:

Zeek state history
    -> Improved XGBoost current attack probability + family
    -> Temporal GNN + LSTM World Model
    -> 5-step recursive future states
    -> Improved XGBoost on future states
    -> UnifiedAnalysisService
    -> MITRE stage + risk + feature contributions
    -> production_forecast.npz/json
"""

from pathlib import Path
import json
import pickle
import sys

import joblib
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ZEEK_STATES = ROOT / "data" / "processed" / "zeek_network_states.npz"
ZEEK_GRAPHS = ROOT / "data" / "processed" / "zeek_network_graphs.pkl"

WORLD_MODEL_FILE = ROOT / "saved_models" / "hybrid_world_model.pt"
IMPROVED_XGB_FILE = (
    ROOT / "saved_models" / "xgboost_attack_classifier_improved.pkl"
)
FAMILY_XGB_FILE = (
    ROOT / "saved_models" / "xgboost_attack_family_classifier.pkl"
)

OUTPUT_NPZ = ROOT / "data" / "processed" / "production_forecast.npz"
OUTPUT_JSON = ROOT / "data" / "processed" / "production_forecast.json"

SEQUENCE_LENGTH = 10
FORECAST_STEPS = 5
EXPECTED_DIMENSION = 33


# ---------------------------------------------------------------------------
# Existing ARJUN modules
# ---------------------------------------------------------------------------

from forecasting.hybrid_k_step_forecaster import (
    ForecastStateScaler,
    find_value,
    get_model_config,
    normalize_graph_sequence,
    forecast_k_steps,
)
from world_model.hybrid_world_model import HybridWorldModel
from risk.unified_analysis import UnifiedAnalysisService
from world_model.state_encoder import NetworkStateEncoder


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def finite_array(value, dtype=np.float32):
    arr = np.asarray(value, dtype=dtype)
    return np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required ARJUN artifact was not found:\n{path}"
        )


# ---------------------------------------------------------------------------
# Load Zeek state history
# ---------------------------------------------------------------------------

def load_zeek_states():
    require_file(ZEEK_STATES)

    data = np.load(ZEEK_STATES, allow_pickle=True)

    if "states" not in data:
        raise KeyError(
            f"{ZEEK_STATES.name} must contain 'states'."
        )

    states = finite_array(data["states"])

    if states.ndim != 2 or states.shape[1] != EXPECTED_DIMENSION:
        raise ValueError(
            f"Expected Zeek states shape (N,33), got {states.shape}."
        )

    if len(states) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Need at least {SEQUENCE_LENGTH} Zeek states, "
            f"found {len(states)}."
        )

    return states


# ---------------------------------------------------------------------------
# Load Zeek graphs
# ---------------------------------------------------------------------------

def load_zeek_graph_history():
    require_file(ZEEK_GRAPHS)

    with open(ZEEK_GRAPHS, "rb") as handle:
        raw = pickle.load(handle)

    if isinstance(raw, dict):
        for key in ("graphs", "graph_sequence", "sequence"):
            if key in raw:
                raw = raw[key]
                break

    if not isinstance(raw, (list, tuple)):
        raise TypeError(
            "Zeek graph artifact must contain a list/tuple of graphs."
        )

    graphs = list(raw)

    if len(graphs) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Need at least {SEQUENCE_LENGTH} Zeek graphs, "
            f"found {len(graphs)}."
        )

    # The Zeek graph artifact is one graph per state window.
    history = graphs[-SEQUENCE_LENGTH:]

    return normalize_graph_sequence(history)


# ---------------------------------------------------------------------------
# Load improved XGBoost
# ---------------------------------------------------------------------------

def load_improved_xgboost():
    require_file(IMPROVED_XGB_FILE)

    artifact = joblib.load(IMPROVED_XGB_FILE)

    if not isinstance(artifact, dict):
        raise ValueError(
            "Improved XGBoost artifact must be a dictionary."
        )

    if "model" not in artifact:
        raise ValueError(
            "Improved XGBoost artifact is missing 'model'."
        )

    if "normalizer" not in artifact:
        raise ValueError(
            "Improved XGBoost artifact is missing 'normalizer'."
        )

    model = artifact["model"]
    normalizer = artifact["normalizer"]
    threshold = float(artifact.get("threshold", 0.50))

    return model, normalizer, threshold


def load_family_xgboost():
    require_file(FAMILY_XGB_FILE)

    artifact = joblib.load(FAMILY_XGB_FILE)

    if not isinstance(artifact, dict):
        raise ValueError(
            "Attack-family XGBoost artifact must be a dictionary."
        )

    if "model" not in artifact:
        raise ValueError(
            "Attack-family artifact is missing 'model'."
        )

    model = artifact["model"]
    classes = np.asarray(
        artifact.get("classes"),
        dtype=str,
    )

    if len(classes) == 0:
        raise ValueError(
            "Attack-family artifact contains no classes."
        )

    return model, classes


# ---------------------------------------------------------------------------
# Classifier inference
# ---------------------------------------------------------------------------

def classifier_probabilities(model, normalizer, states):
    states = finite_array(states)

    normalized = normalizer.transform(states)
    normalized = finite_array(normalized)

    probabilities = model.predict_proba(normalized)

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    if probabilities.ndim == 2:
        if probabilities.shape[1] != 2:
            raise ValueError(
                f"Binary XGBoost expected 2 probability columns, "
                f"got {probabilities.shape}."
            )
        probabilities = probabilities[:, 1]

    probabilities = np.clip(
        probabilities.reshape(-1),
        0.0,
        1.0,
    )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "XGBoost produced NaN/Inf probabilities."
        )

    return probabilities


def family_prediction(model, classes, normalizer, state):
    normalized = normalizer.transform(
        finite_array(state).reshape(1, -1)
    )

    probabilities = np.asarray(
        model.predict_proba(normalized),
        dtype=np.float32,
    )

    if probabilities.ndim != 2:
        raise ValueError(
            f"Unexpected family probability shape: {probabilities.shape}"
        )

    index = int(np.argmax(probabilities[0]))

    if index >= len(classes):
        raise RuntimeError(
            "Family model class index exceeds class list."
        )

    return (
        str(classes[index]),
        float(probabilities[0, index]),
    )


# ---------------------------------------------------------------------------
# World Model loading
# ---------------------------------------------------------------------------

def load_world_model():
    require_file(WORLD_MODEL_FILE)

    checkpoint = torch.load(
        WORLD_MODEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    state_dimension = int(
        find_value(
            checkpoint,
            [
                "state_dimension",
                "input_dimension",
            ],
        )
        or EXPECTED_DIMENSION
    )

    graph_dimension = int(
        find_value(
            checkpoint,
            [
                "graph_input_dimension",
                "graph_dimension",
            ],
        )
        or 6
    )

    config = get_model_config(
        checkpoint,
        state_dimension,
        graph_dimension,
    )

    model = HybridWorldModel(
        state_dimension=state_dimension,
        graph_input_dimension=graph_dimension,
        hidden_dimension=config["hidden_dimension"],
        lstm_layers=config["lstm_layers"],
        residual_scale=config["residual_scale"],
    )

    state_dict = find_value(
        checkpoint,
        [
            "model_state_dict",
            "state_dict",
        ],
    )

    if state_dict is None:
        raise RuntimeError(
            "World Model checkpoint does not contain model weights."
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    # Recover the exact training-time scaler.
    scaler_data = find_value(
        checkpoint,
        [
            "state_scaler",
            "scaler",
            "state_normalizer",
        ],
    )

    center = None
    scale = None

    if scaler_data is not None:
        center = find_value(
            scaler_data,
            [
                "center",
                "median",
                "location",
                "mean",
            ],
        )
        scale = find_value(
            scaler_data,
            [
                "scale",
                "iqr",
                "std",
                "spread",
            ],
        )

    if center is None:
        center = find_value(
            checkpoint,
            [
                "state_center",
                "state_median",
                "state_mean",
            ],
        )

    if scale is None:
        scale = find_value(
            checkpoint,
            [
                "state_scale",
                "state_iqr",
                "state_std",
            ],
        )

    if center is None or scale is None:
        raise RuntimeError(
            "Could not recover World Model state scaler."
        )

    scaler = ForecastStateScaler(
        center=center,
        scale=scale,
    )

    return model, scaler, config


# ---------------------------------------------------------------------------
# Main production pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("ARJUN PHASE A.2 - PRODUCTION INTELLIGENCE PIPELINE")
    print("=" * 78)

    print("\n[1/7] LOADING ZEEK TELEMETRY")
    print("-" * 78)

    states = load_zeek_states()
    graph_history = load_zeek_graph_history()

    current_state = states[-1]
    state_history = states[-SEQUENCE_LENGTH:]

    print(f"Zeek states       : {states.shape}")
    print(f"Context states    : {state_history.shape}")
    print(f"Graph context     : {len(graph_history)}")
    print("Telemetry load    : PASS")

    print("\n[2/7] LOADING IMPROVED XGBOOST")
    print("-" * 78)

    xgb_model, xgb_normalizer, threshold = load_improved_xgboost()
    family_model, family_classes = load_family_xgboost()

    current_probability = float(
        classifier_probabilities(
            xgb_model,
            xgb_normalizer,
            current_state.reshape(1, -1),
        )[0]
    )

    current_family, family_confidence = family_prediction(
        family_model,
        family_classes,
        xgb_normalizer,
        current_state,
    )

    current_attack = current_probability >= threshold

    print(f"Attack probability : {current_probability:.4%}")
    print(f"Decision threshold : {threshold:.2f}")
    print(f"Current decision   : {'ATTACK' if current_attack else 'BENIGN'}")
    print(f"Attack family      : {current_family}")
    print(f"Family confidence  : {family_confidence:.4%}")
    print("XGBoost inference  : PASS")

    print("\n[3/7] LOADING ARJUN WORLD MODEL")
    print("-" * 78)

    world_model, world_scaler, world_config = load_world_model()

    print(
        f"State dimension    : {world_config['state_dimension']}"
    )
    print(
        f"Graph dimension    : {world_config['graph_dimension']}"
    )
    print(
        f"Hidden dimension   : {world_config['hidden_dimension']}"
    )
    print("World Model load   : PASS")

    print("\n[4/7] GENERATING 5-STEP FUTURE STATES")
    print("-" * 78)

    future_states = forecast_k_steps(
        model=world_model,
        initial_states=state_history,
        graph_sequence=graph_history,
        scaler=world_scaler,
        k=FORECAST_STEPS,
    )

    future_states = finite_array(future_states)

    expected = (
        FORECAST_STEPS,
        EXPECTED_DIMENSION,
    )

    if future_states.shape != expected:
        raise RuntimeError(
            f"Unexpected future state shape: "
            f"{future_states.shape}; expected {expected}"
        )

    print(f"Future states      : {future_states.shape}")
    print(f"Finite forecast    : {np.isfinite(future_states).all()}")
    print("K-step rollout     : PASS")

    print("\n[5/7] XGBOOST ON PREDICTED FUTURE STATES")
    print("-" * 78)

    future_probabilities = classifier_probabilities(
        xgb_model,
        xgb_normalizer,
        future_states,
    )

    future_families = []

    for future_state in future_states:
        family, confidence = family_prediction(
            family_model,
            family_classes,
            xgb_normalizer,
            future_state,
        )
        future_families.append(
            {
                "family": family,
                "confidence": confidence,
            }
        )

    for step, probability in enumerate(
        future_probabilities,
        start=1,
    ):
        family = future_families[step - 1]
        print(
            f"t+{step}  "
            f"Attack={probability:.2%}  "
            f"Family={family['family']}  "
            f"Confidence={family['confidence']:.2%}"
        )

    print("Future classifier : PASS")

    print("\n[6/7] UNIFIED RISK + MITRE + EXPLAINABILITY")
    print("-" * 78)

    feature_names = list(
        NetworkStateEncoder().feature_names
    )

    if len(feature_names) != EXPECTED_DIMENSION:
        raise RuntimeError(
            f"Expected 33 feature names, got {len(feature_names)}."
        )

    service = UnifiedAnalysisService(feature_names)

    analysis = service.analyze(
        current_state=current_state,
        future_states=future_states,
        current_attack_probability=current_probability,
        future_attack_probabilities=future_probabilities,
    )

    timeline = analysis.get("timeline", [])

    for item in timeline:
        print(
            f"t+{item.get('step', '?')}  "
            f"Risk={float(item.get('risk_score', 0.0)):.2%}  "
            f"Level={item.get('risk_level', 'UNKNOWN')}  "
            f"Stage={item.get('mitre_stage', 'Unknown')}"
        )

    summary = analysis.get("summary", {})

    print(
        f"\nPeak future attack probability : "
        f"{float(np.max(future_probabilities)):.2%}"
    )
    print(
        f"Maximum unified risk           : "
        f"{float(summary.get('maximum_forecast_risk', 0.0)):.2%}"
    )
    print(
        f"Maximum risk level             : "
        f"{summary.get('maximum_risk_level', 'UNKNOWN')}"
    )
    print(
        f"Highest-risk MITRE stage       : "
        f"{summary.get('highest_risk_mitre_stage', 'Unknown')}"
    )

    print("Unified analysis : PASS")

    print("\n[7/7] SAVING PRODUCTION ARTIFACTS")
    print("-" * 78)

    contributions = analysis.get(
        "feature_contributions",
        analysis.get("contributions", []),
    )

    output_json = {
        "pipeline": "ARJUN_PHASE_A2",
        "classifier": {
            "model": "improved_xgboost",
            "threshold": threshold,
            "current_attack_probability": current_probability,
            "current_attack_decision": bool(current_attack),
            "current_attack_family": current_family,
            "family_confidence": family_confidence,
        },
        "world_model": {
            "type": "Temporal GNN + LSTM",
            "transition": "P(S_t+1 | S_t, G_t)",
            "context_length": SEQUENCE_LENGTH,
            "forecast_steps": FORECAST_STEPS,
        },
        "future_attack_probabilities": [
            float(x) for x in future_probabilities
        ],
        "future_attack_families": future_families,
        "timeline": timeline,
        "stage_timeline": analysis.get(
            "stage_timeline",
            [],
        ),
        "summary": summary,
        "feature_contributions": contributions,
        "note": (
            "Attack probabilities are model outputs and are not "
            "calibrated real-world probabilities."
        ),
    }

    OUTPUT_NPZ.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_NPZ,
        current_state=current_state.astype(np.float32),
        future_states=future_states.astype(np.float32),
        current_attack_probability=np.float32(
            current_probability
        ),
        future_attack_probability=future_probabilities.astype(
            np.float32
        ),
        risk_score=np.asarray(
            [
                float(x.get("risk_score", 0.0))
                for x in timeline
            ],
            dtype=np.float32,
        ),
        mitre_stage=np.asarray(
            [
                str(x.get("mitre_stage", "Unknown"))
                for x in timeline
            ],
            dtype=str,
        ),
        current_attack_family=np.asarray(
            current_family
        ),
        threshold=np.float32(threshold),
    )

    OUTPUT_JSON.write_text(
        json.dumps(
            output_json,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"NPZ output         : {OUTPUT_NPZ}")
    print(f"JSON output        : {OUTPUT_JSON}")

    # Final numerical integrity checks.
    saved = np.load(
        OUTPUT_NPZ,
        allow_pickle=True,
    )

    if saved["future_states"].shape != (
        FORECAST_STEPS,
        EXPECTED_DIMENSION,
    ):
        raise RuntimeError(
            "Saved future state shape validation failed."
        )

    if not np.isfinite(
        saved["future_states"]
    ).all():
        raise RuntimeError(
            "Saved future states contain NaN/Inf."
        )

    if len(
        saved["future_attack_probability"]
    ) != FORECAST_STEPS:
        raise RuntimeError(
            "Saved future probability count is incorrect."
        )

    print("Saved artifact check: PASS")

    print("\n" + "=" * 78)
    print("PHASE A.2 PRODUCTION INTEGRATION: COMPLETE")
    print("=" * 78)
    print(
        "Improved XGBoost -> World Model -> K-step rollout -> "
        "future XGBoost -> MITRE/Risk/Explainability"
    )


if __name__ == "__main__":
    main()
