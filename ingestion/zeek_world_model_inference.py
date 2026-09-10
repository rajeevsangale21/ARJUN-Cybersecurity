"""ARJUN Z3: Zeek conn.log -> trained World Model -> 5-step forecast."""
from __future__ import annotations
import pickle, shutil, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.zeek_to_state import build_zeek_states
from forecasting.hybrid_k_step_forecaster import (
    load_checkpoint, load_state_scaler, load_model, forecast_k_steps
)

INPUT = ROOT / "data" / "raw" / "logs" / "synthetic_conn_14windows.log"
OUT = ROOT / "data" / "processed"
CHECKPOINT = ROOT / "saved_models" / "hybrid_world_model.pt"


def main():
    print("=" * 72)
    print("ARJUN Z3: ZEEK -> WORLD MODEL INFERENCE")
    print("=" * 72)
    if not INPUT.exists():
        raise FileNotFoundError(f"Synthetic Zeek log not found: {INPUT}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"World Model checkpoint not found: {CHECKPOINT}")

    _, states, window_ids, graphs = build_zeek_states(INPUT, OUT)
    print(f"Zeek states     : {states.shape}")
    print(f"Graphs          : {len(graphs)}")
    if len(states) < 10:
        raise RuntimeError("Need at least 10 Zeek state windows for ARJUN temporal inference.")
    if len(graphs) != len(states):
        raise RuntimeError("State/graph alignment failure.")

    checkpoint = load_checkpoint()
    scaler = load_state_scaler(checkpoint)
    model, config = load_model(checkpoint, states.shape[1], graphs[0]["node_features"].shape[1])

    initial_states = states[-10:].astype(np.float32)
    graph_history = graphs[-10:]
    forecast = forecast_k_steps(model, initial_states, graph_history, scaler, k=5)

    np.savez_compressed(
        OUT / "zeek_world_model_forecast.npz",
        forecast=forecast,
        input_states=initial_states,
        input_window_ids=window_ids[-10:],
        feature_names=np.asarray(_feature_names(), dtype=object),
    )

    print(f"Temporal context: {initial_states.shape}")
    print(f"Model config    : {config}")
    print(f"Forecast shape  : {forecast.shape}")
    print("Forecast rollout:")
    for i, state in enumerate(forecast, 1):
        delta = state - (initial_states[-1] if i == 1 else forecast[i-2])
        print(f"  t+{i}: mean |delta|={np.mean(np.abs(delta)):.6f}, max |delta|={np.max(np.abs(delta)):.6f}")

    print("\nZ3 checks:")
    assert forecast.shape == (5, 33)
    assert np.isfinite(forecast).all()
    print("  33-feature state dimension : PASS")
    print("  10-state temporal context  : PASS")
    print("  Recursive 5-step rollout    : PASS")
    print("  Finite predictions          : PASS")
    print(f"\nSaved: {OUT / 'zeek_world_model_forecast.npz'}")
    print("ZEEK -> WORLD MODEL: PASSED")


def _feature_names():
    f = ROOT / "data" / "processed" / "training_states.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        if "feature_names" in d:
            return [str(x) for x in d["feature_names"]]
    return [f"feature_{i}" for i in range(33)]


if __name__ == "__main__":
    main()
