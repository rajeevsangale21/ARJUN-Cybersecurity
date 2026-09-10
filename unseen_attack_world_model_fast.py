from __future__ import annotations

"""
ARJUN PHASE A.3 v3
Correct unseen-attack-family evaluation.

v2 exposed an important evaluator bug: the production World Model was
trained with a signed-log1p + median/IQR StateScaler, but v2 treated the
saved median/scale as if they were a direct raw-space scaler.

The production checkpoint explicitly stores scaler.state_dict(), whose
values are the median and scale AFTER signed_log1p. This version restores
that exact transformation and inverse transformation.

It evaluates the existing production checkpoint only.
It does NOT retrain or modify the production model.
"""

from pathlib import Path
import json
import pickle
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.hybrid_world_model import HybridWorldModel
from world_model.hybrid_trainer import StateScaler


STATE_SEQUENCE_FILE = ROOT / "data" / "processed" / "graph_sequences_states.npz"
GRAPH_SEQUENCE_FILE = ROOT / "data" / "processed" / "graph_sequences.pkl"
UNSEEN_LABEL_FILE = ROOT / "data" / "processed" / "unseen_attack_state_labels.npz"
CHECKPOINT_FILE = ROOT / "saved_models" / "hybrid_world_model.pt"
REPORT_FILE = ROOT / "evaluation" / "unseen_attack_world_model_fast_report.json"

SEQUENCE_LENGTH = 10
BATCH_SIZE = 256
TARGET_FAMILY = "Infilteration"


def finite(x):
    return np.nan_to_num(
        np.asarray(x, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def load_checkpoint():
    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(CHECKPOINT_FILE)

    return torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=False,
    )


def load_exact_scaler(checkpoint):
    scaler_state = checkpoint.get("state_scaler")

    if not isinstance(scaler_state, dict):
        raise RuntimeError(
            "Checkpoint does not contain the expected state_scaler "
            "state_dict."
        )

    if "median" not in scaler_state or "scale" not in scaler_state:
        raise RuntimeError(
            "Checkpoint state_scaler is missing median/scale."
        )

    scaler = StateScaler.from_state_dict(
        scaler_state
    )

    return scaler


def normalize_graph_sequence(item):
    if isinstance(item, dict):
        for key in ("graphs", "graph_sequence", "sequence"):
            if key in item:
                item = item[key]
                break

    if not isinstance(item, (list, tuple)):
        raise TypeError(
            "Graph sequence must be a list/tuple."
        )

    result = []

    for graph in item:
        if not isinstance(graph, dict):
            raise TypeError(
                "Graph entry must be a dictionary."
            )

        node_features = graph.get(
            "node_features",
            graph.get("features"),
        )
        adjacency = graph.get(
            "adjacency",
            graph.get("adj"),
        )

        if node_features is None or adjacency is None:
            raise KeyError(
                "Graph must contain node_features and adjacency."
            )

        result.append(
            {
                **graph,
                "node_features": finite(node_features),
                "adjacency": finite(adjacency),
            }
        )

    return result


def load_sequences():
    if not STATE_SEQUENCE_FILE.exists():
        raise FileNotFoundError(STATE_SEQUENCE_FILE)

    if not GRAPH_SEQUENCE_FILE.exists():
        raise FileNotFoundError(GRAPH_SEQUENCE_FILE)

    data = np.load(
        STATE_SEQUENCE_FILE,
        allow_pickle=True,
    )

    X = finite(
        data["X"] if "X" in data else data["states"]
    )

    targets = finite(
        data["targets"] if "targets" in data else data["y"]
    )

    with open(GRAPH_SEQUENCE_FILE, "rb") as handle:
        graph_data = pickle.load(handle)

    if isinstance(graph_data, dict):
        for key in ("graphs", "graph_sequence", "sequence"):
            if key in graph_data:
                graph_data = graph_data[key]
                break

    return X, targets, list(graph_data)


def load_unseen_labels():
    if not UNSEEN_LABEL_FILE.exists():
        raise FileNotFoundError(UNSEEN_LABEL_FILE)

    data = np.load(
        UNSEEN_LABEL_FILE,
        allow_pickle=True,
    )

    for key in (
        "state_families",
        "families",
        "family",
        "attack_family",
    ):
        if key in data:
            return np.asarray(
                data[key],
                dtype=str,
            ).reshape(-1)

    raise KeyError(
        f"No attack-family labels found. Keys: {list(data.keys())}"
    )


def build_model(checkpoint, state_dimension):
    graph_dimension = int(
        checkpoint.get(
            "graph_input_dimension",
            6,
        )
    )

    model = HybridWorldModel(
        state_dimension=state_dimension,
        graph_input_dimension=graph_dimension,
        hidden_dimension=64,
        lstm_layers=1,
        residual_scale=0.75,
    )

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if state_dict is None:
        raise RuntimeError(
            "Checkpoint has no model_state_dict."
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model


def main():
    print("=" * 78)
    print("ARJUN PHASE A.3 - UNSEEN ATTACK GENERALIZATION v3")
    print("=" * 78)

    print("\n[1/6] LOADING SEQUENCE DATA")
    print("-" * 78)

    X, targets, graphs = load_sequences()
    families = load_unseen_labels()

    if X.ndim != 3:
        raise ValueError(
            f"Expected X shape (N,T,D), got {X.shape}"
        )

    n, t, d = X.shape

    if t != SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected sequence length {SEQUENCE_LENGTH}, got {t}"
        )

    if d != 33:
        raise ValueError(
            f"Expected 33 state features, got {d}"
        )

    if len(X) != len(graphs):
        raise ValueError(
            f"Sequence/graph mismatch: {len(X)} vs {len(graphs)}"
        )

    # State-family labels describe the original 56,700 states.
    # Sequence targets start after the first 10 context states.
    if len(families) == n + SEQUENCE_LENGTH:
        families = families[SEQUENCE_LENGTH:]
        print(
            f"Family labels aligned: skipped first "
            f"{SEQUENCE_LENGTH} context states."
        )
    elif len(families) != n:
        raise ValueError(
            f"Sequence/family mismatch: {n} vs {len(families)}"
        )

    print(f"Sequences         : {n:,}")
    print(f"Sequence length   : {t}")
    print(f"State dimension   : {d}")
    print(f"Graph sequences   : {len(graphs):,}")

    print("\n[2/6] BUILDING STRICT HELD-OUT TARGET SET")
    print("-" * 78)

    target_mask = families == TARGET_FAMILY
    target_indices = np.flatnonzero(target_mask)

    if len(target_indices) == 0:
        raise RuntimeError(
            f"No {TARGET_FAMILY} targets found."
        )

    strict_mask = target_mask.copy()

    for idx in target_indices:
        start = max(
            0,
            int(idx) - SEQUENCE_LENGTH + 1,
        )

        context = families[
            start:int(idx) + 1
        ]

        if not np.all(
            context == TARGET_FAMILY
        ):
            strict_mask[idx] = False

    strict_indices = np.flatnonzero(
        strict_mask
    )

    print(f"Target family     : {TARGET_FAMILY}")
    print(f"All target seqs   : {len(target_indices):,}")
    print(f"Strict held-out   : {len(strict_indices):,}")

    if len(strict_indices) == 0:
        raise RuntimeError(
            "No strict held-out targets remain."
        )

    print("\n[3/6] LOADING EXACT PRODUCTION CHECKPOINT")
    print("-" * 78)

    checkpoint = load_checkpoint()

    scaler = load_exact_scaler(
        checkpoint
    )

    model = build_model(
        checkpoint,
        d,
    )

    print(
        "World Model       : PASS"
    )
    print(
        "Exact StateScaler : PASS"
    )
    print(
        "Checkpoint modified: NO"
    )

    print("\n[4/6] RUNNING HELD-OUT INFERENCE")
    print("-" * 78)

    selected_X = X[strict_indices]
    selected_targets = targets[strict_indices]

    selected_graphs = []

    for idx in strict_indices:
        g = normalize_graph_sequence(
            graphs[int(idx)]
        )

        if len(g) != SEQUENCE_LENGTH:
            raise ValueError(
                f"Sequence {idx} has {len(g)} graphs; "
                f"expected {SEQUENCE_LENGTH}."
            )

        selected_graphs.append(g)

    # IMPORTANT:
    # Use the exact StateScaler implementation used during training:
    # signed_log1p -> median center -> IQR scale.
    normalized_X = scaler.transform(
        selected_X.reshape(
            -1,
            d,
        )
    ).reshape(
        len(selected_X),
        SEQUENCE_LENGTH,
        d,
    )

    normalized_X = finite(
        normalized_X
    )

    predictions = []

    with torch.no_grad():
        for start in range(
            0,
            len(normalized_X),
            BATCH_SIZE,
        ):
            stop = min(
                start + BATCH_SIZE,
                len(normalized_X),
            )

            xb = torch.from_numpy(
                normalized_X[start:stop]
            )

            gb = selected_graphs[
                start:stop
            ]

            delta_z = model.forward_batch(
                xb,
                gb,
            )

            delta_z = (
                delta_z
                .detach()
                .cpu()
                .numpy()
            )

            latest_z = normalized_X[
                start:stop,
                -1,
                :,
            ]

            pred_z = latest_z + delta_z

            pred_raw = scaler.inverse_transform(
                pred_z
            )

            predictions.append(
                finite(pred_raw)
            )

    predicted_targets = np.concatenate(
        predictions,
        axis=0,
    )

    if predicted_targets.shape != selected_targets.shape:
        raise RuntimeError(
            "Prediction/target shape mismatch: "
            f"{predicted_targets.shape} vs "
            f"{selected_targets.shape}"
        )

    if not np.isfinite(
        predicted_targets
    ).all():
        raise RuntimeError(
            "Predictions contain NaN/Inf."
        )

    print(
        f"Evaluated sequences: "
        f"{len(predicted_targets):,}"
    )
    print(
        "Inference         : PASS"
    )

    print("\n[5/6] COMPARING WORLD MODEL VS PERSISTENCE")
    print("-" * 78)

    current_states = selected_X[
        :,
        -1,
        :,
    ]

    model_error = (
        predicted_targets
        - selected_targets
    )

    persistence_error = (
        current_states
        - selected_targets
    )

    mse_model = float(
        np.mean(
            model_error ** 2
        )
    )

    mse_persistence = float(
        np.mean(
            persistence_error ** 2
        )
    )

    rmse_model = float(
        np.sqrt(mse_model)
    )

    rmse_persistence = float(
        np.sqrt(mse_persistence)
    )

    mae_model = float(
        np.mean(
            np.abs(model_error)
        )
    )

    mae_persistence = float(
        np.mean(
            np.abs(persistence_error)
        )
    )

    mae_gain = (
        (
            mae_persistence
            - mae_model
        )
        / mae_persistence
        * 100.0
        if mae_persistence > 0
        else 0.0
    )

    rmse_gain = (
        (
            rmse_persistence
            - rmse_model
        )
        / rmse_persistence
        * 100.0
        if rmse_persistence > 0
        else 0.0
    )

    # Correct normalized-space comparison.
    z_target = scaler.transform(
        selected_targets
    )

    z_current = scaler.transform(
        current_states
    )

    z_prediction = scaler.transform(
        predicted_targets
    )

    normalized_mae_model = float(
        np.mean(
            np.abs(
                z_prediction
                - z_target
            )
        )
    )

    normalized_mae_persistence = float(
        np.mean(
            np.abs(
                z_current
                - z_target
            )
        )
    )

    normalized_gain = (
        (
            normalized_mae_persistence
            - normalized_mae_model
        )
        / normalized_mae_persistence
        * 100.0
        if normalized_mae_persistence > 0
        else 0.0
    )

    # Verify that the model actually changes at least some predictions.
    mean_prediction_delta = float(
        np.mean(
            np.abs(
                predicted_targets
                - current_states
            )
        )
    )

    print(
        f"World Model MAE       : {mae_model:,.4f}"
    )
    print(
        f"Persistence MAE       : {mae_persistence:,.4f}"
    )
    print(
        f"MAE improvement       : {mae_gain:.2f}%"
    )
    print(
        f"World Model RMSE      : {rmse_model:,.4f}"
    )
    print(
        f"Persistence RMSE      : {rmse_persistence:,.4f}"
    )
    print(
        f"RMSE improvement      : {rmse_gain:.2f}%"
    )
    print(
        f"Normalized MAE        : "
        f"{normalized_mae_model:.6f}"
    )
    print(
        f"Normalized persistence: "
        f"{normalized_mae_persistence:.6f}"
    )
    print(
        f"Normalized improvement: "
        f"{normalized_gain:.2f}%"
    )
    print(
        f"Mean |prediction-current|: "
        f"{mean_prediction_delta:,.4f}"
    )

    if mean_prediction_delta == 0.0:
        print(
            "WARNING: model predictions exactly match persistence."
        )

    print("\n[6/6] SAVING REPORT")
    print("-" * 78)

    report = {
        "experiment": (
            "leave_one_attack_family_out_world_model"
        ),
        "held_out_family": TARGET_FAMILY,
        "sequence_length": SEQUENCE_LENGTH,
        "state_dimension": d,
        "all_target_sequences": int(
            len(target_indices)
        ),
        "strict_held_out_sequences": int(
            len(strict_indices)
        ),
        "world_model": {
            "mae": mae_model,
            "rmse": rmse_model,
            "mse": mse_model,
            "normalized_mae": normalized_mae_model,
        },
        "persistence": {
            "mae": mae_persistence,
            "rmse": rmse_persistence,
            "mse": mse_persistence,
            "normalized_mae": (
                normalized_mae_persistence
            ),
        },
        "improvement_percent": {
            "mae": mae_gain,
            "rmse": rmse_gain,
            "normalized_mae": normalized_gain,
        },
        "mean_prediction_current_distance": (
            mean_prediction_delta
        ),
        "production_checkpoint_modified": False,
        "scaler": (
            "exact production StateScaler: "
            "signed_log1p + median + IQR"
        ),
    }

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Report: {REPORT_FILE}"
    )

    print("\n" + "=" * 78)
    print(
        "PHASE A.3 UNSEEN ATTACK EVALUATION: COMPLETE"
    )
    print("=" * 78)

    if mae_gain > 0:
        print(
            f"World Model beats persistence by "
            f"{mae_gain:.2f}% MAE on strict "
            f"{TARGET_FAMILY} targets."
        )
    else:
        print(
            "World Model did not beat persistence "
            "on this held-out family."
        )

    print(
        "Production checkpoint was NOT modified."
    )


if __name__ == "__main__":
    main()
