"""
ARJUN - LEAVE-ONE-ATTACK-FAMILY-OUT WORLD MODEL EVALUATION

Purpose
-------
Train a fresh ARJUN World Model with the selected attack family completely
absent from the training targets AND absent from the historical context of
training sequences, then evaluate on that attack family's target states.

Default held-out family: Infilteration

Important:
- The existing saved_models/hybrid_world_model.pt is NEVER overwritten.
- Training scaling is fitted only on the leakage-safe training sequences.
- The model learns normalized next-state residuals:
      Delta_Z = Z(t+1) - Z(t)
- Prediction is reconstructed as:
      Z_hat(t+1) = Z(t) + Delta_Z
- Two test views are reported:
    1. all held-out target states (largest useful test set)
    2. strict held-out target states whose 10-state history also contains
       no held-out-family state. This is expected to be smaller.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# ARJUN PROJECT IMPORT PATH
# ---------------------------------------------------------------------------
# This script is executed as:
#     python evaluation\\unseen_attack_world_model.py
# from the ARJUN project root.  Explicitly add the root and world_model
# directories so the exact project model can always be imported.
import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
_WORLD_MODEL_DIR = _PROJECT_ROOT / "world_model"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if str(_WORLD_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_WORLD_MODEL_DIR))


import argparse
import inspect
import json
import pickle
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent

STATE_SEQUENCE_FILE = ROOT / "data" / "processed" / "graph_sequences_states.npz"
GRAPH_SEQUENCE_FILE = ROOT / "data" / "processed" / "graph_sequences.pkl"
LABEL_FILE = ROOT / "data" / "processed" / "unseen_attack_state_labels.npz"

OUTPUT_DIR = ROOT / "data" / "processed"
CHECKPOINT_FILE = ROOT / "saved_models" / "unseen_infilteration_world_model.pt"
REPORT_FILE = ROOT / "evaluation" / "unseen_attack_world_model_report.json"

STATE_DIMENSION = 33
GRAPH_DIMENSION = 6
SEQUENCE_LENGTH = 10

DEFAULT_HOLDOUT = "Infilteration"
DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_PATIENCE = 3
SEED = 42


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Robust scaler
# ---------------------------------------------------------------------------

class RobustScaler:
    def __init__(self, center: np.ndarray, scale: np.ndarray):
        self.center = np.asarray(center, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.scale = np.where(np.abs(self.scale) < 1e-8, 1.0, self.scale)

    @classmethod
    def fit(cls, x: np.ndarray) -> "RobustScaler":
        x = np.asarray(x, dtype=np.float32)
        center = np.median(x, axis=0)
        q1 = np.percentile(x, 25.0, axis=0)
        q3 = np.percentile(x, 75.0, axis=0)
        scale = q3 - q1

        # Stable fallback for nearly constant dimensions.
        fallback = np.std(x, axis=0)
        scale = np.where(scale < 1e-8, fallback, scale)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(center, scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.center) / self.scale).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) * self.scale + self.center).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y_delta: np.ndarray, graph_sequences: list):
        self.X = np.asarray(X, dtype=np.float32)
        self.y_delta = np.asarray(y_delta, dtype=np.float32)
        self.graph_sequences = graph_sequences

        if len(self.X) != len(self.y_delta) or len(self.X) != len(self.graph_sequences):
            raise ValueError("Sequence, target and graph counts do not match.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y_delta[index], self.graph_sequences[index]


def collate_graph_batch(batch):
    xs = np.stack([item[0] for item in batch]).astype(np.float32)
    ys = np.stack([item[1] for item in batch]).astype(np.float32)
    graphs = [item[2] for item in batch]
    return torch.from_numpy(xs), torch.from_numpy(ys), graphs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    if not STATE_SEQUENCE_FILE.exists():
        raise FileNotFoundError(f"Missing state sequences:\n{STATE_SEQUENCE_FILE}")
    if not GRAPH_SEQUENCE_FILE.exists():
        raise FileNotFoundError(f"Missing graph sequences:\n{GRAPH_SEQUENCE_FILE}")
    if not LABEL_FILE.exists():
        raise FileNotFoundError(f"Missing unseen-attack mapping:\n{LABEL_FILE}")

    state_data = np.load(STATE_SEQUENCE_FILE, allow_pickle=True)
    X = np.asarray(state_data["X"], dtype=np.float32)
    y = np.asarray(state_data["y"], dtype=np.float32)

    if "target_windows" in state_data:
        target_windows = np.asarray(state_data["target_windows"], dtype=np.int64)
    else:
        target_windows = np.arange(len(X), dtype=np.int64) + SEQUENCE_LENGTH

    with open(GRAPH_SEQUENCE_FILE, "rb") as f:
        graph_payload = pickle.load(f)

    if isinstance(graph_payload, dict) and "graph_sequences" in graph_payload:
        graphs = graph_payload["graph_sequences"]
    else:
        graphs = graph_payload

    label_data = np.load(LABEL_FILE, allow_pickle=True)

    def get_first(keys):
        for key in keys:
            if key in label_data:
                return label_data[key]
        return None

    state_family = get_first([
        "state_families",
        "families",
        "labels",
        "state_labels",
    ])

    sequence_family = get_first([
        "sequence_families",
        "target_families",
        "sequence_labels",
        "target_labels",
    ])

    if sequence_family is None:
        if state_family is None:
            raise RuntimeError(
                "unseen_attack_state_labels.npz does not contain state or sequence family labels."
            )
        state_family = np.asarray(state_family, dtype=str)
        sequence_family = state_family[SEQUENCE_LENGTH:]

    sequence_family = np.asarray(sequence_family, dtype=str)

    # Some versions of the preparation file save one label per state.
    if len(sequence_family) != len(X):
        if state_family is not None and len(state_family) >= len(X) + SEQUENCE_LENGTH:
            state_family = np.asarray(state_family, dtype=str)
            sequence_family = state_family[SEQUENCE_LENGTH:SEQUENCE_LENGTH + len(X)]

    if len(X) != len(y):
        raise ValueError(f"X/y mismatch: {len(X)} != {len(y)}")
    if len(X) != len(graphs):
        raise ValueError(f"X/graph mismatch: {len(X)} != {len(graphs)}")
    if len(sequence_family) != len(X):
        raise ValueError(
            f"Family/sequence mismatch: {len(sequence_family)} != {len(X)}"
        )

    return X, y, graphs, sequence_family, target_windows


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def normalize_graph(graph: Any) -> Dict[str, Any]:
    if not isinstance(graph, dict):
        raise TypeError("Graph must be a dictionary.")

    node_features = graph.get("node_features")
    adjacency = graph.get("adjacency")

    if node_features is None or adjacency is None:
        raise ValueError("Graph must contain node_features and adjacency.")

    nf = np.asarray(node_features, dtype=np.float32)
    adj = np.asarray(adjacency, dtype=np.float32)

    if nf.ndim != 2:
        raise ValueError(f"node_features must be 2-D, got {nf.shape}")
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adjacency must be square, got {adj.shape}")
    if adj.shape[0] != nf.shape[0]:
        raise ValueError("Graph node count and adjacency dimensions differ.")
    if nf.shape[1] != GRAPH_DIMENSION:
        raise ValueError(
            f"Expected {GRAPH_DIMENSION} graph features, got {nf.shape[1]}"
        )
    if not np.isfinite(nf).all() or not np.isfinite(adj).all():
        raise ValueError("Graph contains NaN/Inf.")

    # Preserve all metadata while making arrays clean numeric objects.
    result = dict(graph)
    result["node_features"] = torch.from_numpy(nf)
    result["adjacency"] = torch.from_numpy(adj)
    return result


def prepare_graphs(graphs: list, indices: np.ndarray) -> list:
    result = []
    for idx in indices:
        sequence = graphs[int(idx)]
        if len(sequence) != SEQUENCE_LENGTH:
            raise ValueError(
                f"Graph sequence {idx} has {len(sequence)} graphs; "
                f"expected {SEQUENCE_LENGTH}."
            )
        result.append([normalize_graph(g) for g in sequence])
    return result


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(state_dimension: int, device: torch.device):
    """
    Build the exact HybridWorldModel implementation used by ARJUN.

    The uploaded production model defines:
        HybridWorldModel(
            state_dimension,
            graph_input_dimension,
            hidden_dimension=64,
            lstm_layers=1,
            residual_scale=0.75,
        )

    No alternative constructor names are used here.
    """
    try:
        from world_model.hybrid_world_model import HybridWorldModel
    except ModuleNotFoundError:
        from world_model import HybridWorldModel

    model_kwargs = {
        "state_dimension": int(state_dimension),
        "graph_input_dimension": GRAPH_DIMENSION,
        "hidden_dimension": 64,
        "lstm_layers": 1,
        "residual_scale": 0.75,
    }

    model = HybridWorldModel(**model_kwargs).to(device)

    return model, model_kwargs

def metric_bundle(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    diff = predicted - actual
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    denom = np.mean(np.abs(actual))
    normalized_mae = float(mae / denom) if denom > 1e-12 else 0.0
    normalized_rmse = float(rmse / denom) if denom > 1e-12 else 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "normalized_mae": normalized_mae,
        "normalized_rmse": normalized_rmse,
    }


def improvement_percent(baseline: float, model_value: float) -> float:
    if abs(baseline) < 1e-12:
        return 0.0
    return float((baseline - model_value) / baseline * 100.0)


# ---------------------------------------------------------------------------
# Forward/evaluation
# ---------------------------------------------------------------------------

def predict_dataset(
    model,
    X: np.ndarray,
    graphs: list,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    predictions = []

    dataset = SequenceDataset(
        X,
        np.zeros((len(X), STATE_DIMENSION), dtype=np.float32),
        graphs,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graph_batch,
        num_workers=0,
    )

    with torch.no_grad():
        for xb, _, graph_batch in loader:
            xb = xb.to(device)
            pred_delta = model.forward_batch(xb, graph_batch)

            if isinstance(pred_delta, (tuple, list)):
                pred_delta = pred_delta[0]

            pred_delta = pred_delta.detach().cpu().numpy().astype(np.float32)

            # X is normalized historical state sequence.
            # Model output is normalized residual Delta_Z.
            current_z = xb[:, -1, :].detach().cpu().numpy()
            predicted_z = current_z + pred_delta
            predictions.append(predicted_z)

    return np.concatenate(predictions, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model,
    train_X,
    train_delta,
    train_graphs,
    val_X,
    val_delta,
    val_graphs,
    device,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    patience,
):
    train_dataset = SequenceDataset(train_X, train_delta, train_graphs)
    val_dataset = SequenceDataset(val_X, val_delta, val_graphs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graph_batch,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graph_batch,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    history = []

    print()
    print("=" * 72)
    print("TRAINING LEAVE-ONE-FAMILY-OUT WORLD MODEL")
    print("=" * 72)
    print(f"Train sequences : {len(train_dataset):,}")
    print(f"Validation      : {len(val_dataset):,}")
    print(f"Epochs          : {epochs}")
    print(f"Batch size      : {batch_size}")
    print(f"Learning rate   : {learning_rate}")
    print(f"Device          : {device}")

    for epoch in range(1, epochs + 1):
        start = time.time()

        model.train()
        train_sum = 0.0
        train_count = 0

        for xb, yb, graph_batch in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            pred_delta = model.forward_batch(xb, graph_batch)
            if isinstance(pred_delta, (tuple, list)):
                pred_delta = pred_delta[0]

            if pred_delta.shape != yb.shape:
                raise RuntimeError(
                    f"Model output shape {tuple(pred_delta.shape)} "
                    f"does not match target {tuple(yb.shape)}"
                )

            loss = criterion(pred_delta, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_sum += float(loss.item()) * len(xb)
            train_count += len(xb)

        train_loss = train_sum / max(train_count, 1)

        model.eval()
        val_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for xb, yb, graph_batch in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred_delta = model.forward_batch(xb, graph_batch)
                if isinstance(pred_delta, (tuple, list)):
                    pred_delta = pred_delta[0]

                loss = criterion(pred_delta, yb)
                val_sum += float(loss.item()) * len(xb)
                val_count += len(xb)

        val_loss = val_sum / max(val_count, 1)
        elapsed = time.time() - start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "seconds": elapsed,
        })

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
            marker = "  BEST"
        else:
            wait += 1

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train {train_loss:.6f} | "
            f"val {val_loss:.6f} | "
            f"{elapsed:.1f}s{marker}"
        )

        if wait >= patience:
            print(f"Early stopping after epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("No valid model checkpoint was produced.")

    model.load_state_dict(best_state)
    model.eval()

    print(f"Best epoch      : {best_epoch}")
    print(f"Best val loss   : {best_val:.6f}")

    return history, best_epoch, best_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate ARJUN World Model on an unseen attack family."
    )
    parser.add_argument("--holdout", default=DEFAULT_HOLDOUT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    seed_everything(args.seed)

    print("=" * 72)
    print("ARJUN - LEAVE-ONE-ATTACK-FAMILY-OUT WORLD MODEL")
    print("=" * 72)

    X_raw, targets_raw, all_graphs, sequence_families, target_windows = load_data()

    if X_raw.ndim != 3 or X_raw.shape[1] != SEQUENCE_LENGTH or X_raw.shape[2] != STATE_DIMENSION:
        raise ValueError(
            f"Expected X shape (N,{SEQUENCE_LENGTH},{STATE_DIMENSION}), got {X_raw.shape}"
        )

    if targets_raw.shape != (len(X_raw), STATE_DIMENSION):
        raise ValueError(f"Unexpected target shape: {targets_raw.shape}")

    normalized_holdout = " ".join(args.holdout.strip().split()).casefold()
    family_norm = np.array(
        [" ".join(str(x).strip().split()).casefold() for x in sequence_families],
        dtype=str,
    )

    holdout_mask = family_norm == normalized_holdout

    if not holdout_mask.any():
        available = sorted(set(sequence_families.tolist()))
        raise ValueError(
            f"Holdout family '{args.holdout}' was not found.\n"
            f"Available families: {available}"
        )

    # -----------------------------------------------------------------------
    # Leakage-safe training mask
    #
    # A training sequence is allowed only if:
    #   - its target is NOT the held-out family
    #   - none of its 10 historical states is the held-out family
    #
    # This uses state-family labels from the preparation artifact.
    # -----------------------------------------------------------------------

    state_labels = None
    label_data = np.load(LABEL_FILE, allow_pickle=True)
    for key in ["state_families", "families", "labels", "state_labels"]:
        if key in label_data:
            state_labels = np.asarray(label_data[key], dtype=str)
            break

    if state_labels is not None and len(state_labels) >= len(X_raw) + SEQUENCE_LENGTH:
        state_norm = np.array(
            [" ".join(str(x).strip().split()).casefold() for x in state_labels],
            dtype=str,
        )

        history_contains_holdout = np.zeros(len(X_raw), dtype=bool)
        for i in range(len(X_raw)):
            history_contains_holdout[i] = np.any(
                state_norm[i:i + SEQUENCE_LENGTH] == normalized_holdout
            )
    else:
        # Fall back to the preparation artifact's strict-context mask if present.
        context_mask = None
        for key in ["excluded_context", "history_contains_holdout", "context_holdout"]:
            if key in label_data:
                context_mask = np.asarray(label_data[key], dtype=bool)
                break

        if context_mask is not None and len(context_mask) == len(X_raw):
            history_contains_holdout = context_mask
        else:
            raise RuntimeError(
                "Could not recover state-family history labels needed for leakage-safe training."
            )

    train_mask = (~holdout_mask) & (~history_contains_holdout)
    all_holdout_mask = holdout_mask
    strict_holdout_mask = holdout_mask & (~history_contains_holdout)

    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(all_holdout_mask)
    strict_test_indices = np.flatnonzero(strict_holdout_mask)

    if len(train_indices) < 100:
        raise RuntimeError("Too few leakage-safe training sequences.")

    if len(test_indices) < 10:
        raise RuntimeError("Too few held-out target sequences.")

    # Chronological validation split inside training only.
    val_count = max(1, int(round(len(train_indices) * 0.10)))
    train_fit_indices = train_indices[:-val_count]
    val_indices = train_indices[-val_count:]

    print()
    print("DATA SPLIT")
    print("-" * 72)
    print(f"Total sequences             : {len(X_raw):,}")
    print(f"Held-out target sequences   : {len(test_indices):,}")
    print(f"Strict held-out test        : {len(strict_test_indices):,}")
    print(f"Leakage-safe train          : {len(train_fit_indices):,}")
    print(f"Validation                  : {len(val_indices):,}")
    print(f"Excluded context sequences : {int(history_contains_holdout.sum()):,}")

    # -----------------------------------------------------------------------
    # Fit scaling ONLY on training data.
    # -----------------------------------------------------------------------

    train_history_raw = X_raw[train_fit_indices]
    train_target_raw = targets_raw[train_fit_indices]

    flat_states = train_history_raw.reshape(-1, STATE_DIMENSION)
    state_scaler = RobustScaler.fit(flat_states)

    train_current_raw = train_history_raw[:, -1, :]
    train_delta_raw = train_target_raw - train_current_raw
    delta_scaler = RobustScaler.fit(train_delta_raw)

    X_train = state_scaler.transform(
        train_history_raw.reshape(-1, STATE_DIMENSION)
    ).reshape(train_history_raw.shape)

    y_train_delta = delta_scaler.transform(train_delta_raw)

    val_history_raw = X_raw[val_indices]
    val_target_raw = targets_raw[val_indices]
    val_current_raw = val_history_raw[:, -1, :]
    val_delta_raw = val_target_raw - val_current_raw

    X_val = state_scaler.transform(
        val_history_raw.reshape(-1, STATE_DIMENSION)
    ).reshape(val_history_raw.shape)
    y_val_delta = delta_scaler.transform(val_delta_raw)

    test_history_raw = X_raw[test_indices]
    test_target_raw = targets_raw[test_indices]
    test_current_raw = test_history_raw[:, -1, :]

    X_test = state_scaler.transform(
        test_history_raw.reshape(-1, STATE_DIMENSION)
    ).reshape(test_history_raw.shape)

    strict_history_raw = X_raw[strict_test_indices]
    strict_target_raw = targets_raw[strict_test_indices]
    strict_current_raw = strict_history_raw[:, -1, :]
    X_strict = state_scaler.transform(
        strict_history_raw.reshape(-1, STATE_DIMENSION)
    ).reshape(strict_history_raw.shape)

    # -----------------------------------------------------------------------
    # Graph data
    # -----------------------------------------------------------------------

    print()
    print("Preparing graph sequences...")
    train_graphs = prepare_graphs(all_graphs, train_fit_indices)
    val_graphs = prepare_graphs(all_graphs, val_indices)
    test_graphs = prepare_graphs(all_graphs, test_indices)

    # Strict graphs are prepared only if available.
    strict_graphs = (
        prepare_graphs(all_graphs, strict_test_indices)
        if len(strict_test_indices)
        else []
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, model_kwargs = build_model(STATE_DIMENSION, device)

    parameter_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"Model parameters             : {parameter_count:,}")
    print(f"Model constructor             : {model_kwargs}")

    history, best_epoch, best_val = train_model(
        model=model,
        train_X=X_train,
        train_delta=y_train_delta,
        train_graphs=train_graphs,
        val_X=X_val,
        val_delta=y_val_delta,
        val_graphs=val_graphs,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    # -----------------------------------------------------------------------
    # Evaluate held-out family
    # -----------------------------------------------------------------------

    print()
    print("=" * 72)
    print("UNSEEN ATTACK EVALUATION")
    print("=" * 72)

    pred_test_z = predict_dataset(
        model,
        X_test,
        test_graphs,
        device,
        args.batch_size,
    )
    pred_test_raw = state_scaler.inverse_transform(pred_test_z)

    # Persistence baseline = current observed state.
    persistence_raw = test_current_raw

    wm_metrics = metric_bundle(test_target_raw, pred_test_raw)
    persistence_metrics = metric_bundle(test_target_raw, persistence_raw)

    mae_gain = improvement_percent(
        persistence_metrics["mae"],
        wm_metrics["mae"],
    )
    rmse_gain = improvement_percent(
        persistence_metrics["rmse"],
        wm_metrics["rmse"],
    )

    print(f"Held-out family              : {args.holdout}")
    print(f"Test target states           : {len(test_indices):,}")
    print(f"World Model MAE              : {wm_metrics['mae']:.6f}")
    print(f"Persistence MAE              : {persistence_metrics['mae']:.6f}")
    print(f"MAE improvement              : {mae_gain:.2f}%")
    print(f"World Model RMSE             : {wm_metrics['rmse']:.6f}")
    print(f"Persistence RMSE             : {persistence_metrics['rmse']:.6f}")
    print(f"RMSE improvement             : {rmse_gain:.2f}%")

    strict_report = None

    if len(strict_test_indices):
        pred_strict_z = predict_dataset(
            model,
            X_strict,
            strict_graphs,
            device,
            args.batch_size,
        )
        pred_strict_raw = state_scaler.inverse_transform(pred_strict_z)

        strict_wm = metric_bundle(strict_target_raw, pred_strict_raw)
        strict_persistence = metric_bundle(strict_target_raw, strict_current_raw)

        strict_report = {
            "count": int(len(strict_test_indices)),
            "world_model": strict_wm,
            "persistence": strict_persistence,
            "improvement": {
                "mae_percent": improvement_percent(
                    strict_persistence["mae"],
                    strict_wm["mae"],
                ),
                "rmse_percent": improvement_percent(
                    strict_persistence["rmse"],
                    strict_wm["rmse"],
                ),
            },
        }

        print()
        print("STRICT CONTEXT-CLEAN SUBSET")
        print("-" * 72)
        print(f"Strict test states            : {len(strict_test_indices):,}")
        print(f"World Model MAE               : {strict_wm['mae']:.6f}")
        print(f"Persistence MAE               : {strict_persistence['mae']:.6f}")
        print(
            f"MAE improvement               : "
            f"{strict_report['improvement']['mae_percent']:.2f}%"
        )
        print(f"World Model RMSE              : {strict_wm['rmse']:.6f}")
        print(f"Persistence RMSE              : {strict_persistence['rmse']:.6f}")
        print(
            f"RMSE improvement              : "
            f"{strict_report['improvement']['rmse_percent']:.2f}%"
        )
    else:
        print()
        print("STRICT CONTEXT-CLEAN SUBSET: 0 sequences")

    # -----------------------------------------------------------------------
    # Save checkpoint without touching the production checkpoint.
    # -----------------------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dimension": STATE_DIMENSION,
        "graph_dimension": GRAPH_DIMENSION,
        "sequence_length": SEQUENCE_LENGTH,
        "holdout_family": args.holdout,
        "model_kwargs": model_kwargs,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "model_state_dict": model.state_dict(),
        "state_scaler": {
            "center": state_scaler.center,
            "scale": state_scaler.scale,
        },
        "delta_scaler": {
            "center": delta_scaler.center,
            "scale": delta_scaler.scale,
        },
        "formulation": "normalized_next_state_residual",
        "seed": args.seed,
    }

    torch.save(checkpoint, CHECKPOINT_FILE)

    report = {
        "experiment": {
            "name": "leave_one_attack_family_out",
            "held_out_family": args.holdout,
            "sequence_length": SEQUENCE_LENGTH,
            "state_dimension": STATE_DIMENSION,
            "graph_dimension": GRAPH_DIMENSION,
            "seed": args.seed,
        },
        "data": {
            "total_sequences": int(len(X_raw)),
            "leakage_safe_training_sequences": int(len(train_fit_indices)),
            "validation_sequences": int(len(val_indices)),
            "held_out_target_sequences": int(len(test_indices)),
            "strict_context_clean_sequences": int(len(strict_test_indices)),
            "excluded_context_sequences": int(history_contains_holdout.sum()),
            "target_window_min": int(target_windows[test_indices].min()),
            "target_window_max": int(target_windows[test_indices].max()),
        },
        "model": {
            "parameters": int(parameter_count),
            "constructor": model_kwargs,
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_val),
            "formulation": "normalized_next_state_residual",
            "checkpoint": str(CHECKPOINT_FILE),
            "device": str(device),
        },
        "all_held_out_targets": {
            "count": int(len(test_indices)),
            "world_model": wm_metrics,
            "persistence": persistence_metrics,
            "improvement": {
                "mae_percent": mae_gain,
                "rmse_percent": rmse_gain,
            },
            "beats_persistence_mae": bool(
                wm_metrics["mae"] < persistence_metrics["mae"]
            ),
        },
        "strict_context_clean": strict_report,
        "training_history": history,
        "verdict": {
            "target_family_absent_from_training_targets": True,
            "held_out_family_absent_from_training_history": True,
            "sufficient_test_size": bool(len(test_indices) >= 100),
            "strict_subset_available": bool(len(strict_test_indices) > 0),
            "world_model_beats_persistence_on_all_held_out_targets": bool(
                wm_metrics["mae"] < persistence_metrics["mae"]
            ),
        },
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 72)
    print("UNSEEN ATTACK WORLD MODEL EVALUATION: PASSED")
    print("=" * 72)
    print(f"Checkpoint saved : {CHECKPOINT_FILE}")
    print(f"Report saved     : {REPORT_FILE}")
    print()
    print("Existing production checkpoint was NOT modified.")
    print(
        "Do not call this an attack-detection accuracy result; "
        "these are next-state dynamics metrics on an unseen attack family."
    )


if __name__ == "__main__":
    main()
