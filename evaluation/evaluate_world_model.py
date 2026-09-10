"""
ARJUN World Model Evaluation

Evaluates the trained Hybrid World Model against a persistence baseline.

IMPORTANT:
The sequence dataset explicitly contains:

    X       = observed state sequence [S(t-9) ... S(t)]
    targets = actual next state S(t+1)
    target_labels = label of S(t+1)

Therefore this evaluator compares:

    ARJUN:
        X -> World Model -> predicted S(t+1)

    Persistence:
        S(t+1) = S(t)

The checkpoint contains the StateScaler used during training.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


# ============================================================
# PATH SETUP
# ============================================================

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if ROOT_DIR not in sys.path:
    sys.path.insert(
        0,
        ROOT_DIR,
    )


from world_model.hybrid_world_model import (
    HybridWorldModel,
)

from world_model.hybrid_trainer import (
    StateScaler,
)


# ============================================================
# CONFIGURATION
# ============================================================

STATE_FILE = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "graph_sequences_states.npz",
)

GRAPH_FILE = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "graph_sequences.pkl",
)

CHECKPOINT_FILE = os.path.join(
    ROOT_DIR,
    "saved_models",
    "hybrid_world_model.pt",
)

REPORT_FILE = os.path.join(
    ROOT_DIR,
    "evaluation",
    "world_model_feature_evaluation.json",
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

TRAIN_RATIO = 0.80


# ============================================================
# FEATURE NAMES
# ============================================================

DEFAULT_FEATURE_NAMES = [
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


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Sequence,
    List[str],
]:
    """
    Load:

        X
        targets
        target_labels
        graphs
        feature names
    """

    if not os.path.exists(
        STATE_FILE
    ):
        raise FileNotFoundError(
            f"State dataset not found:\n"
            f"{STATE_FILE}"
        )

    if not os.path.exists(
        GRAPH_FILE
    ):
        raise FileNotFoundError(
            f"Graph dataset not found:\n"
            f"{GRAPH_FILE}"
        )

    data = np.load(
        STATE_FILE,
        allow_pickle=True,
    )

    required = [
        "X",
        "targets",
        "target_labels",
    ]

    for key in required:
        if key not in data:
            raise KeyError(
                f"Required dataset key missing: "
                f"{key}"
            )

    X = np.asarray(
        data["X"],
        dtype=np.float32,
    )

    targets = np.asarray(
        data["targets"],
        dtype=np.float32,
    )

    target_labels = np.asarray(
        data["target_labels"],
        dtype=np.int8,
    )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = list(
        DEFAULT_FEATURE_NAMES
    )

    if (
        "feature_names" in data
        and data["feature_names"].size > 0
    ):

        feature_names = [
            str(x)
            for x in data["feature_names"]
        ]

    if len(feature_names) != X.shape[-1]:

        print(
            "WARNING: feature name count "
            "does not match state dimension."
        )

        feature_names = [
            f"feature_{i}"
            for i in range(
                X.shape[-1]
            )
        ]

    # --------------------------------------------------------
    # Graphs
    # --------------------------------------------------------

    import pickle

    with open(
        GRAPH_FILE,
        "rb",
    ) as f:

        graph_data = pickle.load(f)

    if isinstance(
        graph_data,
        dict,
    ):

        for key in [
            "graph_sequences",
            "graphs",
            "sequences",
            "X_graph",
        ]:

            if key in graph_data:

                graphs = graph_data[
                    key
                ]

                break

        else:

            raise KeyError(
                "Could not find graph sequences "
                "inside graph pickle."
            )

    else:

        graphs = graph_data

    if len(X) != len(targets):
        raise ValueError(
            "X and targets have different "
            f"lengths: {len(X)} vs {len(targets)}"
        )

    if len(X) != len(target_labels):
        raise ValueError(
            "X and target_labels have different "
            f"lengths: {len(X)} vs "
            f"{len(target_labels)}"
        )

    if len(X) != len(graphs):
        raise ValueError(
            "X and graph sequences have different "
            f"lengths: {len(X)} vs {len(graphs)}"
        )

    return (
        X,
        targets,
        target_labels,
        graphs,
        feature_names,
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    state_dimension: int,
    graph_dimension: int,
) -> Tuple[
    torch.nn.Module,
    StateScaler,
]:
    """
    Load the trained checkpoint and its scaler.
    """

    if not os.path.exists(
        CHECKPOINT_FILE
    ):
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_FILE}"
        )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE,
        weights_only=False,
    )

    print()
    print(
        "Checkpoint configuration:"
    )

    print(
        "  Formulation           : "
        f"{checkpoint.get('formulation', 'unknown')}"
    )

    print(
        "  Target source         : "
        f"{checkpoint.get('target_source', 'unknown')}"
    )

    stored_dimension = checkpoint.get(
        "state_dimension"
    )

    if (
        stored_dimension is not None
        and int(stored_dimension)
        != state_dimension
    ):

        raise ValueError(
            "Checkpoint state dimension "
            "does not match dataset."
        )

    scaler_data = checkpoint.get(
        "state_scaler"
    )

    if scaler_data is None:
        raise KeyError(
            "Checkpoint does not contain "
            "the StateScaler."
        )

    scaler = StateScaler.from_state_dict(
        scaler_data
    )

    model = HybridWorldModel(
        state_dimension,
        graph_dimension,
    ).to(DEVICE)

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if state_dict is None:
        raise KeyError(
            "model_state_dict missing "
            "from checkpoint."
        )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return (
        model,
        scaler,
    )


# ============================================================
# SINGLE PREDICTION
# ============================================================

@torch.no_grad()
def predict_one(
    model: torch.nn.Module,
    scaler: StateScaler,
    state_sequence: np.ndarray,
    graph_sequence: Sequence,
) -> np.ndarray:
    """
    Predict next state.

    Model output is a normalized residual:

        Delta_hat

    Reconstruct:

        Z_hat(t+1)
            =
        Z(t) + Delta_hat

    Then inverse-transform to original scale.
    """

    scaled = scaler.transform(
        state_sequence
    )

    tensor = torch.from_numpy(
        scaled.astype(
            np.float32
        )
    ).to(DEVICE)

    output = model(
        tensor,
        graph_sequence,
    )

    if isinstance(
        output,
        tuple,
    ):
        output = output[0]

    predicted_delta = (
        output.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    latest_scaled = scaled[
        -1
    ]

    predicted_scaled = (
        latest_scaled
        + predicted_delta
    )

    predicted = (
        scaler.inverse_transform(
            predicted_scaled
        )
    )

    return predicted


# ============================================================
# METRICS
# ============================================================

def mae(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:

    return float(
        np.mean(
            np.abs(
                prediction
                - target
            )
        )
    )


def mse(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:

    return float(
        np.mean(
            (
                prediction
                - target
            )
            ** 2
        )
    )


def rmse(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:

    return float(
        np.sqrt(
            mse(
                prediction,
                target,
            )
        )
    )


def normalized_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    scaler: StateScaler,
) -> Dict[str, float]:

    pred_scaled = scaler.transform(
        prediction
    )

    target_scaled = scaler.transform(
        target
    )

    error = (
        pred_scaled
        - target_scaled
    )

    return {
        "mae": float(
            np.mean(
                np.abs(error)
            )
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        ),
    }


# ============================================================
# SAFE GAIN
# ============================================================

def percentage_gain(
    baseline: float,
    model_value: float,
) -> float:

    if abs(baseline) < 1e-12:
        return 0.0

    return (
        (
            baseline
            - model_value
        )
        / abs(baseline)
    ) * 100.0


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate() -> None:

    print()
    print("=" * 72)
    print(
        "ARJUN WORLD MODEL "
        "FEATURE-LEVEL EVALUATION"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    (
        X,
        targets,
        target_labels,
        graphs,
        feature_names,
    ) = load_data()

    n = len(X)

    state_dimension = X.shape[-1]

    sequence_length = X.shape[1]

    graph_dimension = 6

    print()
    print(
        f"State sequence shape : "
        f"{X.shape}"
    )

    print(
        f"Target shape         : "
        f"{targets.shape}"
    )

    print(
        f"Target labels        : "
        f"{target_labels.shape}"
    )

    print(
        f"Feature count        : "
        f"{state_dimension}"
    )

    print(
        f"Graph sequences      : "
        f"{len(graphs)}"
    )

    print(
        f"Graph node dimension : "
        f"{graph_dimension}"
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model, scaler = load_model(
        state_dimension,
        graph_dimension,
    )

    print()
    print(
        "Model constructor    : "
        "state_dimension, graph_input_dimension"
    )

    print(
        "Checkpoint            : loaded"
    )

    print(
        f"Device                : "
        f"{DEVICE}"
    )

    # --------------------------------------------------------
    # CHRONOLOGICAL VALIDATION
    # --------------------------------------------------------

    train_count = int(
        n * TRAIN_RATIO
    )

    X_val = X[
        train_count:
    ]

    targets_val = targets[
        train_count:
    ]

    labels_val = target_labels[
        train_count:
    ]

    graphs_val = graphs[
        train_count:
    ]

    print()
    print(
        f"Training sequences    : "
        f"{train_count}"
    )

    print(
        f"Validation sequences  : "
        f"{len(X_val)}"
    )

    print(
        "Split                 : "
        "chronological"
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    predictions = []

    persistence = []

    total = len(
        X_val
    )

    print()
    print(
        "Generating predictions..."
    )

    for i in range(
        total
    ):

        prediction = predict_one(
            model,
            scaler,
            X_val[i],
            graphs_val[i],
        )

        prediction = np.asarray(
            prediction,
            dtype=np.float32,
        )

        if prediction.shape != (
            state_dimension,
        ):

            raise RuntimeError(
                "Invalid prediction shape: "
                f"{prediction.shape}"
            )

        if not np.all(
            np.isfinite(
                prediction
            )
        ):

            raise RuntimeError(
                f"Non-finite prediction "
                f"at validation index {i}."
            )

        predictions.append(
            prediction
        )

        persistence.append(
            X_val[i, -1]
        )

        if (
            (i + 1) % 100 == 0
            or i + 1 == total
        ):

            print(
                f"  {i + 1}/{total}"
            )

    predictions = np.asarray(
        predictions,
        dtype=np.float32,
    )

    persistence = np.asarray(
        persistence,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    wm_mse = mse(
        predictions,
        targets_val,
    )

    wm_rmse = rmse(
        predictions,
        targets_val,
    )

    wm_mae = mae(
        predictions,
        targets_val,
    )

    base_mse = mse(
        persistence,
        targets_val,
    )

    base_rmse = rmse(
        persistence,
        targets_val,
    )

    base_mae = mae(
        persistence,
        targets_val,
    )

    rmse_gain = percentage_gain(
        base_rmse,
        wm_rmse,
    )

    mae_gain = percentage_gain(
        base_mae,
        wm_mae,
    )

    print()
    print("-" * 72)
    print(
        "OVERALL ORIGINAL-SCALE METRICS"
    )
    print("-" * 72)

    print(
        f"World model MSE      : "
        f"{wm_mse:.6e}"
    )

    print(
        f"World model RMSE     : "
        f"{wm_rmse:.6f}"
    )

    print(
        f"World model MAE      : "
        f"{wm_mae:.6f}"
    )

    print()

    print(
        f"Persistence MSE      : "
        f"{base_mse:.6e}"
    )

    print(
        f"Persistence RMSE     : "
        f"{base_rmse:.6f}"
    )

    print(
        f"Persistence MAE      : "
        f"{base_mae:.6f}"
    )

    print()

    print(
        f"RMSE improvement     : "
        f"{rmse_gain:.2f}%"
    )

    print(
        f"MAE improvement      : "
        f"{mae_gain:.2f}%"
    )

    # --------------------------------------------------------
    # NORMALIZED
    # --------------------------------------------------------

    wm_normalized = (
        normalized_metrics(
            predictions,
            targets_val,
            scaler,
        )
    )

    base_normalized = (
        normalized_metrics(
            persistence,
            targets_val,
            scaler,
        )
    )

    normalized_mae_gain = (
        percentage_gain(
            base_normalized["mae"],
            wm_normalized["mae"],
        )
    )

    normalized_rmse_gain = (
        percentage_gain(
            base_normalized["rmse"],
            wm_normalized["rmse"],
        )
    )

    print()
    print("-" * 72)
    print(
        "NORMALIZED METRICS"
    )
    print("-" * 72)

    print(
        f"World model MAE      : "
        f"{wm_normalized['mae']:.6f}"
    )

    print(
        f"Persistence MAE      : "
        f"{base_normalized['mae']:.6f}"
    )

    print(
        f"World model RMSE     : "
        f"{wm_normalized['rmse']:.6f}"
    )

    print(
        f"Persistence RMSE     : "
        f"{base_normalized['rmse']:.6f}"
    )

    print(
        f"Normalized MAE gain  : "
        f"{normalized_mae_gain:.2f}%"
    )

    print(
        f"Normalized RMSE gain : "
        f"{normalized_rmse_gain:.2f}%"
    )

    # --------------------------------------------------------
    # FEATURE LEVEL
    # --------------------------------------------------------

    print()
    print("-" * 105)
    print(
        "FEATURE-LEVEL PERFORMANCE"
    )
    print("-" * 105)

    print(
        f"{'Feature':<34}"
        f"{'WM MAE':>15}"
        f"{'Base MAE':>15}"
        f"{'Gain':>12}"
        f"{'Pred Std':>15}"
    )

    print("-" * 105)

    feature_results = []

    for j, name in enumerate(
        feature_names
    ):

        wm_error = np.mean(
            np.abs(
                predictions[:, j]
                - targets_val[:, j]
            )
        )

        base_error = np.mean(
            np.abs(
                persistence[:, j]
                - targets_val[:, j]
            )
        )

        gain = percentage_gain(
            float(base_error),
            float(wm_error),
        )

        pred_std = np.std(
            predictions[:, j]
        )

        result = {
            "feature": name,
            "world_model_mae": float(
                wm_error
            ),
            "persistence_mae": float(
                base_error
            ),
            "gain_percent": float(
                gain
            ),
            "prediction_std": float(
                pred_std
            ),
        }

        feature_results.append(
            result
        )

        print(
            f"{name:<34}"
            f"{wm_error:>15.4f}"
            f"{base_error:>15.4f}"
            f"{gain:>12.2f}%"
            f"{pred_std:>15.4f}"
        )

    # --------------------------------------------------------
    # PREDICTION DYNAMICS
    # --------------------------------------------------------

    prediction_delta = (
        predictions
        - persistence
    )

    actual_delta = (
        targets_val
        - persistence
    )

    mean_prediction_delta = float(
        np.mean(
            np.abs(
                prediction_delta
            )
        )
    )

    mean_actual_delta = float(
        np.mean(
            np.abs(
                actual_delta
            )
        )
    )

    prediction_delta_std = float(
        np.std(
            prediction_delta
        )
    )

    actual_delta_std = float(
        np.std(
            actual_delta
        )
    )

    near_constant_features = 0

    dynamic_threshold = 1e-5

    for j in range(
        state_dimension
    ):

        if np.std(
            prediction_delta[:, j]
        ) < dynamic_threshold:

            near_constant_features += 1

    # --------------------------------------------------------
    # SIGNIFICANT MOVEMENT
    # --------------------------------------------------------

    nonzero_prediction_fraction = float(
        np.mean(
            np.abs(
                prediction_delta
            ) > 1e-6
        )
    )

    print()
    print("-" * 72)
    print(
        "PREDICTION DYNAMICS DIAGNOSTIC"
    )
    print("-" * 72)

    print(
        f"Mean |predicted - current| : "
        f"{mean_prediction_delta:.6f}"
    )

    print(
        f"Mean |actual - current|    : "
        f"{mean_actual_delta:.6f}"
    )

    print(
        f"Prediction delta std       : "
        f"{prediction_delta_std:.6f}"
    )

    print(
        f"Actual delta std           : "
        f"{actual_delta_std:.6f}"
    )

    print(
        f"Non-zero prediction frac   : "
        f"{nonzero_prediction_fraction:.4f}"
    )

    print(
        f"Near-constant features     : "
        f"{near_constant_features}/"
        f"{state_dimension}"
    )

    # --------------------------------------------------------
    # ATTACK / BENIGN
    # --------------------------------------------------------

    attack_mask = (
        labels_val == 1
    )

    benign_mask = (
        labels_val == 0
    )

    attack_count = int(
        np.sum(
            attack_mask
        )
    )

    benign_count = int(
        np.sum(
            benign_mask
        )
    )

    print()
    print("-" * 72)
    print(
        "ATTACK / BENIGN TARGET ANALYSIS"
    )
    print("-" * 72)

    print(
        f"Attack target states : "
        f"{attack_count}"
    )

    print(
        f"Benign target states : "
        f"{benign_count}"
    )

    attack_metrics = {}

    benign_metrics = {}

    if attack_count > 0:

        attack_metrics = {
            "world_model_mae": mae(
                predictions[
                    attack_mask
                ],
                targets_val[
                    attack_mask
                ],
            ),
            "persistence_mae": mae(
                persistence[
                    attack_mask
                ],
                targets_val[
                    attack_mask
                ],
            ),
            "world_model_rmse": rmse(
                predictions[
                    attack_mask
                ],
                targets_val[
                    attack_mask
                ],
            ),
            "persistence_rmse": rmse(
                persistence[
                    attack_mask
                ],
                targets_val[
                    attack_mask
                ],
            ),
        }

        attack_metrics[
            "mae_gain_percent"
        ] = percentage_gain(
            attack_metrics[
                "persistence_mae"
            ],
            attack_metrics[
                "world_model_mae"
            ],
        )

        print(
            f"Attack WM MAE        : "
            f"{attack_metrics['world_model_mae']:.6f}"
        )

        print(
            f"Attack baseline MAE  : "
            f"{attack_metrics['persistence_mae']:.6f}"
        )

        print(
            f"Attack MAE gain      : "
            f"{attack_metrics['mae_gain_percent']:.2f}%"
        )

    if benign_count > 0:

        benign_metrics = {
            "world_model_mae": mae(
                predictions[
                    benign_mask
                ],
                targets_val[
                    benign_mask
                ],
            ),
            "persistence_mae": mae(
                persistence[
                    benign_mask
                ],
                targets_val[
                    benign_mask
                ],
            ),
            "world_model_rmse": rmse(
                predictions[
                    benign_mask
                ],
                targets_val[
                    benign_mask
                ],
            ),
            "persistence_rmse": rmse(
                persistence[
                    benign_mask
                ],
                targets_val[
                    benign_mask
                ],
            ),
        }

        benign_metrics[
            "mae_gain_percent"
        ] = percentage_gain(
            benign_metrics[
                "persistence_mae"
            ],
            benign_metrics[
                "world_model_mae"
            ],
        )

        print(
            f"Benign WM MAE       : "
            f"{benign_metrics['world_model_mae']:.6f}"
        )

        print(
            f"Benign baseline MAE : "
            f"{benign_metrics['persistence_mae']:.6f}"
        )

        print(
            f"Benign MAE gain     : "
            f"{benign_metrics['mae_gain_percent']:.2f}%"
        )

    # --------------------------------------------------------
    # BEST / WORST
    # --------------------------------------------------------

    sorted_features = sorted(
        feature_results,
        key=lambda x:
        x["gain_percent"],
        reverse=True,
    )

    print()
    print("-" * 72)
    print(
        "BEST WORLD-MODEL FEATURES"
    )
    print("-" * 72)

    for item in sorted_features[
        :10
    ]:

        print(
            f"{item['feature']:<32}"
            f"{item['gain_percent']:>10.2f}%"
        )

    print()
    print("-" * 72)
    print(
        "WORST WORLD-MODEL FEATURES"
    )
    print("-" * 72)

    for item in sorted_features[
        -10:
    ]:

        print(
            f"{item['feature']:<32}"
            f"{item['gain_percent']:>10.2f}%"
        )

    # --------------------------------------------------------
    # VERDICT
    # --------------------------------------------------------

    world_model_beats_persistence = (
        wm_mae < base_mae
    )

    print()
    print("-" * 72)
    print(
        "WORLD MODEL VERDICT"
    )
    print("-" * 72)

    if world_model_beats_persistence:

        print(
            "ARJUN World Model "
            "beats persistence on MAE."
        )

    else:

        print(
            "ARJUN World Model does NOT "
            "beat persistence on MAE."
        )

    print(
        "This result is reported honestly; "
        "no artificial benchmark adjustment "
        "is applied."
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = {

        "dataset": {
            "state_sequence_shape":
                list(X.shape),

            "target_shape":
                list(targets.shape),

            "validation_sequences":
                int(len(X_val)),

            "sequence_length":
                int(sequence_length),

            "state_dimension":
                int(state_dimension),
        },

        "model": {
            "checkpoint":
                CHECKPOINT_FILE,

            "formulation":
                "next_state_normalized_residual",

            "target_source":
                "targets[]",

            "device":
                str(DEVICE),
        },

        "overall": {

            "world_model": {
                "mse": wm_mse,
                "rmse": wm_rmse,
                "mae": wm_mae,
            },

            "persistence": {
                "mse": base_mse,
                "rmse": base_rmse,
                "mae": base_mae,
            },

            "improvement": {
                "mae_percent": mae_gain,
                "rmse_percent": rmse_gain,
            },
        },

        "normalized": {

            "world_model":
                wm_normalized,

            "persistence":
                base_normalized,

            "improvement": {
                "mae_percent":
                    normalized_mae_gain,

                "rmse_percent":
                    normalized_rmse_gain,
            },
        },

        "dynamics": {

            "mean_prediction_delta":
                mean_prediction_delta,

            "mean_actual_delta":
                mean_actual_delta,

            "prediction_delta_std":
                prediction_delta_std,

            "actual_delta_std":
                actual_delta_std,

            "nonzero_prediction_fraction":
                nonzero_prediction_fraction,

            "near_constant_features":
                near_constant_features,
        },

        "attack_benign": {
            "attack_count":
                attack_count,

            "benign_count":
                benign_count,

            "attack":
                attack_metrics,

            "benign":
                benign_metrics,
        },

        "features":
            feature_results,

        "verdict": {
            "beats_persistence_mae":
                bool(
                    world_model_beats_persistence
                ),
        },
    }

    os.makedirs(
        os.path.dirname(
            REPORT_FILE
        ),
        exist_ok=True,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print(
        "EVALUATION COMPLETE"
    )
    print("=" * 72)

    print(
        "Report saved:"
    )

    print(
        os.path.abspath(
            REPORT_FILE
        )
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    evaluate()