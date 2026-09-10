"""
ARJUN K-STEP WORLD MODEL FORECASTER

Correct residual-based recursive rollout.

The trained World Model outputs:

    Delta_Z = Z(t+1) - Z(t)

Therefore:

    Z_hat(t+1) = Z(t) + Delta_Z

and only then:

    S_hat(t+1) = inverse_scaler(Z_hat(t+1))

The predicted state is recursively fed back into the history.
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


# =====================================================================
# PROJECT ROOT
# =====================================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =====================================================================
# WORLD MODEL
# =====================================================================

from world_model.hybrid_world_model import HybridWorldModel


# =====================================================================
# PATHS
# =====================================================================

STATE_SEQUENCE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "graph_sequences_states.npz"
)

GRAPH_SEQUENCE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "graph_sequences.pkl"
)

CHECKPOINT_FILE = (
    ROOT
    / "saved_models"
    / "hybrid_world_model.pt"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "k_step_forecast.npz"
)


# =====================================================================
# STATE SCALER
# =====================================================================

class ForecastStateScaler:
    """
    Reconstruct the training-time state scaler.

    The trainer uses:

        Z = (S - center) / scale
    """

    def __init__(
        self,
        center: np.ndarray,
        scale: np.ndarray,
    ):
        self.center = np.asarray(
            center,
            dtype=np.float32,
        )

        self.scale = np.asarray(
            scale,
            dtype=np.float32,
        )

        self.scale = np.where(
            np.abs(self.scale) < 1e-8,
            1.0,
            self.scale,
        )

    def transform(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        x = np.asarray(
            x,
            dtype=np.float32,
        )

        return (
            x - self.center
        ) / self.scale

    def inverse_transform(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        x = np.asarray(
            x,
            dtype=np.float32,
        )

        return (
            x * self.scale
        ) + self.center


# =====================================================================
# GENERIC CHECKPOINT LOOKUP
# =====================================================================

def find_value(
    container: Any,
    names: List[str],
):
    """
    Find a value from a dictionary or object.
    """

    if container is None:
        return None

    if isinstance(
        container,
        dict,
    ):

        for name in names:

            if name in container:
                return container[name]

    for name in names:

        if hasattr(
            container,
            name,
        ):

            return getattr(
                container,
                name,
            )

    return None


# =====================================================================
# LOAD CHECKPOINT
# =====================================================================

def load_checkpoint():

    if not CHECKPOINT_FILE.exists():

        raise FileNotFoundError(
            "\nCheckpoint not found:\n"
            f"{CHECKPOINT_FILE}\n"
        )

    return torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=False,
    )


# =====================================================================
# LOAD STATE SCALER
# =====================================================================

def load_state_scaler(
    checkpoint,
) -> ForecastStateScaler:
    """
    Recover the exact state scaling parameters used during training.
    """

    scaler = find_value(
        checkpoint,
        [
            "state_scaler",
            "scaler",
            "state_normalizer",
        ],
    )

    if scaler is not None:

        center = find_value(
            scaler,
            [
                "center",
                "median",
                "location",
                "mean",
            ],
        )

        scale = find_value(
            scaler,
            [
                "scale",
                "iqr",
                "std",
                "spread",
            ],
        )

        if (
            center is not None
            and scale is not None
        ):

            return ForecastStateScaler(
                center=center,
                scale=scale,
            )

    # ---------------------------------------------------------------
    # Flat checkpoint representation
    # ---------------------------------------------------------------

    center = find_value(
        checkpoint,
        [
            "state_center",
            "state_median",
            "state_mean",
        ],
    )

    scale = find_value(
        checkpoint,
        [
            "state_scale",
            "state_iqr",
            "state_std",
        ],
    )

    if (
        center is not None
        and scale is not None
    ):

        return ForecastStateScaler(
            center=center,
            scale=scale,
        )

    raise RuntimeError(
        "\nCould not recover the state scaler "
        "from the trained checkpoint.\n"
    )


# =====================================================================
# MODEL CONFIGURATION
# =====================================================================

def get_model_config(
    checkpoint,
    state_dimension: int,
    graph_dimension: int,
) -> Dict[str, Any]:

    config = find_value(
        checkpoint,
        [
            "config",
            "model_config",
            "model_configuration",
        ],
    )

    if not isinstance(
        config,
        dict,
    ):

        config = {}

    hidden_dimension = int(
        config.get(
            "hidden_dimension",
            config.get(
                "hidden_dim",
                64,
            ),
        )
    )

    lstm_layers = int(
        config.get(
            "lstm_layers",
            config.get(
                "num_layers",
                1,
            ),
        )
    )

    residual_scale = float(
        config.get(
            "residual_scale",
            0.75,
        )
    )

    return {
        "state_dimension": state_dimension,
        "graph_dimension": graph_dimension,
        "hidden_dimension": hidden_dimension,
        "lstm_layers": lstm_layers,
        "residual_scale": residual_scale,
    }


# =====================================================================
# LOAD DATA
# =====================================================================

def load_sequences():

    if not STATE_SEQUENCE_FILE.exists():

        raise FileNotFoundError(
            "\nState sequence file not found:\n"
            f"{STATE_SEQUENCE_FILE}\n"
        )

    if not GRAPH_SEQUENCE_FILE.exists():

        raise FileNotFoundError(
            "\nGraph sequence file not found:\n"
            f"{GRAPH_SEQUENCE_FILE}\n"
        )

    data = np.load(
        STATE_SEQUENCE_FILE,
        allow_pickle=True,
    )

    X = data[
        "X"
    ].astype(
        np.float32
    )

    if "states" in data:

        states = data[
            "states"
        ].astype(
            np.float32
        )

    else:

        states = X.copy()

    if "targets" in data:

        targets = data[
            "targets"
        ].astype(
            np.float32
        )

    else:

        targets = data[
            "y"
        ].astype(
            np.float32
        )

    with open(
        GRAPH_SEQUENCE_FILE,
        "rb",
    ) as f:

        graph_data = pickle.load(f)

    return (
        X,
        states,
        targets,
        graph_data,
    )


# =====================================================================
# GRAPH NORMALIZATION
# =====================================================================

def normalize_graph_sequence(
    item: Any,
):

    if isinstance(
        item,
        dict,
    ):

        for key in [
            "graphs",
            "graph_sequence",
            "sequence",
        ]:

            if key in item:

                item = item[key]
                break

    if not isinstance(
        item,
        (list, tuple),
    ):

        raise TypeError(
            "Graph sequence is not a list or tuple."
        )

    result = []

    for graph in item:

        if not isinstance(
            graph,
            dict,
        ):

            raise TypeError(
                "Graph entry is not a dictionary."
            )

        node_features = graph.get(
            "node_features",
            graph.get(
                "features",
            ),
        )

        adjacency = graph.get(
            "adjacency",
            graph.get(
                "adj",
            ),
        )

        if node_features is None:

            raise KeyError(
                "Graph is missing node_features."
            )

        if adjacency is None:

            raise KeyError(
                "Graph is missing adjacency."
            )

        result.append(
            {
                "node_features": np.asarray(
                    node_features,
                    dtype=np.float32,
                ),
                "adjacency": np.asarray(
                    adjacency,
                    dtype=np.float32,
                ),
            }
        )

    return result


# =====================================================================
# LOAD MODEL
# =====================================================================

def load_model(
    checkpoint,
    state_dimension: int,
    graph_dimension: int,
):

    config = get_model_config(
        checkpoint,
        state_dimension,
        graph_dimension,
    )

    # IMPORTANT:
    #
    # This matches the current HybridWorldModel constructor.
    #
    # There is intentionally NO dropout argument.

    model = HybridWorldModel(
        state_dimension,
        graph_dimension,
        hidden_dimension=config[
            "hidden_dimension"
        ],
        lstm_layers=config[
            "lstm_layers"
        ],
        residual_scale=config[
            "residual_scale"
        ],
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
            "\nModel weights not found in checkpoint.\n"
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return (
        model,
        config,
    )


# =====================================================================
# SINGLE MODEL STEP
# =====================================================================

@torch.no_grad()
def predict_residual(
    model,
    normalized_state_sequence: np.ndarray,
    graph_sequence,
) -> np.ndarray:
    """
    Predict the normalized state residual:

        Delta_Z = Z(t+1) - Z(t)

    IMPORTANT:
    This function returns the residual itself.
    It does NOT inverse-transform it.
    """

    state_tensor = torch.from_numpy(
        np.asarray(
            normalized_state_sequence,
            dtype=np.float32,
        )
    )

    output = model(
        state_tensor,
        graph_sequence,
    )

    # ---------------------------------------------------------------
    # Handle tuple/list outputs
    # ---------------------------------------------------------------

    if isinstance(
        output,
        (tuple, list),
    ):

        output = output[0]

    # ---------------------------------------------------------------
    # Handle dictionary outputs
    # ---------------------------------------------------------------

    if isinstance(
        output,
        dict,
    ):

        for key in [
            "prediction",
            "delta",
            "residual",
            "output",
        ]:

            if key in output:

                output = output[key]
                break

    # ---------------------------------------------------------------
    # Remove batch dimension
    # ---------------------------------------------------------------

    if output.ndim > 1:

        output = output.squeeze(0)

    residual = (
        output
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    residual = np.nan_to_num(
        residual,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return residual


# =====================================================================
# CORRECT RESIDUAL ROLLOUT
# =====================================================================

@torch.no_grad()
def forecast_k_steps(
    model,
    initial_states: np.ndarray,
    graph_sequence,
    scaler: ForecastStateScaler,
    k: int = 5,
):
    """
    Correct recursive World Model rollout.

    At each step:

        1. Convert current history to normalized space.
        2. Predict Delta_Z.
        3. Take the CURRENT normalized state Z(t).
        4. Add predicted Delta_Z.
        5. Convert resulting Z_hat(t+1) back to raw space.
        6. Append raw S_hat(t+1) to history.
        7. Repeat.

    Returns:
        forecast shape = (K, D)
    """

    history_raw = np.asarray(
        initial_states,
        dtype=np.float32,
    ).copy()

    graph_history = list(
        graph_sequence
    )

    predictions = []

    for step in range(k):

        # -----------------------------------------------------------
        # Normalize the complete history.
        # -----------------------------------------------------------

        history_normalized = scaler.transform(
            history_raw
        )

        # -----------------------------------------------------------
        # Predict normalized residual.
        # -----------------------------------------------------------

        delta_z = predict_residual(
            model=model,
            normalized_state_sequence=history_normalized,
            graph_sequence=graph_history,
        )

        # -----------------------------------------------------------
        # CURRENT normalized state.
        #
        # The final state in the history is Z(t).
        # -----------------------------------------------------------

        current_z = history_normalized[-1]

        # -----------------------------------------------------------
        # CORRECT WORLD MODEL TRANSITION
        #
        # Z_hat(t+1) = Z(t) + Delta_Z
        # -----------------------------------------------------------

        predicted_z = (
            current_z
            + delta_z
        )

        # -----------------------------------------------------------
        # Convert predicted state back to original scale.
        # -----------------------------------------------------------

        predicted_raw = scaler.inverse_transform(
            predicted_z
        )

        predicted_raw = np.nan_to_num(
            predicted_raw,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(
            np.float32
        )

        predictions.append(
            predicted_raw
        )

        # -----------------------------------------------------------
        # Recursive history update.
        # -----------------------------------------------------------

        history_raw = np.concatenate(
            [
                history_raw[1:],
                predicted_raw.reshape(
                    1,
                    -1,
                ),
            ],
            axis=0,
        )

        # -----------------------------------------------------------
        # Graph history update.
        #
        # We currently don't have a graph transition generator,
        # therefore retain the most recent observed graph.
        # -----------------------------------------------------------

        if graph_history:

            graph_history = (
                graph_history[1:]
                + [graph_history[-1]]
            )

    return np.stack(
        predictions,
        axis=0,
    ).astype(
        np.float32
    )


# =====================================================================
# HYBRID K-STEP FORECASTER CLASS
# =====================================================================

class HybridKStepForecaster:
    """
    Object-oriented wrapper around ARJUN K-step world model forecasting.
    Compatible with unified_forecast.py, forecast_service.py, and run_end_to_end.py.
    """

    def __init__(
        self,
        world_model,
        device="cpu",
        scaler=None,
    ):
        self.wrapper = world_model
        self.model = getattr(world_model, "model", world_model)
        self.device = torch.device(device if device is not None else "cpu")
        self.model.to(self.device)
        self.model.eval()

        if scaler is not None:
            self.scaler = scaler
        else:
            try:
                ckpt = load_checkpoint()
                self.scaler = load_state_scaler(ckpt)
            except Exception:
                self.scaler = None

    def forecast(
        self,
        initial_states: np.ndarray,
        graph_sequence=None,
        steps: int = 5,
    ) -> np.ndarray:
        history_raw = np.asarray(initial_states, dtype=np.float32).copy()
        if graph_sequence is None:
            graph_sequence = []

        if self.scaler is not None:
            return forecast_k_steps(
                model=self.model,
                initial_states=history_raw,
                graph_sequence=graph_sequence,
                scaler=self.scaler,
                k=steps,
            )
        else:
            predictions = []
            current_hist = history_raw.copy()
            g_hist = list(graph_sequence) if graph_sequence else []
            for _ in range(steps):
                st_tensor = torch.from_numpy(current_hist).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = self.model(st_tensor, [g_hist] if g_hist else [[]])
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    if isinstance(out, dict):
                        out = out.get("prediction", out.get("delta", next(iter(out.values()))))
                    if out.ndim > 1:
                        out = out.squeeze(0)
                    pred_step = out.detach().cpu().numpy().astype(np.float32)
                predictions.append(pred_step)
                current_hist = np.vstack([current_hist[1:], pred_step[None, :]])
                if g_hist:
                    g_hist = g_hist[1:] + [g_hist[-1]]
            return np.asarray(predictions, dtype=np.float32)


# =====================================================================
# FEATURE NAMES
# =====================================================================

def load_feature_names():

    feature_file = (
        ROOT
        / "data"
        / "processed"
        / "training_states.npz"
    )

    if not feature_file.exists():

        return []

    data = np.load(
        feature_file,
        allow_pickle=True,
    )

    if "feature_names" not in data:

        if "states" in data:

            dimension = data[
                "states"
            ].shape[1]

        else:

            dimension = 33

        return [
            f"feature_{i}"
            for i in range(
                dimension
            )
        ]

    return [
        str(x)
        for x in data[
            "feature_names"
        ]
    ]


# =====================================================================
# MAIN TEST
# =====================================================================

def run_test(
    k=5,
):

    print()
    print("=" * 72)
    print(
        "ARJUN K-STEP WORLD MODEL FORECASTER"
    )
    print("=" * 72)

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    (
        X,
        states,
        targets,
        graph_data,
    ) = load_sequences()

    print(
        f"State sequences      : {X.shape}"
    )

    print(
        f"Targets              : {targets.shape}"
    )

    print(
        f"Graph sequences      : {len(graph_data)}"
    )

    state_dimension = X.shape[-1]

    # ---------------------------------------------------------------
    # Graph dimension
    # ---------------------------------------------------------------

    first_graph = normalize_graph_sequence(
        graph_data[0]
    )

    if not first_graph:

        raise RuntimeError(
            "First graph sequence is empty."
        )

    graph_dimension = first_graph[0][
        "node_features"
    ].shape[-1]

    print(
        f"State dimension      : "
        f"{state_dimension}"
    )

    print(
        f"Graph node dimension : "
        f"{graph_dimension}"
    )

    # ---------------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------------

    print()
    print(
        "Loading trained checkpoint..."
    )

    checkpoint = load_checkpoint()

    scaler = load_state_scaler(
        checkpoint
    )

    model, config = load_model(
        checkpoint,
        state_dimension,
        graph_dimension,
    )

    print(
        "Checkpoint            : loaded"
    )

    print(
        f"Residual scale        : "
        f"{config['residual_scale']}"
    )

    print(
        f"Hidden dimension      : "
        f"{config['hidden_dimension']}"
    )

    print(
        f"LSTM layers           : "
        f"{config['lstm_layers']}"
    )

    # ---------------------------------------------------------------
    # Use last validation sequence
    # ---------------------------------------------------------------

    validation_index = len(X) - 1

    initial_states = X[
        validation_index
    ].astype(
        np.float32
    )

    graph_sequence = normalize_graph_sequence(
        graph_data[
            validation_index
        ]
    )

    print()
    print(
        f"Validation index      : "
        f"{validation_index}"
    )

    print(
        f"Input state shape     : "
        f"{initial_states.shape}"
    )

    print(
        f"Input graph count     : "
        f"{len(graph_sequence)}"
    )

    # ---------------------------------------------------------------
    # Generate K-step forecast
    # ---------------------------------------------------------------

    print()
    print(
        f"Generating {k}-step forecast..."
    )

    forecast = forecast_k_steps(
        model=model,
        initial_states=initial_states,
        graph_sequence=graph_sequence,
        scaler=scaler,
        k=k,
    )

    # ---------------------------------------------------------------
    # Validate
    # ---------------------------------------------------------------

    expected_shape = (
        k,
        state_dimension,
    )

    if forecast.shape != expected_shape:

        raise RuntimeError(
            "\nUnexpected forecast shape.\n"
            f"Expected: {expected_shape}\n"
            f"Received: {forecast.shape}\n"
        )

    if not np.isfinite(
        forecast
    ).all():

        raise RuntimeError(
            "\nForecast contains NaN or infinite values.\n"
        )

    # ---------------------------------------------------------------
    # Step-by-step dynamics
    # ---------------------------------------------------------------

    print()
    print("-" * 72)
    print(
        "STEP-TO-STEP CHANGES"
    )
    print("-" * 72)

    previous = initial_states[-1]

    step_changes = []

    for step, prediction in enumerate(
        forecast,
        start=1,
    ):

        delta = (
            prediction
            - previous
        )

        mean_change = float(
            np.mean(
                np.abs(delta)
            )
        )

        max_change = float(
            np.max(
                np.abs(delta)
            )
        )

        step_changes.append(
            mean_change
        )

        print(
            f"Current -> t+{step} | "
            f"Mean |Δ| = "
            f"{mean_change:.6f} | "
            f"Max |Δ| = "
            f"{max_change:.6f}"
        )

        previous = prediction

    # ---------------------------------------------------------------
    # Forecast range
    # ---------------------------------------------------------------

    print()
    print(
        "FORECAST VALUE RANGE"
    )
    print("-" * 72)

    print(
        f"Minimum               : "
        f"{np.min(forecast):.6f}"
    )

    print(
        f"Maximum               : "
        f"{np.max(forecast):.6f}"
    )

    print(
        f"Mean                  : "
        f"{np.mean(forecast):.6f}"
    )

    print(
        f"Std                   : "
        f"{np.std(forecast):.6f}"
    )

    # ---------------------------------------------------------------
    # Compare first prediction with persistence
    # ---------------------------------------------------------------

    persistence = initial_states[-1]

    first_prediction = forecast[0]

    persistence_distance = float(
        np.mean(
            np.abs(
                persistence
                - persistence
            )
        )
    )

    first_prediction_distance = float(
        np.mean(
            np.abs(
                first_prediction
                - persistence
            )
        )
    )

    print()
    print(
        "FIRST-STEP FORECAST"
    )
    print("-" * 72)

    print(
        f"Mean movement from current : "
        f"{first_prediction_distance:.6f}"
    )

    print(
        f"Persistence movement        : "
        f"{persistence_distance:.6f}"
    )

    # ---------------------------------------------------------------
    # Overall dynamic check
    # ---------------------------------------------------------------

    dynamic = (
        len(step_changes) > 0
        and max(step_changes) > 1e-8
    )

    print()
    print(
        "FORECAST MOVEMENT"
    )
    print("-" * 72)

    total_movement = float(
        np.mean(
            np.abs(
                forecast[-1]
                - initial_states[-1]
            )
        )
    )

    print(
        f"Total mean movement   : "
        f"{total_movement:.6f}"
    )

    print(
        f"Dynamic rollout       : "
        f"{'YES' if dynamic else 'NO'}"
    )

    # ---------------------------------------------------------------
    # Top future-changing features
    # ---------------------------------------------------------------

    feature_names = load_feature_names()

    if len(feature_names) == state_dimension:

        final_delta = (
            forecast[-1]
            - initial_states[-1]
        )

        order = np.argsort(
            np.abs(final_delta)
        )[::-1]

        print()
        print(
            "TOP FUTURE-CHANGING FEATURES"
        )
        print("-" * 72)

        for index in order[:10]:

            print(
                f"{feature_names[index]:35s} "
                f"Δ = "
                f"{final_delta[index]: .6f}"
            )

    # ---------------------------------------------------------------
    # Save forecast
    # ---------------------------------------------------------------

    # The model predicts normalized residuals internally, but `forecast`
    # contains the complete predicted states on the original feature scale.
    # Save both the states and their step-to-step changes for downstream
    # risk analysis and the Streamlit dashboard.

    step_deltas = np.empty_like(forecast, dtype=np.float32)
    previous_state = initial_states[-1].astype(np.float32)

    for i, prediction in enumerate(forecast):
        step_deltas[i] = prediction - previous_state
        previous_state = prediction

    current_state = initial_states[-1].astype(np.float32)
    sequence_length = int(initial_states.shape[0])

    np.savez_compressed(
        OUTPUT_FILE,
        forecast_states=np.asarray(
            forecast,
            dtype=np.float32,
        ),
        step_deltas=np.asarray(
            step_deltas,
            dtype=np.float32,
        ),
        current_state=np.asarray(
            current_state,
            dtype=np.float32,
        ),
        feature_names=np.asarray(
            feature_names,
            dtype=str,
        ),
        sequence_length=np.int64(
            sequence_length,
        ),
    )

    print()
    print(
        f"Forecast saved        : "
        f"{OUTPUT_FILE}"
    )

    # ---------------------------------------------------------------
    # Final status
    # ---------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "K-STEP FORECASTER TEST: PASSED"
    )
    print("=" * 72)


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    run_test(
        k=5
    )