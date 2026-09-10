"""
ARJUN Hybrid World Model Trainer

Training formulation:

    Input:
        X[t] = [S(t-9), ..., S(t)]

    Target:
        targets[t] = S(t+1)

The model learns the residual transition in normalized
state space:

    Z_t     = normalize(S_t)
    Z_t+1   = normalize(S_t+1)

    Delta_t = Z_t+1 - Z_t

    Delta_hat = WorldModel(X, G)

    Z_hat_t+1 = Z_t + Delta_hat
"""

from __future__ import annotations

import json
import os
import random
from copy import deepcopy
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


try:
    from .hybrid_world_model import HybridWorldModel
except ImportError:
    from hybrid_world_model import HybridWorldModel


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DATA_DIR = "data/processed"
MODEL_DIR = "saved_models"

STATE_FILE = os.path.join(
    DATA_DIR,
    "graph_sequences_states.npz",
)

GRAPH_FILE = os.path.join(
    DATA_DIR,
    "graph_sequences.pkl",
)

CHECKPOINT_FILE = os.path.join(
    MODEL_DIR,
    "hybrid_world_model.pt",
)

HISTORY_FILE = os.path.join(
    MODEL_DIR,
    "hybrid_world_model_history.json",
)

TRAIN_RATIO = 0.80

BATCH_SIZE = 16
EPOCHS = 30

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

PATIENCE = 7
MIN_DELTA = 1e-5

GRADIENT_CLIP = 1.0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed: int = SEED) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# STATE SCALER
# ============================================================

class StateScaler:
    """
    Robust feature-wise transformation.

    Step 1:
        signed log1p

    Step 2:
        robust centering using median

    Step 3:
        robust scaling using IQR
    """

    def __init__(self) -> None:

        self.median_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

        self.fitted = False

    @staticmethod
    def signed_log1p(
        x: np.ndarray,
    ) -> np.ndarray:

        return (
            np.sign(x)
            * np.log1p(np.abs(x))
        )

    def fit(
        self,
        x: np.ndarray,
    ) -> "StateScaler":

        x = np.asarray(
            x,
            dtype=np.float64,
        )

        if x.ndim != 2:
            raise ValueError(
                "StateScaler.fit expects a "
                f"2D array, got {x.shape}"
            )

        transformed = self.signed_log1p(x)

        median = np.median(
            transformed,
            axis=0,
        )

        q25 = np.percentile(
            transformed,
            25,
            axis=0,
        )

        q75 = np.percentile(
            transformed,
            75,
            axis=0,
        )

        iqr = q75 - q25

        std = np.std(
            transformed,
            axis=0,
        )

        scale = np.where(
            iqr > 1e-6,
            iqr / 1.349,
            std,
        )

        scale = np.where(
            scale > 1e-6,
            scale,
            1.0,
        )

        self.median_ = median
        self.scale_ = scale

        self.fitted = True

        return self

    def transform(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        if not self.fitted:
            raise RuntimeError(
                "StateScaler has not been fitted."
            )

        x = np.asarray(
            x,
            dtype=np.float64,
        )

        transformed = self.signed_log1p(x)

        result = (
            transformed
            - self.median_
        ) / self.scale_

        result = np.nan_to_num(
            result,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return result.astype(
            np.float32
        )

    def inverse_transform(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        if not self.fitted:
            raise RuntimeError(
                "StateScaler has not been fitted."
            )

        x = np.asarray(
            x,
            dtype=np.float64,
        )

        transformed = (
            x * self.scale_
            + self.median_
        )

        result = (
            np.sign(transformed)
            * np.expm1(
                np.abs(transformed)
            )
        )

        result = np.nan_to_num(
            result,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return result.astype(
            np.float32
        )

    def state_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "median": self.median_.tolist(),
            "scale": self.scale_.tolist(),
        }

    @classmethod
    def from_state_dict(
        cls,
        data: Dict[str, Any],
    ) -> "StateScaler":

        scaler = cls()

        scaler.median_ = np.asarray(
            data["median"],
            dtype=np.float64,
        )

        scaler.scale_ = np.asarray(
            data["scale"],
            dtype=np.float64,
        )

        scaler.fitted = True

        return scaler


# ============================================================
# DATASET
# ============================================================

class HybridTrainingDataset(
    Dataset
):
    """
    Dataset for next-state prediction.

    states:
        (N, T, D)

    targets:
        (N, D)

    graphs:
        N graph sequences
    """

    def __init__(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        graphs: Sequence,
        scaler: StateScaler,
    ) -> None:

        self.states_raw = np.asarray(
            states,
            dtype=np.float32,
        )

        self.targets_raw = np.asarray(
            targets,
            dtype=np.float32,
        )

        self.graphs = graphs

        if len(self.states_raw) != len(
            self.targets_raw
        ):
            raise ValueError(
                "State/target count mismatch."
            )

        if len(self.states_raw) != len(
            self.graphs
        ):
            raise ValueError(
                "State/graph count mismatch."
            )

        if self.states_raw.ndim != 3:
            raise ValueError(
                "Expected states with shape "
                "(N,T,D). Got "
                f"{self.states_raw.shape}"
            )

        if self.targets_raw.ndim != 2:
            raise ValueError(
                "Expected targets with shape "
                "(N,D). Got "
                f"{self.targets_raw.shape}"
            )

        self.state_dimension = (
            self.states_raw.shape[-1]
        )

        # ----------------------------------------------------
        # Normalize input sequence
        # ----------------------------------------------------

        flat_states = (
            self.states_raw.reshape(
                -1,
                self.state_dimension,
            )
        )

        scaled_states = scaler.transform(
            flat_states
        )

        self.states = scaled_states.reshape(
            self.states_raw.shape
        )

        # ----------------------------------------------------
        # Normalize actual next-state target
        # ----------------------------------------------------

        self.targets = scaler.transform(
            self.targets_raw
        )

        # ----------------------------------------------------
        # Final observed state
        # ----------------------------------------------------

        self.latest_state = (
            self.states[:, -1, :]
        )

        # ----------------------------------------------------
        # TRUE normalized transition target
        # ----------------------------------------------------

        self.target_delta = (
            self.targets
            - self.latest_state
        )

    def __len__(self) -> int:
        return len(
            self.states
        )

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:

        return {
            "states": torch.from_numpy(
                self.states[index]
            ),
            "latest_state": torch.from_numpy(
                self.latest_state[index]
            ),
            "target_state": torch.from_numpy(
                self.targets[index]
            ),
            "target_delta": torch.from_numpy(
                self.target_delta[index]
            ),
            "graphs": self.graphs[index],
        }


# ============================================================
# COLLATE
# ============================================================

def hybrid_collate(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:

    return {
        "states": torch.stack(
            [
                item["states"]
                for item in batch
            ]
        ),

        "latest_state": torch.stack(
            [
                item["latest_state"]
                for item in batch
            ]
        ),

        "target_state": torch.stack(
            [
                item["target_state"]
                for item in batch
            ]
        ),

        "target_delta": torch.stack(
            [
                item["target_delta"]
                for item in batch
            ]
        ),

        "graphs": [
            item["graphs"]
            for item in batch
        ],
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset() -> Tuple[
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
]:
    """
    Load X and the REAL next-state target.

    The sequence builder explicitly stores:

        X
        y
        states
        targets
        target_windows
        target_labels
    """

    if not os.path.exists(
        STATE_FILE
    ):
        raise FileNotFoundError(
            f"Missing state dataset:\n"
            f"{STATE_FILE}"
        )

    data = np.load(
        STATE_FILE,
        allow_pickle=True,
    )

    if "X" not in data:
        raise KeyError(
            "X is missing from dataset."
        )

    if "targets" not in data:
        raise KeyError(
            "targets is missing from dataset."
        )

    X = np.asarray(
        data["X"],
        dtype=np.float32,
    )

    targets = np.asarray(
        data["targets"],
        dtype=np.float32,
    )

    if X.shape[0] != targets.shape[0]:
        raise ValueError(
            "X and targets have different "
            f"sample counts: "
            f"{X.shape[0]} vs "
            f"{targets.shape[0]}"
        )

    metadata = {}

    for key in [
        "target_windows",
        "target_labels",
        "sequence_length",
        "state_dimension",
    ]:

        if key in data:
            metadata[key] = data[key]

    return (
        X,
        targets,
        metadata,
    )


def load_graph_sequences() -> Sequence:

    if not os.path.exists(
        GRAPH_FILE
    ):
        raise FileNotFoundError(
            f"Missing graph dataset:\n"
            f"{GRAPH_FILE}"
        )

    import pickle

    with open(
        GRAPH_FILE,
        "rb",
    ) as f:

        data = pickle.load(f)

    if isinstance(
        data,
        dict,
    ):

        for key in [
            "graph_sequences",
            "graphs",
            "sequences",
            "X_graph",
        ]:

            if key in data:
                return data[key]

    return data


# ============================================================
# MODEL PREDICTION
# ============================================================

def model_predict_delta(
    model: nn.Module,
    states: torch.Tensor,
    graphs: Sequence,
) -> torch.Tensor:

    if hasattr(
        model,
        "forward_batch",
    ):

        try:

            output = model.forward_batch(
                states,
                graphs,
            )

            if isinstance(
                output,
                tuple,
            ):
                output = output[0]

            return output

        except Exception:
            pass

    predictions = []

    for i in range(
        states.shape[0]
    ):

        output = model(
            states[i],
            graphs[i],
        )

        if isinstance(
            output,
            tuple,
        ):
            output = output[0]

        predictions.append(
            output
        )

    return torch.stack(
        predictions,
        dim=0,
    )


# ============================================================
# TRAIN
# ============================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:

    model.train()

    criterion = nn.SmoothL1Loss()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:

        states = batch[
            "states"
        ].to(DEVICE)

        target_delta = batch[
            "target_delta"
        ].to(DEVICE)

        graphs = batch[
            "graphs"
        ]

        optimizer.zero_grad(
            set_to_none=True
        )

        predicted_delta = (
            model_predict_delta(
                model,
                states,
                graphs,
            )
        )

        if predicted_delta.shape != (
            target_delta.shape
        ):
            raise RuntimeError(
                "Prediction/target shape "
                "mismatch:\n"
                f"Prediction: "
                f"{predicted_delta.shape}\n"
                f"Target: "
                f"{target_delta.shape}"
            )

        loss = criterion(
            predicted_delta,
            target_delta,
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "Training loss became "
                "NaN or Inf."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP,
        )

        optimizer.step()

        count = states.shape[0]

        total_loss += (
            loss.item()
            * count
        )

        total_samples += count

    return (
        total_loss
        / max(total_samples, 1)
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
) -> float:

    model.eval()

    criterion = nn.SmoothL1Loss()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:

        states = batch[
            "states"
        ].to(DEVICE)

        target_delta = batch[
            "target_delta"
        ].to(DEVICE)

        graphs = batch[
            "graphs"
        ]

        predicted_delta = (
            model_predict_delta(
                model,
                states,
                graphs,
            )
        )

        loss = criterion(
            predicted_delta,
            target_delta,
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "Validation loss became "
                "NaN or Inf."
            )

        count = states.shape[0]

        total_loss += (
            loss.item()
            * count
        )

        total_samples += count

    return (
        total_loss
        / max(total_samples, 1)
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model: nn.Module,
    scaler: StateScaler,
    history: Dict[str, Any],
    state_dimension: int,
    sequence_length: int,
) -> None:

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "state_dimension":
            state_dimension,

        "sequence_length":
            sequence_length,

        "graph_input_dimension":
            6,

        "state_scaler":
            scaler.state_dict(),

        "history":
            history,

        "formulation":
            "next_state_normalized_residual",

        "target_source":
            "graph_sequences_states.npz:targets",

        "description":
            (
                "ARJUN predicts the next "
                "network state using the "
                "previous state sequence "
                "and graph sequence."
            ),
    }

    torch.save(
        checkpoint,
        CHECKPOINT_FILE,
    )


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train() -> None:

    set_seed()

    print()
    print("=" * 72)
    print(
        "ARJUN HYBRID WORLD MODEL TRAINING"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    X, targets, metadata = (
        load_dataset()
    )

    graphs = load_graph_sequences()

    if len(X) != len(graphs):
        raise ValueError(
            "State/graph sequence count "
            "mismatch:\n"
            f"States : {len(X)}\n"
            f"Graphs : {len(graphs)}"
        )

    n = len(X)

    sequence_length = X.shape[1]

    state_dimension = X.shape[2]

    print(
        f"State dataset         : "
        f"{os.path.abspath(STATE_FILE)}"
    )

    print(
        f"Graph dataset         : "
        f"{os.path.abspath(GRAPH_FILE)}"
    )

    print(
        f"Device                : "
        f"{DEVICE}"
    )

    print()
    print(
        f"Sequences             : {n}"
    )

    print(
        f"Sequence length       : "
        f"{sequence_length}"
    )

    print(
        f"State dimension       : "
        f"{state_dimension}"
    )

    print(
        "Graph node dimension  : 6"
    )

    print()
    print(
        "Target source         : "
        "targets[]"
    )

    print(
        f"Target shape          : "
        f"{targets.shape}"
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    train_count = int(
        n * TRAIN_RATIO
    )

    X_train = X[
        :train_count
    ]

    X_val = X[
        train_count:
    ]

    targets_train = targets[
        :train_count
    ]

    targets_val = targets[
        train_count:
    ]

    graphs_train = graphs[
        :train_count
    ]

    graphs_val = graphs[
        train_count:
    ]

    print()
    print(
        f"Training sequences    : "
        f"{len(X_train)}"
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
    # SCALER
    # --------------------------------------------------------

    print()
    print(
        "Fitting training-only "
        "state scaler..."
    )

    scaler = StateScaler()

    # IMPORTANT:
    # Fit ONLY on training inputs and
    # training targets.
    #
    # This avoids validation leakage while
    # giving the scaler visibility into the
    # true range of next states.

    scaler_fit_data = np.concatenate(
        [
            X_train.reshape(
                -1,
                state_dimension,
            ),
            targets_train,
        ],
        axis=0,
    )

    scaler.fit(
        scaler_fit_data
    )

    print(
        "State scaler          : fitted"
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = (
        HybridTrainingDataset(
            X_train,
            targets_train,
            graphs_train,
            scaler,
        )
    )

    val_dataset = (
        HybridTrainingDataset(
            X_val,
            targets_val,
            graphs_val,
            scaler,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=hybrid_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=hybrid_collate,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = HybridWorldModel(
        state_dimension,
        6,
    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        f"Trainable parameters   : "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
        )
    )

    print()
    print("-" * 72)
    print(
        f"Epochs                : "
        f"{EPOCHS}"
    )

    print(
        f"Batch size            : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Learning rate         : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay          : "
        f"{WEIGHT_DECAY}"
    )

    print(
        "Objective             : "
        "TRUE next-state residual"
    )

    print(
        "Loss                  : "
        "SmoothL1"
    )

    print(
        "Scaler                : "
        "training-only"
    )

    print("-" * 72)

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    best_loss = float(
        "inf"
    )

    best_epoch = 0

    best_state = None

    patience_counter = 0

    history = {
        "train_loss": [],
        "validation_loss": [],
        "learning_rate": [],
    }

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
        )

        val_loss = validate_epoch(
            model,
            val_loader,
        )

        scheduler.step(
            val_loss
        )

        lr = optimizer.param_groups[
            0
        ]["lr"]

        history[
            "train_loss"
        ].append(
            float(train_loss)
        )

        history[
            "validation_loss"
        ].append(
            float(val_loss)
        )

        history[
            "learning_rate"
        ].append(
            float(lr)
        )

        is_best = (
            val_loss
            < best_loss - MIN_DELTA
        )

        marker = ""

        if is_best:

            best_loss = val_loss

            best_epoch = epoch

            best_state = deepcopy(
                model.state_dict()
            )

            patience_counter = 0

            marker = (
                " * BEST"
            )

        else:

            patience_counter += 1

        print(
            f"Epoch {epoch:02d} | "
            f"Train {train_loss:.6f} | "
            f"Val {val_loss:.6f} | "
            f"LR {lr:.6g}"
            f"{marker}"
        )

        if (
            patience_counter
            >= PATIENCE
        ):

            print()
            print(
                f"Early stopping at "
                f"epoch {epoch}."
            )

            break

    # --------------------------------------------------------
    # RESTORE BEST
    # --------------------------------------------------------

    if best_state is None:
        raise RuntimeError(
            "No valid model state was "
            "saved during training."
        )

    model.load_state_dict(
        best_state
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history[
        "best_epoch"
    ] = best_epoch

    history[
        "best_validation_loss"
    ] = float(best_loss)

    history[
        "state_dimension"
    ] = state_dimension

    history[
        "sequence_length"
    ] = sequence_length

    history[
        "train_sequences"
    ] = len(X_train)

    history[
        "validation_sequences"
    ] = len(X_val)

    history[
        "formulation"
    ] = (
        "next_state_normalized_residual"
    )

    history[
        "target_source"
    ] = (
        "graph_sequences_states.npz:targets"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_checkpoint(
        model,
        scaler,
        history,
        state_dimension,
        sequence_length,
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "TRAINING COMPLETE"
    )
    print("=" * 72)

    print(
        f"Best epoch            : "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss  : "
        f"{best_loss:.6f}"
    )

    print(
        f"Checkpoint            : "
        f"{os.path.abspath(CHECKPOINT_FILE)}"
    )

    print(
        f"History               : "
        f"{os.path.abspath(HISTORY_FILE)}"
    )

    print()
    print(
        "Target formulation:"
    )

    print(
        "  Input  = X[t] = "
        "[S(t-9) ... S(t)]"
    )

    print(
        "  Target = targets[t] = S(t+1)"
    )

    print(
        "  Delta  = Z(t+1) - Z(t)"
    )

    print(
        "  Output = Z(t) + Delta_hat"
    )

    print()
    print(
        "TRAINING STATUS: PASSED"
    )

    print("=" * 72)


# ============================================================
# SCALER TEST
# ============================================================

def run_scaler_test() -> None:

    print()
    print("=" * 72)
    print(
        "ARJUN TRAINER SCALER TEST"
    )
    print("=" * 72)

    rng = np.random.default_rng(
        SEED
    )

    x = rng.lognormal(
        mean=2.0,
        sigma=2.0,
        size=(100, 33),
    ).astype(
        np.float32
    )

    scaler = StateScaler()

    scaler.fit(x)

    transformed = scaler.transform(
        x
    )

    restored = (
        scaler.inverse_transform(
            transformed
        )
    )

    if not np.all(
        np.isfinite(
            transformed
        )
    ):
        raise RuntimeError(
            "Transformed values are "
            "not finite."
        )

    if not np.all(
        np.isfinite(
            restored
        )
    ):
        raise RuntimeError(
            "Restored values are "
            "not finite."
        )

    error = np.max(
        np.abs(
            restored - x
        )
    )

    print(
        f"Input shape            : "
        f"{x.shape}"
    )

    print(
        f"Transformed shape      : "
        f"{transformed.shape}"
    )

    print(
        f"Maximum reconstruction : "
        f"{error:.6f}"
    )

    if error > 1e-3:
        raise RuntimeError(
            "Scaler reconstruction "
            "error is too high."
        )

    print(
        "Scaler test             : "
        "PASSED"
    )

    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_scaler_test()

    train()