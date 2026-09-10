"""
ARJUN Hybrid World Model

Stable residual World Model.

Input:
    network state sequence S(t-k+1)...S(t)
    graph sequence G(t-k+1)...G(t)

Output:
    normalized residual Delta

Next-state reconstruction:

    Z_hat(t+1) = Z(t) + alpha * tanh(raw_delta)

The bounded residual prevents unstable state explosions.
"""

from __future__ import annotations

from typing import Any, List, Sequence

import torch
import torch.nn as nn


try:
    from .temporal_gnn import TemporalGNN
except ImportError:
    from temporal_gnn import TemporalGNN


# ============================================================
# GRAPH ENCODER
# ============================================================

class GraphEncoder(nn.Module):
    """
    Encodes one network graph into a fixed-size vector.
    """

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int = 32,
    ) -> None:

        super().__init__()

        self.gnn = TemporalGNN(
    input_dimension,
    hidden_dimension,
    hidden_dimension,
)

        self.output_dimension = (
            hidden_dimension
        )

    def forward(
        self,
        graph: Any,
    ) -> torch.Tensor:

        if isinstance(
            graph,
            dict,
        ):

            node_features = graph.get(
                "node_features"
            )

            adjacency = graph.get(
                "adjacency"
            )

        else:

            raise TypeError(
                "Graph must be a dictionary."
            )

        if node_features is None:
            raise ValueError(
                "Graph missing node_features."
            )

        if adjacency is None:
            raise ValueError(
                "Graph missing adjacency."
            )

        x = torch.as_tensor(
            node_features,
            dtype=torch.float32,
        )

        a = torch.as_tensor(
            adjacency,
            dtype=torch.float32,
        )

        return self.gnn(
            x,
            a,
        )


# ============================================================
# HYBRID WORLD MODEL
# ============================================================

class HybridWorldModel(nn.Module):
    """
    Hybrid network-state + graph temporal model.

    State sequence:
        (T,D)

    Graph sequence:
        T graph objects

    Output:
        bounded normalized residual (D,)
    """

    def __init__(
        self,
        state_dimension: int,
        graph_input_dimension: int,
        hidden_dimension: int = 64,
        lstm_layers: int = 1,
        residual_scale: float = 0.75,
    ) -> None:

        super().__init__()

        self.state_dimension = (
            state_dimension
        )

        self.graph_input_dimension = (
            graph_input_dimension
        )

        self.hidden_dimension = (
            hidden_dimension
        )

        self.lstm_layers = (
            lstm_layers
        )

        self.residual_scale = (
            float(residual_scale)
        )

        # ----------------------------------------------------
        # GRAPH
        # ----------------------------------------------------

        self.graph_encoder = (
            GraphEncoder(
                graph_input_dimension,
                hidden_dimension,
            )
        )

        # ----------------------------------------------------
        # STATE PROJECTION
        # ----------------------------------------------------

        self.state_projection = nn.Sequential(

            nn.Linear(
                state_dimension,
                hidden_dimension,
            ),

            nn.LayerNorm(
                hidden_dimension
            ),

            nn.GELU(),
        )

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        fusion_dimension = (
            hidden_dimension * 2
        )

        self.fusion = nn.Sequential(

            nn.Linear(
                fusion_dimension,
                hidden_dimension,
            ),

            nn.LayerNorm(
                hidden_dimension
            ),

            nn.GELU(),
        )

        # ----------------------------------------------------
        # TEMPORAL MODEL
        # ----------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=hidden_dimension,
            hidden_size=hidden_dimension,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        self.output_head = nn.Sequential(

            nn.Linear(
                hidden_dimension,
                hidden_dimension,
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dimension,
                state_dimension,
            ),
        )

    # ========================================================
    # SINGLE SEQUENCE
    # ========================================================

    def forward(
        self,
        state_sequence: torch.Tensor,
        graph_sequence: Sequence[Any],
    ) -> torch.Tensor:

        if state_sequence.ndim != 2:

            raise ValueError(
                "state_sequence must have "
                "shape (T,D). Got "
                f"{state_sequence.shape}"
            )

        time_steps = (
            state_sequence.shape[0]
        )

        if len(graph_sequence) != time_steps:

            raise ValueError(
                "State/graph sequence length "
                "mismatch: "
                f"{time_steps} vs "
                f"{len(graph_sequence)}"
            )

        device = (
            state_sequence.device
        )

        sequence_features = []

        for t in range(
            time_steps
        ):

            state = (
                state_sequence[t]
                .unsqueeze(0)
            )

            state_embedding = (
                self.state_projection(
                    state
                )
            )

            graph_embedding = (
                self.graph_encoder(
                    graph_sequence[t]
                )
            )

            graph_embedding = (
                graph_embedding
                .to(device)
                .unsqueeze(0)
            )

            fused = torch.cat(
                [
                    state_embedding,
                    graph_embedding,
                ],
                dim=-1,
            )

            fused = self.fusion(
                fused
            )

            sequence_features.append(
                fused
            )

        temporal_input = torch.cat(
            sequence_features,
            dim=0,
        )

        temporal_input = (
            temporal_input.unsqueeze(0)
        )

        lstm_output, _ = self.lstm(
            temporal_input
        )

        final_embedding = (
            lstm_output[:, -1, :]
        )

        raw_delta = (
            self.output_head(
                final_embedding
            )
        )

        # ----------------------------------------------------
        # BOUNDED RESIDUAL
        # ----------------------------------------------------

        delta = (
            self.residual_scale
            * torch.tanh(
                raw_delta
            )
        )

        return delta.squeeze(0)

    # ========================================================
    # BATCH
    # ========================================================

    def forward_batch(
        self,
        state_sequences: torch.Tensor,
        graph_sequences: Sequence[Any],
    ) -> torch.Tensor:

        if state_sequences.ndim != 3:

            raise ValueError(
                "state_sequences must have "
                "shape (B,T,D). Got "
                f"{state_sequences.shape}"
            )

        batch_size = (
            state_sequences.shape[0]
        )

        if len(graph_sequences) != batch_size:

            raise ValueError(
                "Batch state/graph count "
                "mismatch."
            )

        predictions = []

        for i in range(
            batch_size
        ):

            prediction = self.forward(
                state_sequences[i],
                graph_sequences[i],
            )

            predictions.append(
                prediction
            )

        return torch.stack(
            predictions,
            dim=0,
        )


# ============================================================
# TEST
# ============================================================

def run_test() -> None:

    print()
    print("=" * 72)
    print(
        "ARJUN HYBRID WORLD MODEL TEST"
    )
    print("=" * 72)

    import numpy as np

    torch.manual_seed(
        42
    )

    np.random.seed(
        42
    )

    state_dimension = 33

    graph_dimension = 6

    sequence_length = 10

    model = HybridWorldModel(
        state_dimension,
        graph_dimension,
    )

    states = torch.randn(
        sequence_length,
        state_dimension,
    )

    graphs = []

    for nodes in [
        4,
        5,
        6,
        7,
        8,
        4,
        5,
        6,
        7,
        8,
    ]:

        node_features = (
            np.random.randn(
                nodes,
                graph_dimension,
            ).astype(
                np.float32
            )
        )

        adjacency = (
            np.random.randint(
                0,
                2,
                size=(
                    nodes,
                    nodes,
                ),
            ).astype(
                np.float32
            )
        )

        adjacency = np.maximum(
            adjacency,
            adjacency.T,
        )

        np.fill_diagonal(
            adjacency,
            0,
        )

        graphs.append(
            {
                "node_features":
                    node_features,
                "adjacency":
                    adjacency,
                "nodes":
                    list(
                        range(nodes)
                    ),
            }
        )

    prediction = model(
        states,
        graphs,
    )

    batch_states = torch.stack(
        [
            states,
            states + 0.1,
        ]
    )

    batch_graphs = [
        graphs,
        graphs,
    ]

    batch_prediction = (
        model.forward_batch(
            batch_states,
            batch_graphs,
        )
    )

    print(
        f"State sequence       : "
        f"{tuple(states.shape)}"
    )

    print(
        f"Prediction shape     : "
        f"{tuple(prediction.shape)}"
    )

    print(
        f"Prediction finite    : "
        f"{bool(torch.isfinite(prediction).all())}"
    )

    print(
        f"Prediction min       : "
        f"{prediction.min().item():.6f}"
    )

    print(
        f"Prediction max       : "
        f"{prediction.max().item():.6f}"
    )

    print(
        f"Batch prediction     : "
        f"{tuple(batch_prediction.shape)}"
    )

    if prediction.shape != (
        state_dimension,
    ):
        raise RuntimeError(
            "Invalid prediction shape."
        )

    if not torch.isfinite(
        prediction
    ).all():
        raise RuntimeError(
            "Prediction contains NaN/Inf."
        )

    if (
        prediction.min().item()
        < -0.750001
        or prediction.max().item()
        > 0.750001
    ):
        raise RuntimeError(
            "Residual bound violated."
        )

    print()
    print(
        "Residual bound       : PASSED"
    )

    print(
        "Graph encoding       : PASSED"
    )

    print(
        "Variable graphs      : PASSED"
    )

    print(
        "LSTM                 : PASSED"
    )

    print(
        "Batch inference      : PASSED"
    )

    print(
        "Finite output        : PASSED"
    )

    print(
        "HYBRID WORLD MODEL TEST: PASSED"
    )

    print("=" * 72)


if __name__ == "__main__":
    run_test()