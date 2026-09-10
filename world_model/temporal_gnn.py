"""
ARJUN - Temporal Graph Neural Network

Purpose
-------
Encodes a network graph at one time step into a fixed-size graph
representation.

Input
-----
node_features:
    Shape (N, F)

adjacency:
    Shape (N, N)

Output
------
graph_embedding:
    Shape (output_dim,)

The implementation supports:
- single graphs
- variable number of nodes
- weighted adjacency matrices
- missing/empty graphs
- safe numerical handling
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple graph convolution layer.

    Formula:

        H' = ReLU(A_hat H W + b)

    where A_hat is the normalized adjacency matrix with
    self-loops.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be greater than zero.")

        if output_dim <= 0:
            raise ValueError("output_dim must be greater than zero.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def normalize_adjacency(
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add self-loops and perform symmetric normalization:

            A_hat = D^(-1/2) A D^(-1/2)

        This prevents nodes with many connections from
        dominating the representation.
        """

        if adjacency.ndim != 2:
            raise ValueError(
                "adjacency must have shape (N, N)."
            )

        n = adjacency.shape[0]

        if adjacency.shape[1] != n:
            raise ValueError(
                "adjacency must be square."
            )

        # Remove invalid numerical values.
        adjacency = torch.nan_to_num(
            adjacency,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Adjacency should not contain negative edge weights.
        adjacency = torch.clamp(adjacency, min=0.0)

        # Add self-loops.
        identity = torch.eye(
            n,
            device=adjacency.device,
            dtype=adjacency.dtype,
        )

        adjacency = adjacency + identity

        # Degree.
        degree = adjacency.sum(dim=1)

        # Avoid division by zero.
        degree = torch.clamp(degree, min=1e-12)

        degree_inv_sqrt = torch.pow(
            degree,
            -0.5,
        )

        normalized = (
            degree_inv_sqrt.unsqueeze(1)
            * adjacency
            * degree_inv_sqrt.unsqueeze(0)
        )

        return normalized

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        node_features:
            (N, input_dim)

        adjacency:
            (N, N)

        Returns
        -------
        torch.Tensor
            (N, output_dim)
        """

        if node_features.ndim != 2:
            raise ValueError(
                "node_features must have shape (N, F)."
            )

        if adjacency.ndim != 2:
            raise ValueError(
                "adjacency must have shape (N, N)."
            )

        if node_features.shape[0] != adjacency.shape[0]:
            raise ValueError(
                "Number of graph nodes does not match "
                "adjacency matrix."
            )

        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(
                "adjacency must be square."
            )

        node_features = torch.nan_to_num(
            node_features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        normalized_adjacency = self.normalize_adjacency(
            adjacency
        )

        transformed = self.linear(node_features)

        aggregated = torch.matmul(
            normalized_adjacency,
            transformed,
        )

        aggregated = F.relu(aggregated)

        aggregated = self.dropout(aggregated)

        return aggregated


class TemporalGNN(nn.Module):
    """
    Two-layer Graph Neural Network with global mean pooling.

    Architecture:

        Node Features
             |
             v
        GraphConv 1
             |
             v
        ReLU
             |
             v
        GraphConv 2
             |
             v
        ReLU
             |
             v
        Global Mean Pooling
             |
             v
        Graph Embedding

    The output is a fixed-size vector regardless of the
    number of nodes in the graph.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be greater than zero."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be greater than zero."
            )

        if output_dim <= 0:
            raise ValueError(
                "output_dim must be greater than zero."
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.gcn1 = GraphConvolution(
            input_dim=input_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )

        self.gcn2 = GraphConvolution(
            input_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode one graph.

        Parameters
        ----------
        node_features:
            Tensor of shape (N, input_dim)

        adjacency:
            Tensor of shape (N, N)

        Returns
        -------
        Tensor
            Shape (output_dim,)
        """

        if node_features.numel() == 0:
            return torch.zeros(
                self.output_dim,
                device=node_features.device,
                dtype=node_features.dtype,
            )

        x = self.gcn1(
            node_features,
            adjacency,
        )

        x = self.gcn2(
            x,
            adjacency,
        )

        # Global mean pooling.
        graph_embedding = torch.mean(
            x,
            dim=0,
        )

        graph_embedding = torch.nan_to_num(
            graph_embedding,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return graph_embedding

    def encode_graph(
        self,
        node_features,
        adjacency,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Convenience method for Python/numpy graph dictionaries.

        Parameters
        ----------
        node_features:
            NumPy array or torch Tensor.

        adjacency:
            NumPy array or torch Tensor.

        device:
            Optional torch device.

        Returns
        -------
        torch.Tensor
            Graph embedding.
        """

        if not isinstance(node_features, torch.Tensor):
            node_features = torch.tensor(
                node_features,
                dtype=torch.float32,
            )

        if not isinstance(adjacency, torch.Tensor):
            adjacency = torch.tensor(
                adjacency,
                dtype=torch.float32,
            )

        if device is not None:
            node_features = node_features.to(device)
            adjacency = adjacency.to(device)

        node_features = node_features.float()
        adjacency = adjacency.float()

        return self.forward(
            node_features,
            adjacency,
        )


def test_temporal_gnn() -> None:
    """
    Standalone self-test.

    This test creates a small graph with four nodes
    and verifies the output dimensions and numerical
    stability.
    """

    print("=" * 70)
    print("ARJUN TEMPORAL GNN TEST")
    print("=" * 70)

    torch.manual_seed(42)

    node_count = 4
    input_dim = 6
    hidden_dim = 16
    output_dim = 8

    # Example graph node features.
    node_features = torch.tensor(
        [
            [1.0, 2.0, 0.0, 4.0, 1.0, 0.0],
            [2.0, 1.0, 3.0, 2.0, 0.0, 1.0],
            [0.0, 1.0, 2.0, 3.0, 1.0, 0.0],
            [3.0, 0.0, 1.0, 1.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )

    # Example adjacency.
    adjacency = torch.tensor(
        [
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    model = TemporalGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )

    output = model(
        node_features,
        adjacency,
    )

    print(f"Node features : {tuple(node_features.shape)}")
    print(f"Adjacency     : {tuple(adjacency.shape)}")
    print(f"Output        : {tuple(output.shape)}")

    assert output.shape == (
        output_dim,
    ), (
        "Unexpected GNN output shape: "
        f"{tuple(output.shape)}"
    )

    assert torch.isfinite(output).all(), (
        "GNN output contains NaN or infinity."
    )

    print()
    print("Output finite : YES")
    print("Shape check   : PASSED")
    print("TEMPORAL GNN TEST: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_temporal_gnn()