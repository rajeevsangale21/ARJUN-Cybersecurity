import argparse

import numpy as np
import torch

from pipeline import run_pipeline

from preprocessing.normalizer import (
    NetworkStateNormalizer
)

from evaluation.dataset_split import (
    split_timeline
)

from forecasting.hybrid_k_step_forecaster import (
    HybridKStepForecaster
)

from forecasting.mitre_mapper import (
    MitreStageMapper
)

from risk.attack_progression import (
    AttackProgressionAnalyzer
)

from risk.risk_engine import (
    RiskEngine
)

from world_model.hybrid_wrapper import (
    HybridWorldModelWrapper
)

from world_model.state_encoder import (
    NetworkStateEncoder
)

from features.graph_builder import (
    build_window_graphs
)


def convert_graph(
    graph,
    device
):

    return {
        "node_features": torch.tensor(
            graph["node_features"],
            dtype=torch.float32,
            device=device
        ),

        "adjacency": torch.tensor(
            graph["adjacency"],
            dtype=torch.float32,
            device=device
        )
    }


def build_graph_sequence(
    graphs,
    window_ids,
    sequence_length,
    device
):

    if len(window_ids) < sequence_length:

        raise ValueError(
            "Not enough windows to build "
            "the graph sequence."
        )

    sequence_ids = window_ids[
        -sequence_length:
    ]

    graph_sequence = []

    for window_id in sequence_ids:

        window_id = int(
            window_id
        )

        if window_id not in graphs:

            raise ValueError(
                f"Missing graph for "
                f"window {window_id}"
            )

        graph_sequence.append(
            convert_graph(
                graphs[window_id],
                device
            )
        )

    return graph_sequence


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ARJUN K-step "
            "future-state forecasting"
        )
    )

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--model",
        default=(
            "saved_models/"
            "hybrid_world_model.pt"
        )
    )

    parser.add_argument(
        "--normalizer",
        default=(
            "saved_models/"
            "state_normalizer.pkl"
        )
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=10
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ARJUN K-STEP FORECAST")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load pipeline
    # --------------------------------------------------

    print("\n[1/7] Loading telemetry...")

    result = run_pipeline(
        args.input,
        sequence_length=args.sequence_length
    )

    states = result["states"]
    window_ids = result["window_ids"]
    df = result["data"]

    print(
        f"States: {states.shape}"
    )

    # --------------------------------------------------
    # 2. Build graphs
    # --------------------------------------------------

    print("\n[2/7] Building network graphs...")

    graphs = build_window_graphs(
        df
    )

    # --------------------------------------------------
    # 3. Load normalizer
    # --------------------------------------------------

    print("\n[3/7] Loading state normalizer...")

    normalizer = NetworkStateNormalizer()

    normalizer.load(
        args.normalizer
    )

    normalized_states = (
        normalizer.transform(
            states
        )
    )

    # --------------------------------------------------
    # 4. Load World Model
    # --------------------------------------------------

    print("\n[4/7] Loading World Model...")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = (
        HybridWorldModelWrapper.load(
            args.model,
            device=device
        )
    )

    # --------------------------------------------------
    # 5. Current sequence
    # --------------------------------------------------

    print("\n[5/7] Preparing current state...")

    if len(normalized_states) < (
        args.sequence_length
    ):

        raise ValueError(
            "Not enough network states for "
            "the requested sequence length."
        )

    current_sequence = (
        normalized_states[
            -args.sequence_length:
        ]
    )

    graph_sequence = (
        build_graph_sequence(
            graphs,
            window_ids,
            args.sequence_length,
            device
        )
    )

    # --------------------------------------------------
    # 6. K-step forecasting
    # --------------------------------------------------

    print(
        f"\n[6/7] Forecasting "
        f"{args.steps} future states..."
    )

    forecaster = HybridKStepForecaster(
        model=model,
        device=device
    )

    future_states = forecaster.forecast(
        current_sequence,
        graph_sequence,
        steps=args.steps
    )

    print(
        f"Generated states: "
        f"{future_states.shape}"
    )

    # --------------------------------------------------
    # 7. Risk + MITRE
    # --------------------------------------------------

    print(
        "\n[7/7] Calculating risk timeline..."
    )

    encoder = NetworkStateEncoder()

    progression = (
        AttackProgressionAnalyzer()
    )

    mitre = MitreStageMapper()

    risk_engine = RiskEngine()

    scores = progression.analyze(
        future_states,
        encoder.feature_names
    )

    mitre_results = mitre.map_sequence(
        future_states,
        encoder.feature_names
    )

    print("\n" + "=" * 70)
    print("FORECAST TIMELINE")
    print("=" * 70)

    for index, (
        state,
        score,
        mitre_result
    ) in enumerate(
        zip(
            future_states,
            scores,
            mitre_results
        ),
        start=1
    ):

        probability = (
            progression
            .probability_from_score(
                score
            )
        )

        risk = (
            risk_engine
            .calculate_risk_level(
                probability
            )
        )

        stage = mitre_result[
            "stage"
        ]

        confidence = mitre_result[
            "confidence"
        ]

        print(
            f"t+{index:<3} | "
            f"Attack Probability: "
            f"{probability:.3f} | "
            f"Risk: {risk} | "
            f"MITRE: {stage} "
            f"({confidence:.2f})"
        )

    print("=" * 70)

    print(
        "\nFuture-state forecasting complete."
    )


if __name__ == "__main__":
    main()