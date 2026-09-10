import argparse
from pathlib import Path

import numpy as np

from pipeline import run_pipeline

from world_model.hybrid_wrapper import (
    HybridWorldModelWrapper,
)

from forecasting.hybrid_k_step_forecaster import (
    HybridKStepForecaster,
)

from risk.unified_analysis import (
    UnifiedAnalysisService,
)


def load_world_model(
    model_path,
    state_dimension,
):
    """
    Load the trained ARJUN Hybrid World Model.
    """

    model_path = Path(
        model_path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"World Model not found: {model_path}"
        )

    return (
        HybridWorldModelWrapper.from_checkpoint(
            model_path,
            state_dimension=state_dimension,
            device="cpu",
        )
    )


def calculate_attack_probability(
    state,
    feature_names,
):
    """
    Lightweight behavioural attack probability.

    This is used when a separately trained attack
    classifier is unavailable.

    It is intentionally bounded between 0 and 1.
    """

    state = np.asarray(
        state,
        dtype=np.float32,
    )

    values = dict(
        zip(
            feature_names,
            state,
        )
    )

    def get(name):
        return float(
            values.get(
                name,
                0.0,
            )
        )

    # -----------------------------------------------------
    # Suspicious behavioural indicators
    # -----------------------------------------------------

    port_scan = get(
        "port_scan_count"
    )

    retransmissions = get(
        "retransmission_count"
    )

    rst_count = get(
        "rst_count"
    )

    unique_ports = max(
        get("unique_dst_ports"),
        get("avg_unique_dst_ports"),
    )

    unique_hosts = max(
        get("unique_dst_ips"),
        get("avg_unique_dst_hosts"),
    )

    syn_count = get(
        "syn_count"
    )

    bytes_per_second = get(
        "avg_bytes_per_second"
    )

    # -----------------------------------------------------
    # Normalized components
    # -----------------------------------------------------

    scan_score = min(
        port_scan / 10.0,
        1.0,
    )

    port_score = min(
        unique_ports / 50.0,
        1.0,
    )

    host_score = min(
        unique_hosts / 20.0,
        1.0,
    )

    syn_score = min(
        syn_count / 100.0,
        1.0,
    )

    retransmission_score = min(
        retransmissions / 50.0,
        1.0,
    )

    reset_score = min(
        rst_count / 50.0,
        1.0,
    )

    traffic_score = min(
        bytes_per_second / 1_000_000.0,
        1.0,
    )

    probability = (
        0.25 * scan_score
        + 0.15 * port_score
        + 0.15 * host_score
        + 0.15 * syn_score
        + 0.10 * retransmission_score
        + 0.10 * reset_score
        + 0.10 * traffic_score
    )

    return float(
        np.clip(
            probability,
            0.0,
            1.0,
        )
    )


def forecast_file(
    input_path,
    model_path="saved_models/hybrid_world_model.pt",
    sequence_length=5,
    steps=10,
    window_seconds=5,
):
    """
    Complete ARJUN forecasting pipeline.

    Returns a single dictionary that can be consumed
    directly by Streamlit or another application.
    """

    # =====================================================
    # 1. PROCESS INPUT
    # =====================================================

    pipeline_result = run_pipeline(
        input_path,
        window_seconds=window_seconds,
        sequence_length=sequence_length,
    )

    states = np.asarray(
        pipeline_result["states"],
        dtype=np.float32,
    )

    graph_sequences = (
        pipeline_result["graph_sequences"]
    )

    feature_names = list(
        pipeline_result["feature_names"]
    )

    effective_sequence_length = int(
        pipeline_result["sequence_length"]
    )

    if len(states) < (
        effective_sequence_length
    ):

        raise ValueError(
            "Not enough network states "
            "for forecasting."
        )

    if not graph_sequences:

        raise ValueError(
            "No graph sequences were generated."
        )

    state_dimension = int(
        states.shape[1]
    )

    # =====================================================
    # 2. LOAD WORLD MODEL
    # =====================================================

    world_model = load_world_model(
        model_path,
        state_dimension,
    )

    # =====================================================
    # 3. CURRENT STATE
    # =====================================================

    current_state = states[-1]

    current_sequence = states[
        -effective_sequence_length:
    ]

    current_graph_sequence = (
        graph_sequences[-1]
    )

    # =====================================================
    # 4. CURRENT ATTACK PROBABILITY
    # =====================================================

    current_probability = (
        calculate_attack_probability(
            current_state,
            feature_names,
        )
    )

    # =====================================================
    # 5. K-STEP WORLD MODEL FORECAST
    # =====================================================

    forecaster = (
        HybridKStepForecaster(
            world_model,
            device="cpu",
        )
    )

    future_states = (
        forecaster.forecast(
            current_sequence,
            current_graph_sequence,
            steps=steps,
        )
    )

    future_states = np.asarray(
        future_states,
        dtype=np.float32,
    )

    # =====================================================
    # 6. FUTURE ATTACK PROBABILITIES
    # =====================================================

    future_probabilities = []

    for future_state in future_states:

        probability = (
            calculate_attack_probability(
                future_state,
                feature_names,
            )
        )

        future_probabilities.append(
            probability
        )

    future_probabilities = np.asarray(
        future_probabilities,
        dtype=np.float32,
    )

    # =====================================================
    # 7. UNIFIED ANALYSIS
    # =====================================================

    intelligence = (
        UnifiedAnalysisService(
            feature_names
        )
    )

    analysis = intelligence.analyze(
        current_state=current_state,
        future_states=future_states,
        current_attack_probability=(
            current_probability
        ),
        future_attack_probabilities=(
            future_probabilities
        ),
    )

    # =====================================================
    # 8. RESULT
    # =====================================================

    return {

        "input": str(
            input_path
        ),

        "model": str(
            model_path
        ),

        "feature_names":
            feature_names,

        "state_dimension":
            state_dimension,

        "sequence_length":
            effective_sequence_length,

        "forecast_steps":
            int(steps),

        "current_state":
            current_state.tolist(),

        "future_states":
            future_states.tolist(),

        "current_attack_probability":
            current_probability,

        "future_attack_probabilities":
            future_probabilities.tolist(),

        "timeline":
            analysis["timeline"],

        "progression_scores":
            analysis[
                "progression_scores"
            ],

        "mitre_results":
            analysis[
                "mitre_results"
            ],

        "stage_timeline":
            analysis[
                "stage_timeline"
            ],

        "feature_contributions":
            analysis[
                "feature_contributions"
            ],

        "summary":
            analysis[
                "summary"
            ],
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "ARJUN K-step World Model "
            "Forecast"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="CSV / PCAP / Zeek input",
    )

    parser.add_argument(
        "--model",
        default=(
            "saved_models/"
            "hybrid_world_model.pt"
        ),
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--window-seconds",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    result = forecast_file(
        input_path=args.input,
        model_path=args.model,
        sequence_length=(
            args.sequence_length
        ),
        steps=args.steps,
        window_seconds=(
            args.window_seconds
        ),
    )

    print()
    print("=" * 70)
    print("ARJUN FORECAST")
    print("=" * 70)

    print(
        f"Current attack probability: "
        f"{result['current_attack_probability']:.2%}"
    )

    print()

    for item in result["timeline"]:

        print(
            f"Step {item['step']:02d} | "
            f"Risk: "
            f"{item['risk_score']:.2f} | "
            f"Level: "
            f"{item['risk_level']} | "
            f"MITRE: "
            f"{item['mitre_stage']}"
        )

    print()
    print(
        "Highest risk stage: "
        f"{result['summary']['highest_risk_mitre_stage']}"
    )

    print(
        "Highest risk level: "
        f"{result['summary']['maximum_risk_level']}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()