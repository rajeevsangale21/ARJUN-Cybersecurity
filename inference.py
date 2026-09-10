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
from evaluation.xgboost_classifier import (
    XGBoostAttackClassifier,
)


def main():

    parser = argparse.ArgumentParser(
        description="ARJUN inference"
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--world-model",
        default=(
            "saved_models/"
            "hybrid_world_model.pt"
        ),
    )

    parser.add_argument(
        "--classifier",
        default=(
            "saved_models/"
            "xgboost_attack_classifier.pkl"
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    result = run_pipeline(
        args.input,
        sequence_length=args.sequence_length,
    )

    states = np.asarray(
        result["states"],
        dtype=np.float32,
    )

    state_dimension = states.shape[1]

    world_model = (
        HybridWorldModelWrapper.from_checkpoint(
            Path(args.world_model),
            state_dimension=state_dimension,
            device="cpu",
        )
    )

    classifier = (
        XGBoostAttackClassifier()
    )

    classifier.load(
        args.classifier
    )

    current_state = states[-1]

    current_probability = float(
        classifier.predict_proba(
            current_state.reshape(
                1,
                -1,
            )
        )[0]
    )

    sequence = states[
        -args.sequence_length:
    ]

    forecaster = (
        HybridKStepForecaster(
            world_model
        )
    )

    future_states = (
        forecaster.forecast(
            sequence,
            steps=args.steps,
        )
    )

    future_states = np.asarray(
        future_states,
        dtype=np.float32,
    )

    future_probabilities = []

    for state in future_states:

        probability = float(
            classifier.predict_proba(
                state.reshape(
                    1,
                    -1,
                )
            )[0]
        )

        future_probabilities.append(
            probability
        )

    print()
    print("=" * 60)
    print("ARJUN INFERENCE")
    print("=" * 60)

    print(
        f"Current attack probability: "
        f"{current_probability:.2%}"
    )

    print(
        "\nFuture probability:"
    )

    for index, probability in enumerate(
        future_probabilities,
        start=1,
    ):

        print(
            f"  Step {index}: "
            f"{probability:.2%}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()