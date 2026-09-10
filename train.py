import argparse
from pathlib import Path

import numpy as np

from pipeline import run_pipeline
from world_model.world_model import WorldModel
from world_model.trainer import WorldModelTrainer


def main():

    parser = argparse.ArgumentParser(
        description="Train ARJUN World Model"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV, PCAP or Zeek log"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
        help="Number of previous states"
    )

    parser.add_argument(
        "--output",
        default="saved_models/world_model.pt",
        help="Model output path"
    )

    args = parser.parse_args()

    # --------------------------------
    # Run preprocessing
    # --------------------------------

    print("\n" + "=" * 60)
    print("ARJUN WORLD MODEL TRAINING")
    print("=" * 60)

    result = run_pipeline(
        args.input
    )

    states = result["states"]
    window_ids = result["window_ids"]

    # --------------------------------
    # Rebuild sequences if required
    # --------------------------------

    from world_model.sequence_builder import build_sequences

    if len(states) <= args.sequence_length:

        raise ValueError(
            f"Dataset contains only {len(states)} "
            f"network states. "
            f"Need more than "
            f"{args.sequence_length}."
        )

    X, y, _ = build_sequences(
        states,
        window_ids,
        args.sequence_length
    )

    print("\nTraining data:")
    print(f"Input shape : {X.shape}")
    print(f"Target shape: {y.shape}")

    # --------------------------------
    # Create World Model
    # --------------------------------

    state_dimension = states.shape[1]

    print(
        f"\nState dimension: "
        f"{state_dimension}"
    )

    world_model = WorldModel(
        state_dimension=state_dimension
    )

    print(
        f"Device: "
        f"{world_model.device}"
    )

    # --------------------------------
    # Train
    # --------------------------------

    trainer = WorldModelTrainer(
        world_model
    )

    history = trainer.train(
        X,
        y,
        epochs=args.epochs
    )

    # --------------------------------
    # Save
    # --------------------------------

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    world_model.save(
        output_path
    )

    # --------------------------------
    # Final result
    # --------------------------------

    print("\n" + "=" * 60)
    print("WORLD MODEL TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"States used : {len(states):,}"
    )

    print(
        f"Sequences   : {len(X):,}"
    )

    print(
        f"State size  : {state_dimension}"
    )

    print(
        f"Final loss  : {history[-1]:.6f}"
    )

    print(
        f"Model saved : {output_path}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()