import os
import tempfile
from pathlib import Path

import numpy as np

from pipeline import run_pipeline

from database.repository import (
    create_forecast_record,
    update_forecast_record,
)

from world_model.hybrid_wrapper import (
    HybridWorldModelWrapper,
)

from forecasting.hybrid_k_step_forecaster import (
    HybridKStepForecaster,
)


async def forecast_uploaded_file(
    file,
    steps=10,
    sequence_length=5,
):
    """
    Run ARJUN K-step future attack-state forecasting.
    """

    suffix = Path(
        file.filename or ""
    ).suffix.lower()

    temp_path = None
    forecast_id = None

    try:

        contents = await file.read()

        if not contents:
            raise ValueError(
                "Uploaded file is empty."
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_file.write(contents)
            temp_path = temp_file.name

        # -------------------------------------------------
        # Pipeline
        # -------------------------------------------------

        result = run_pipeline(
            temp_path,
            sequence_length=sequence_length,
        )

        states = np.asarray(
            result["states"],
            dtype=np.float32,
        )

        graph_sequences = result.get(
            "graph_sequences"
        )

        if graph_sequences is None:
            raise ValueError(
                "Pipeline did not produce graph sequences."
            )

        if len(graph_sequences) == 0:
            raise ValueError(
                "No graph sequence is available "
                "for forecasting."
            )

        effective_sequence_length = int(
            result["sequence_length"]
        )

        if len(states) < (
            effective_sequence_length
        ):
            raise ValueError(
                "Not enough network states for forecasting."
            )

        state_dimension = int(
            states.shape[1]
        )

        # -------------------------------------------------
        # Database
        # -------------------------------------------------

        forecast_id = create_forecast_record(
            filename=(
                file.filename or "unknown"
            ),
            steps=steps,
            sequence_length=(
                effective_sequence_length
            ),
        )

        # -------------------------------------------------
        # Load trained model
        # -------------------------------------------------

        model_path = Path(
            "saved_models/"
            "hybrid_world_model.pt"
        )

        if not model_path.exists():

            raise FileNotFoundError(
                "Hybrid World Model not found at "
                f"{model_path}. "
                "Train it using train_hybrid.py first."
            )

        world_model = (
            HybridWorldModelWrapper.from_checkpoint(
                model_path,
                state_dimension=state_dimension,
                device="cpu",
            )
        )

        # -------------------------------------------------
        # Last observed sequence
        # -------------------------------------------------

        current_states = states[
            -effective_sequence_length:
        ]

        current_graphs = (
            graph_sequences[-1]
        )

        # -------------------------------------------------
        # Forecast
        # -------------------------------------------------

        forecaster = (
            HybridKStepForecaster(
                world_model,
                device="cpu",
            )
        )

        future_states = (
            forecaster.forecast(
                current_states,
                current_graphs,
                steps=steps,
            )
        )

        future_states = np.asarray(
            future_states,
            dtype=np.float32,
        )

        # -------------------------------------------------
        # Database update
        # -------------------------------------------------

        update_forecast_record(
            forecast_id=forecast_id,
            status="completed",
        )

        return {
            "forecast_id": forecast_id,
            "filename": file.filename,
            "status": "completed",
            "steps": int(steps),
            "sequence_length": (
                effective_sequence_length
            ),
            "state_dimension": (
                state_dimension
            ),
            "future_states": (
                future_states.tolist()
            ),
        }

    except Exception as exc:

        if forecast_id is not None:

            update_forecast_record(
                forecast_id=forecast_id,
                status="failed",
                error_message=str(exc),
            )

        raise

    finally:

        if (
            temp_path is not None
            and os.path.exists(temp_path)
        ):

            os.remove(temp_path)