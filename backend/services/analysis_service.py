import os
import tempfile
from pathlib import Path

import numpy as np

from pipeline import run_pipeline

from database.repository import (
    create_analysis_record,
    update_analysis_record,
)


async def analyze_uploaded_file(file):
    """
    Save uploaded telemetry temporarily,
    run the ARJUN preprocessing/state pipeline,
    and store the analysis metadata in the database.
    """

    suffix = Path(
        file.filename or ""
    ).suffix.lower()

    temp_path = None
    analysis_id = None

    try:

        # -------------------------------------------------
        # Save uploaded file
        # -------------------------------------------------

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
        # Create database record
        # -------------------------------------------------

        analysis_id = create_analysis_record(
            filename=file.filename or "unknown",
            input_type=suffix.lstrip("."),
        )

        # -------------------------------------------------
        # Run ARJUN pipeline
        # -------------------------------------------------

        result = run_pipeline(
            temp_path,
            sequence_length=5,
        )

        states = np.asarray(
            result.get("states", []),
            dtype=np.float32,
        )

        features = result.get(
            "features",
            result.get("data"),
        )

        state_count = (
            int(len(states))
            if states.ndim >= 1
            else 0
        )

        state_dimension = (
            int(states.shape[1])
            if states.ndim == 2
            else 0
        )

        feature_count = (
            int(features.shape[1])
            if hasattr(features, "shape")
            and len(features.shape) == 2
            else 0
        )

        # -------------------------------------------------
        # Update database
        # -------------------------------------------------

        update_analysis_record(
            analysis_id=analysis_id,
            state_count=state_count,
            state_dimension=state_dimension,
            feature_count=feature_count,
            status="completed",
        )

        # -------------------------------------------------
        # API response
        # -------------------------------------------------

        return {
            "analysis_id": analysis_id,
            "filename": file.filename,
            "status": "completed",
            "state_count": state_count,
            "state_dimension": state_dimension,
            "feature_count": feature_count,
            "sequence_length": result.get(
                "sequence_length",
                5,
            ),
            "feature_names": result.get(
                "feature_names",
                [],
            ),
        }

    except Exception as exc:

        if analysis_id is not None:

            update_analysis_record(
                analysis_id=analysis_id,
                status="failed",
                error_message=str(exc),
            )

        raise

    finally:

        if temp_path and os.path.exists(
            temp_path
        ):

            os.remove(temp_path)