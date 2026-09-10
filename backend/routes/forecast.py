from fastapi import APIRouter, File, UploadFile, HTTPException

from backend.services.forecast_service import (
    forecast_uploaded_file,
)


router = APIRouter(
    prefix="/forecast",
    tags=["Forecast"],
)


@router.post("")
async def forecast_file(
    file: UploadFile = File(...),
    steps: int = 10,
    sequence_length: int = 5,
):
    """
    Run ARJUN K-step future attack forecasting.
    """

    if steps < 1:
        raise HTTPException(
            status_code=400,
            detail="steps must be at least 1.",
        )

    if sequence_length < 1:
        raise HTTPException(
            status_code=400,
            detail="sequence_length must be at least 1.",
        )

    try:

        result = await forecast_uploaded_file(
            file=file,
            steps=steps,
            sequence_length=sequence_length,
        )

        return result

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc