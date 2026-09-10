from fastapi import APIRouter, File, UploadFile, HTTPException

from backend.services.analysis_service import (
    analyze_uploaded_file,
)


router = APIRouter(
    prefix="/analysis",
    tags=["Analysis"],
)


@router.post("")
async def analyze_file(
    file: UploadFile = File(...)
):
    """
    Upload telemetry and run ARJUN analysis.
    """

    allowed_extensions = {
        ".csv",
        ".pcap",
        ".pcapng",
        ".cap",
        ".log",
        ".tsv",
        ".txt",
    }

    filename = file.filename or ""

    extension = ""

    if "." in filename:
        extension = (
            "."
            + filename.rsplit(".", 1)[1].lower()
        )

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Supported formats: CSV, PCAP, "
                "PCAPNG, CAP, LOG, TSV and TXT."
            ),
        )

    try:
        result = await analyze_uploaded_file(
            file
        )

        return result

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc