from fastapi import APIRouter

from database.connection import database_health


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get("")
def health_check():
    """
    Check ARJUN API and database status.
    """

    db_status = database_health()

    return {
        "status": "healthy",
        "service": "ARJUN FastAPI",
        "database": db_status,
    }