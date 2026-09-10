from fastapi import APIRouter, Query
from typing import Optional

from database.repository import (
    get_recent_analyses,
    get_recent_forecasts,
    get_recent_risks,
    get_database_stats,
)

router = APIRouter(
    prefix="/records",
    tags=["Records"],
)


@router.get("/stats")
def read_stats():
    """
    Get summary database statistics.
    """
    return get_database_stats()


@router.get("/analyses")
def read_analyses(
    limit: int = Query(10, ge=1, le=100)
):
    """
    Get recently executed telemetry analyses.
    """
    return get_recent_analyses(limit=limit)


@router.get("/forecasts")
def read_forecasts(
    limit: int = Query(10, ge=1, le=100)
):
    """
    Get recently executed world model attack forecasts.
    """
    return get_recent_forecasts(limit=limit)


@router.get("/risks")
def read_risks(
    limit: int = Query(50, ge=1, le=200),
    analysis_id: Optional[int] = Query(None)
):
    """
    Get recorded risk timeline steps.
    """
    return get_recent_risks(limit=limit, analysis_id=analysis_id)
