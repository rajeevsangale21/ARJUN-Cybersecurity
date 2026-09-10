from datetime import datetime

from database.connection import SessionLocal

from database.models import (
    AnalysisRecord,
    ForecastRecord,
    RiskRecord,
)


# =========================================================
# ANALYSIS
# =========================================================

def create_analysis_record(
    filename,
    input_type,
):
    db = SessionLocal()

    try:

        record = AnalysisRecord(
            filename=filename,
            input_type=input_type,
            status="processing",
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return record.id

    finally:

        db.close()


def update_analysis_record(
    analysis_id,
    state_count=None,
    state_dimension=None,
    feature_count=None,
    status=None,
    error_message=None,
):
    db = SessionLocal()

    try:

        record = (
            db.query(AnalysisRecord)
            .filter(
                AnalysisRecord.id
                == analysis_id
            )
            .first()
        )

        if record is None:
            return

        if state_count is not None:
            record.state_count = state_count

        if state_dimension is not None:
            record.state_dimension = (
                state_dimension
            )

        if feature_count is not None:
            record.feature_count = (
                feature_count
            )

        if status is not None:
            record.status = status

        if error_message is not None:
            record.error_message = (
                error_message
            )

        if status in {
            "completed",
            "failed",
        }:

            record.completed_at = (
                datetime.utcnow()
            )

        db.commit()

    finally:

        db.close()


# =========================================================
# FORECAST
# =========================================================

def create_forecast_record(
    filename,
    steps,
    sequence_length,
):
    db = SessionLocal()

    try:

        record = ForecastRecord(
            filename=filename,
            steps=steps,
            sequence_length=sequence_length,
            status="processing",
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return record.id

    finally:

        db.close()


def update_forecast_record(
    forecast_id,
    status=None,
    error_message=None,
):
    db = SessionLocal()

    try:

        record = (
            db.query(ForecastRecord)
            .filter(
                ForecastRecord.id
                == forecast_id
            )
            .first()
        )

        if record is None:
            return

        if status is not None:
            record.status = status

        if error_message is not None:
            record.error_message = (
                error_message
            )

        if status in {
            "completed",
            "failed",
        }:

            record.completed_at = (
                datetime.utcnow()
            )

        db.commit()

    finally:

        db.close()


# =========================================================
# RISK
# =========================================================

def save_risk_record(
    analysis_id,
    step,
    risk_score,
    risk_level,
    attack_probability,
    mitre_stage,
    mitre_confidence,
):
    db = SessionLocal()

    try:

        record = RiskRecord(
            analysis_id=analysis_id,
            step=step,
            risk_score=risk_score,
            risk_level=risk_level,
            attack_probability=(
                attack_probability
            ),
            mitre_stage=mitre_stage,
            mitre_confidence=(
                mitre_confidence
            ),
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return record.id

    finally:

        db.close()


# =========================================================
# QUERY HELPERS
# =========================================================

def get_recent_analyses(limit=10):
    db = SessionLocal()
    try:
        records = (
            db.query(AnalysisRecord)
            .order_by(AnalysisRecord.id.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "filename": r.filename,
                "input_type": r.input_type,
                "state_count": r.state_count,
                "state_dimension": r.state_dimension,
                "feature_count": r.feature_count,
                "status": r.status,
                "error_message": r.error_message,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            }
            for r in records
        ]
    finally:
        db.close()


def get_recent_forecasts(limit=10):
    db = SessionLocal()
    try:
        records = (
            db.query(ForecastRecord)
            .order_by(ForecastRecord.id.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "filename": r.filename,
                "steps": r.steps,
                "sequence_length": r.sequence_length,
                "status": r.status,
                "error_message": r.error_message,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            }
            for r in records
        ]
    finally:
        db.close()


def get_recent_risks(limit=50, analysis_id=None):
    db = SessionLocal()
    try:
        query = db.query(RiskRecord)
        if analysis_id is not None:
            query = query.filter(RiskRecord.analysis_id == analysis_id)
        records = query.order_by(RiskRecord.id.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "analysis_id": r.analysis_id,
                "step": r.step,
                "risk_score": r.risk_score,
                "risk_level": r.risk_level,
                "attack_probability": r.attack_probability,
                "mitre_stage": r.mitre_stage,
                "mitre_confidence": r.mitre_confidence,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]
    finally:
        db.close()


def get_database_stats():
    db = SessionLocal()
    try:
        total_analyses = db.query(AnalysisRecord).count()
        total_forecasts = db.query(ForecastRecord).count()
        total_risks = db.query(RiskRecord).count()
        return {
            "total_analyses": total_analyses,
            "total_forecasts": total_forecasts,
            "total_risks": total_risks,
        }
    finally:
        db.close()