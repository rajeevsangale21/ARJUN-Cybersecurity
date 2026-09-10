from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)

from database.connection import Base


class AnalysisRecord(Base):

    __tablename__ = "analysis_records"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    filename = Column(
        String(512),
        nullable=False,
    )

    input_type = Column(
        String(50),
        nullable=False,
    )

    state_count = Column(
        Integer,
        default=0,
    )

    state_dimension = Column(
        Integer,
        default=0,
    )

    feature_count = Column(
        Integer,
        default=0,
    )

    status = Column(
        String(50),
        default="processing",
    )

    error_message = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    completed_at = Column(
        DateTime,
        nullable=True,
    )


class ForecastRecord(Base):

    __tablename__ = "forecast_records"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    filename = Column(
        String(512),
        nullable=False,
    )

    steps = Column(
        Integer,
        nullable=False,
    )

    sequence_length = Column(
        Integer,
        nullable=False,
    )

    status = Column(
        String(50),
        default="processing",
    )

    error_message = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    completed_at = Column(
        DateTime,
        nullable=True,
    )


class RiskRecord(Base):

    __tablename__ = "risk_records"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    analysis_id = Column(
        Integer,
        nullable=True,
    )

    step = Column(
        Integer,
        nullable=True,
    )

    risk_score = Column(
        Float,
        nullable=True,
    )

    risk_level = Column(
        String(50),
        nullable=True,
    )

    attack_probability = Column(
        Float,
        nullable=True,
    )

    mitre_stage = Column(
        String(100),
        nullable=True,
    )

    mitre_confidence = Column(
        Float,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )