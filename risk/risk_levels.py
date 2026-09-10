"""
ARJUN Risk Level Utilities
--------------------------
Maps a normalized risk score in the range [0, 1] to the
risk levels used by the ARJUN risk engine and dashboard.

Thresholds:
    0.00 - 0.19 : LOW
    0.20 - 0.39 : GUARDED
    0.40 - 0.59 : MEDIUM
    0.60 - 0.79 : HIGH
    0.80 - 1.00 : CRITICAL
"""

from __future__ import annotations

import math


RISK_LEVELS = (
    "LOW",
    "GUARDED",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
)


def get_risk_level(risk_score: float) -> str:
    """
    Convert a normalized risk score to an ARJUN risk level.

    Parameters
    ----------
    risk_score:
        Numeric risk score expected in the range [0.0, 1.0].

    Returns
    -------
    str
        One of LOW, GUARDED, MEDIUM, HIGH, or CRITICAL.

    Raises
    ------
    ValueError
        If the supplied score is not finite or cannot be interpreted
        as a numeric value.
    """
    try:
        score = float(risk_score)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Risk score must be numeric, got {risk_score!r}") from exc

    if not math.isfinite(score):
        raise ValueError(f"Risk score must be finite, got {risk_score!r}")

    # RiskEngine normally clips scores to [0, 1]. Keeping this helper
    # defensive makes it safe for direct use by other modules.
    score = max(0.0, min(1.0, score))

    if score < 0.20:
        return "LOW"
    if score < 0.40:
        return "GUARDED"
    if score < 0.60:
        return "MEDIUM"
    if score < 0.80:
        return "HIGH"
    return "CRITICAL"


def get_risk_level_description(risk_level: str) -> str:
    """
    Return a short human-readable description for an ARJUN risk level.
    """
    descriptions = {
        "LOW": "Low observed or forecasted cyber risk.",
        "GUARDED": "Elevated activity warrants continued monitoring.",
        "MEDIUM": "Meaningful cyber risk requiring analyst attention.",
        "HIGH": "High cyber risk requiring prompt investigation.",
        "CRITICAL": "Critical cyber risk requiring immediate response.",
    }

    level = str(risk_level).strip().upper()

    if level not in descriptions:
        raise ValueError(f"Unknown ARJUN risk level: {risk_level!r}")

    return descriptions[level]


def risk_level_to_score_range(risk_level: str) -> tuple[float, float]:
    """
    Return the inclusive nominal score range associated with a risk level.
    """
    ranges = {
        "LOW": (0.00, 0.19),
        "GUARDED": (0.20, 0.39),
        "MEDIUM": (0.40, 0.59),
        "HIGH": (0.60, 0.79),
        "CRITICAL": (0.80, 1.00),
    }

    level = str(risk_level).strip().upper()

    if level not in ranges:
        raise ValueError(f"Unknown ARJUN risk level: {risk_level!r}")

    return ranges[level]


__all__ = [
    "RISK_LEVELS",
    "get_risk_level",
    "get_risk_level_description",
    "risk_level_to_score_range",
]
