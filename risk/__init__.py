def get_risk_level(score):
    """
    Convert a risk score in the range 0-1
    into a human-readable risk level.
    """

    score = max(
        0.0,
        min(1.0, float(score))
    )

    if score < 0.20:
        return "LOW"

    if score < 0.40:
        return "MEDIUM"

    if score < 0.70:
        return "HIGH"

    return "CRITICAL"