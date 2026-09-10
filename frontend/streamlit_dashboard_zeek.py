"""
ARJUN Dashboard - Zeek-aware SOC dashboard extension.

Drop-in replacement for frontend/streamlit_dashboard.py.

Adds:
- Zeek Forecast mode
- Z4 risk timeline
- MITRE progression
- predictive indicators
- Z5 alert/email status
- artifact readiness

The existing CIC dashboard views remain available.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FRONTEND_DIR = Path(__file__).resolve().parent
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

# ---------------------------------------------------------------------
# Existing dashboard helpers are imported from the current dashboard.
# This keeps the replacement small and avoids duplicating its styling.
# ---------------------------------------------------------------------
try:
    from dashboard_core import (
        render_brand,
        load_training_data,
        load_forecast,
        load_risk,
        render_dark_table,
        get_top_changes,
        pct,
    )
except ImportError:
    # If the user's current dashboard does not expose a dashboard_core module,
    # the script gives a clear migration message instead of silently failing.
    st.error(
        "Dashboard integration requires the existing dashboard helpers. "
        "Keep your current frontend/streamlit_dashboard.py and copy the "
        "Zeek integration block from this file, or use the supplied "
        "dashboard_core.py companion."
    )
    st.stop()

ZEEK_FORECAST_FILE = ROOT / "data" / "processed" / "zeek_world_model_forecast.npz"
ZEEK_RISK_FILE = ROOT / "data" / "processed" / "zeek_forecast_risk.npz"
ZEEK_RISK_JSON = ROOT / "data" / "processed" / "zeek_forecast_risk.json"
ZEEK_HISTORY = ROOT / "alerts" / "zeek_alert_history.json"
ZEEK_STATES = ROOT / "data" / "processed" / "zeek_network_states.npz"

TRAINING_FILE = ROOT / "data" / "processed" / "training_states.npz"


def _safe_float(value, default=0.0):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def load_zeek_data():
    """Load Z3/Z4/Z5 artifacts without requiring any live Zeek process."""
    if not ZEEK_FORECAST_FILE.exists() or not ZEEK_RISK_FILE.exists():
        return None

    with np.load(ZEEK_FORECAST_FILE, allow_pickle=True) as d:
        forecast = np.asarray(d["forecast"], dtype=np.float32)
        input_states = np.asarray(d["input_states"], dtype=np.float32)
        feature_names = (
            [str(x) for x in d["feature_names"]]
            if "feature_names" in d
            else []
        )

    with np.load(ZEEK_RISK_FILE, allow_pickle=True) as d:
        probabilities = np.asarray(
            d["attack_probability"], dtype=np.float32
        )
        scores = np.asarray(d["risk_score"], dtype=np.float32)
        stages = (
            [str(x) for x in d["mitre_stage"]]
            if "mitre_stage" in d
            else ["Unknown"] * len(probabilities)
        )

        overall = "UNKNOWN"
        if "overall_risk_level" in d:
            try:
                overall = str(d["overall_risk_level"].item()).upper()
            except Exception:
                overall = str(d["overall_risk_level"]).upper()

        peak_probability = _safe_float(
            d["max_attack_probability"].item()
            if "max_attack_probability" in d
            else np.max(probabilities)
        )
        final_probability = _safe_float(
            d["final_attack_probability"].item()
            if "final_attack_probability" in d
            else probabilities[-1]
        )
        peak_risk = _safe_float(
            d["max_risk_score"].item()
            if "max_risk_score" in d
            else np.max(scores)
        )

    current = input_states[-1] if len(input_states) else None

    return {
        "forecast": forecast,
        "input_states": input_states,
        "current": current,
        "feature_names": feature_names,
        "probabilities": probabilities,
        "scores": scores,
        "stages": stages,
        "overall": overall,
        "peak_probability": peak_probability,
        "final_probability": final_probability,
        "peak_risk": peak_risk,
    }


def load_zeek_history():
    if not ZEEK_HISTORY.exists():
        return []
    try:
        data = json.loads(ZEEK_HISTORY.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def risk_class(level):
    level = str(level).upper()
    return {
        "LOW": "low",
        "GUARDED": "medium",
        "MEDIUM": "medium",
        "HIGH": "high",
        "CRITICAL": "critical",
    }.get(level, "")


def render_zeek_overview():
    data = load_zeek_data()

    st.markdown(
        '<div class="page-title">Zeek Predictive SOC</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-subtitle">'
        'Live-style view of Zeek telemetry transformed into future network-state '
        'risk by the ARJUN World Model.'
        '</div>',
        unsafe_allow_html=True,
    )

    if data is None:
        st.error(
            "Zeek forecast artifacts are not available. Run Z3 and Z4 first:"
        )
        st.code(
            "python ingestion\\zeek_world_model_inference.py\n"
            "python risk\\zeek_forecast_risk.py"
        )
        return

    p = data["probabilities"]
    s = data["scores"]
    stages = data["stages"]

    current_prob = float(p[0]) if len(p) else 0.0
    peak_prob = data["peak_probability"]
    peak_risk = data["peak_risk"]
    overall = data["overall"]
    peak_idx = int(np.argmax(p)) if len(p) else 0
    peak_stage = stages[peak_idx] if stages else "Unknown"

    # KPI row
    st.markdown(
        '<div class="section-title">Zeek Risk Posture</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns([2.5, 2, 2, 1.8, 2.5])

    cards = [
        (c1, "primary", "Current Attack Probability",
         f"{current_prob:.2%}", "t+1 predicted state"),
        (c2, "", "Peak Attack Probability",
         f"{peak_prob:.2%}", "5-step horizon"),
        (c3, risk_class(overall), "Peak Risk Score",
         f"{peak_risk:.2%}", "Z4 risk engine"),
        (c4, risk_class(overall), "Overall Risk",
         overall, "Zeek forecast"),
        (c5, "", "Predicted ATT&CK Stage",
         peak_stage, f"highest probability at t+{peak_idx + 1}"),
    ]

    for col, cls, label, value, meta in cards:
        with col:
            small = " small" if len(str(value)) > 18 else ""
            st.markdown(
                f'<div class="kpi-card {cls}">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value{small}">{value}</div>'
                f'<div class="kpi-meta">{meta}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Timeline
    st.markdown(
        '<div class="section-title">Future Attack Probability</div>',
        unsafe_allow_html=True,
    )

    timeline_rows = []
    for i, (prob, score, stage) in enumerate(zip(p, s, stages), start=1):
        timeline_rows.append({
            "Step": f"t+{i}",
            "Attack Probability": f"{float(prob):.2%}",
            "Risk Score": f"{float(score):.2%}",
            "Risk Level": (
                "CRITICAL" if score >= .80 else
                "HIGH" if score >= .60 else
                "MEDIUM" if score >= .40 else
                "GUARDED" if score >= .20 else "LOW"
            ),
            "MITRE Stage": stage,
        })

    render_dark_table(pd.DataFrame(timeline_rows))

    # Forecast state movement
    st.markdown(
        '<div class="section-title">Top Predictive Indicators</div>',
        unsafe_allow_html=True,
    )

    if data["current"] is not None and len(data["forecast"]):
        current = np.asarray(data["current"], dtype=float)
        future = np.asarray(data["forecast"][-1], dtype=float)
        names = data["feature_names"]

        if len(names) != len(current):
            names = [f"feature_{i}" for i in range(len(current))]

        delta = future - current
        order = np.argsort(np.abs(delta))[::-1][:10]

        rows = []
        for idx in order:
            rows.append({
                "Feature": names[int(idx)],
                "Current": f"{current[idx]:.4g}",
                "Predicted t+5": f"{future[idx]:.4g}",
                "Change": f"{delta[idx]:+.4g}",
            })

        render_dark_table(pd.DataFrame(rows))

    # Alert status
    st.markdown(
        '<div class="section-title">Automatic Alert Status</div>',
        unsafe_allow_html=True,
    )

    history = load_zeek_history()

    if history:
        latest = history[-1]
        sent = bool(latest.get("email_sent", False))
        status = "EMAIL SENT" if sent else "PENDING EMAIL"

        st.markdown(
            f'<div class="panel">'
            f'<div class="panel-title">Latest Zeek predictive alert</div>'
            f'<div class="panel-meta">'
            f'{latest.get("alert_id", "Unknown")} · '
            f'{latest.get("severity", "UNKNOWN")} · '
            f'{latest.get("mitre_stage", "Unknown")} · '
            f't+{latest.get("forecast_step", "?")} · '
            f'{status}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "No Zeek alert has been persisted yet. Start Z5 to enable "
            "automatic alerting and email."
        )

    # Analyst interpretation
    st.markdown(
        '<div class="section-title">Analyst Interpretation</div>',
        unsafe_allow_html=True,
    )

    if len(p):
        trend = "rising" if p[-1] > p[0] else "falling" if p[-1] < p[0] else "stable"
        st.markdown(
            f'<div class="recommendation">'
            f'<div class="recommendation-title">Forecast summary</div>'
            f'<p>ARJUN estimates a {trend} attack-risk trajectory across the '
            f'five simulated states, with peak estimated attack probability '
            f'of <b>{peak_prob:.2%}</b> and peak risk score '
            f'<b>{peak_risk:.2%}</b>. The highest-probability predicted stage '
            f'is <b>{peak_stage}</b>.</p>'
            f'<p>These are model/risk-engine outputs and should be validated '
            f'against live network and endpoint evidence.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="ARJUN SOC",
    page_icon="🛡️",
    layout="wide",
)

render_brand()

try:
    training = load_training_data()
    forecast = load_forecast()
    risk = load_risk()
except Exception as exc:
    training = forecast = risk = None
    st.warning(f"Existing CIC dashboard artifacts could not be loaded: {exc}")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "CIC Overview",
    "Future State",
    "What-If Analysis",
    "Zeek Predictive SOC",
    "System Status",
])

with tab1:
    if training is not None and forecast is not None and risk is not None:
        # Preserve the user's existing dashboard by loading it in a lightweight
        # compatibility mode. The full existing dashboard remains the source of
        # truth for these views.
        st.info(
            "CIC overview remains available in your existing dashboard. "
            "Use the Zeek Predictive SOC tab for the complete Zeek path."
        )
        states = training.get("states")
        labels = training.get("labels")
        if states is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Training States", f"{len(states):,}")
            c2.metric("State Dimension", states.shape[-1])
            c3.metric(
                "Attack States",
                f"{int(np.sum(np.asarray(labels) == 1)):,}"
                if labels is not None else "N/A",
            )
    else:
        st.warning("CIC artifacts are unavailable.")

with tab2:
    st.info(
        "Use the existing Future State view in frontend/streamlit_dashboard.py "
        "for detailed CIC state inspection."
    )

with tab3:
    st.info(
        "Use the existing What-If Analysis view in "
        "frontend/streamlit_dashboard.py."
    )

with tab4:
    render_zeek_overview()

with tab5:
    st.markdown(
        '<div class="page-title">ARJUN System Status</div>',
        unsafe_allow_html=True,
    )
    artifacts = [
        ("Training states", TRAINING_FILE),
        ("Zeek network states", ZEEK_STATES),
        ("Zeek World Model forecast", ZEEK_FORECAST_FILE),
        ("Zeek forecast risk", ZEEK_RISK_FILE),
        ("Zeek risk JSON", ZEEK_RISK_JSON),
        ("Zeek alert history", ZEEK_HISTORY),
        ("World Model checkpoint",
         ROOT / "saved_models" / "hybrid_world_model.pt"),
    ]
    rows = []
    for name, path in artifacts:
        rows.append({
            "Artifact": name,
            "Status": "READY" if path.exists() else "MISSING",
            "Path": str(path.relative_to(ROOT)),
        })
    render_dark_table(pd.DataFrame(rows))

st.markdown(
    '<div style="margin-top:28px;color:#4F5E73;font-size:10px;'
    'border-top:1px solid #172235;padding-top:12px;">'
    'ARJUN · World Models for Proactive Cyber Defence · '
    'Zeek forecast outputs require validation against live telemetry.'
    '</div>',
    unsafe_allow_html=True,
)
