"""ARJUN predictive/pre-attack alert engine.

Reads the current Zeek forecast-risk artifact, creates a forecast-driven
alert, and persists it to the same alert-history file used by the dashboard.
Supports both JSON and NPZ risk artifacts so the alert layer does not depend
on one serialization format.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass
class Alert:
    alert_id: str
    timestamp: str
    status: str
    severity: str
    attack_probability: float
    risk_score: float
    forecast_step: int
    mitre_stage: str
    indicators: List[Dict[str, Any]]
    recommendation: str
    acknowledged: bool = False
    email_sent: bool = False

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            alert_id=str(data.get("alert_id", "")),
            timestamp=str(data.get("timestamp", "")),
            status=str(data.get("status", "PREDICTED")),
            severity=str(data.get("severity", "MEDIUM")),
            attack_probability=float(data.get("attack_probability", 0.0)),
            risk_score=float(data.get("risk_score", 0.0)),
            forecast_step=int(data.get("forecast_step", 1)),
            mitre_stage=str(data.get("mitre_stage", "Unknown")),
            indicators=list(data.get("indicators", [])),
            recommendation=str(data.get("recommendation", "")),
            acknowledged=bool(data.get("acknowledged", False)),
            email_sent=bool(data.get("email_sent", False)),
        )


class AlertEngine:
    def __init__(self, risk_file=None, history_file=None, root=None):
        if root is not None:
            root = Path(root)
            if risk_file is None:
                risk_file = root / "data" / "processed" / "zeek_forecast_risk.json"
            if history_file is None:
                history_file = root / "alerts" / "zeek_alert_history.json"

        self.risk_file = Path(risk_file) if risk_file else Path(
            "data/processed/zeek_forecast_risk.json"
        )
        self.history_file = Path(history_file) if history_file else Path(
            "alerts/zeek_alert_history.json"
        )

    @staticmethod
    def _f(v, default=0.0):
        try:
            x = float(np.asarray(v).reshape(-1)[0])
            return x if np.isfinite(x) else default
        except (TypeError, ValueError, IndexError):
            return default

    @staticmethod
    def _prob(v):
        x = AlertEngine._f(v)
        if abs(x) > 1.5:
            x /= 100.0
        return float(np.clip(x, 0.0, 1.0))

    @staticmethod
    def _risk(v):
        x = AlertEngine._f(v)
        if abs(x) > 1.5:
            x /= 100.0
        return float(np.clip(x, 0.0, 1.0))

    @staticmethod
    def _level(v):
        text = str(v).upper().strip()
        return text if text in {
            "LOW", "GUARDED", "MEDIUM", "HIGH", "CRITICAL"
        } else "UNKNOWN"

    @staticmethod
    def _array(value):
        if value is None:
            return np.asarray([])
        try:
            return np.asarray(value).reshape(-1)
        except Exception:
            return np.asarray([])

    def _load_json(self):
        if not self.risk_file.exists():
            raise FileNotFoundError(self.risk_file)

        data = json.loads(self.risk_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return [], [], [], []

        # Preferred production/dashboard representation.
        timeline = data.get("timeline") or data.get("risk_timeline")
        if isinstance(timeline, list) and timeline:
            probs, risks, stages, levels = [], [], [], []
            for item in timeline:
                if not isinstance(item, dict):
                    continue
                probs.append(self._prob(
                    item.get("attack_probability",
                             item.get("probability", 0.0))
                ))
                risks.append(self._risk(item.get("risk_score", 0.0)))
                stages.append(str(item.get(
                    "mitre_stage", item.get("stage", "Unknown")
                )))
                levels.append(self._level(item.get(
                    "risk_level", item.get("level", "UNKNOWN")
                )))
            return probs, risks, stages, levels

        probs = self._array(
            data.get("attack_probability",
                     data.get("attack_probabilities",
                              data.get("probabilities", [])))
        )
        risks = self._array(
            data.get("risk_score",
                     data.get("risk_scores", []))
        )
        stages = data.get(
            "mitre_stage", data.get("mitre_stages",
            data.get("stages", []))
        )
        levels = data.get(
            "risk_level", data.get("risk_levels", [])
        )

        probs = [self._prob(x) for x in probs]
        risks = [self._risk(x) for x in risks]
        stages = [str(x) for x in self._array(stages)]
        levels = [self._level(x) for x in self._array(levels)]
        return probs, risks, stages, levels

    def _load_npz(self):
        if not self.risk_file.exists():
            raise FileNotFoundError(self.risk_file)

        with np.load(self.risk_file, allow_pickle=True) as d:
            probs = d["attack_probability"] if "attack_probability" in d else (
                d["attack_probabilities"]
                if "attack_probabilities" in d else np.asarray([])
            )
            risks = d["risk_score"] if "risk_score" in d else (
                d["risk_scores"] if "risk_scores" in d else np.asarray([])
            )
            stages = d["mitre_stage"] if "mitre_stage" in d else (
                d["mitre_stages"] if "mitre_stages" in d else np.asarray([])
            )
            levels = d["risk_level"] if "risk_level" in d else (
                d["risk_levels"] if "risk_levels" in d else np.asarray([])
            )

        return (
            [self._prob(x) for x in self._array(probs)],
            [self._risk(x) for x in self._array(risks)],
            [str(x) for x in self._array(stages)],
            [self._level(x) for x in self._array(levels)],
        )

    def load(self):
        suffix = self.risk_file.suffix.lower()
        if suffix == ".json":
            return self._load_json()
        if suffix == ".npz":
            return self._load_npz()

        # Graceful fallback for an artifact with an unexpected extension.
        try:
            return self._load_json()
        except Exception:
            return self._load_npz()

    def severity(self, probability, risk_score, level="UNKNOWN"):
        level = self._level(level)
        if level in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}:
            return level

        # Dashboard/architecture risk levels include GUARDED. For alert
        # severity, GUARDED is promoted to MEDIUM so it is actionable.
        if risk_score >= 0.80 or probability >= 0.70:
            return "CRITICAL"
        if risk_score >= 0.60 or probability >= 0.45:
            return "HIGH"
        if risk_score >= 0.20 or probability >= 0.30:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def recommendation(severity, stage):
        if severity == "CRITICAL":
            return (
                f"Immediately investigate activity associated with the predicted "
                f"{stage} stage and follow incident-response procedures."
            )
        if severity == "HIGH":
            return (
                f"Investigate unusual network behaviour associated with the predicted "
                f"{stage} stage and correlate with live telemetry."
            )
        if severity == "MEDIUM":
            return (
                f"Review network activity related to the predicted {stage} stage "
                "and validate against additional security evidence."
            )
        return (
            "Continue monitoring and validate the forecast against available telemetry."
        )

    def _history(self):
        if not self.history_file.exists():
            return []
        try:
            value = json.loads(self.history_file.read_text(encoding="utf-8"))
            return value if isinstance(value, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def _save(self, history):
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.history_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(history, indent=2), encoding="utf-8")
        tmp.replace(self.history_file)

    def build_alert(self, indicators=None):
        probs, risks, stages, levels = self.load()
        if not probs:
            return None

        # Alert on the highest-risk forecast point, matching the dashboard.
        n = len(probs)
        risk_values = [
            risks[i] if i < len(risks) else 0.0 for i in range(n)
        ]
        i = int(np.argmax(np.asarray(risk_values, dtype=float)))

        probability = self._prob(probs[i])
        risk_score = self._risk(risk_values[i])
        stage = stages[i] if i < len(stages) else "Unknown"
        level = levels[i] if i < len(levels) else "UNKNOWN"
        severity = self.severity(probability, risk_score, level)

        evidence = indicators or [
            {
                "name": "Forecasted attack probability",
                "value": f"{probability:.2%}",
            },
            {
                "name": "Forecast risk score",
                "value": f"{risk_score:.2%}",
            },
            {
                "name": "Trigger",
                "value": "ARJUN World Model forecast",
            },
        ]

        return Alert(
            alert_id=f"ARJUN-PREDICTED-{uuid.uuid4().hex[:8].upper()}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="PREDICTED",
            severity=severity,
            attack_probability=probability,
            risk_score=risk_score,
            forecast_step=i + 1,
            mitre_stage=stage,
            indicators=evidence,
            recommendation=self.recommendation(severity, stage),
        )

    @staticmethod
    def _is_duplicate(old, alert):
        try:
            return (
                old.get("status") == "PREDICTED"
                and int(old.get("forecast_step", -1)) == alert.forecast_step
                and old.get("mitre_stage") == alert.mitre_stage
                and abs(
                    AlertEngine._f(old.get("attack_probability"))
                    - alert.attack_probability
                ) < 1e-6
                and abs(
                    AlertEngine._f(old.get("risk_score"))
                    - alert.risk_score
                ) < 1e-6
            )
        except Exception:
            return False

    def raise_alert(self, indicators=None):
        """Create a new alert, suppressing exact duplicates.

        If an earlier identical alert was created but its email was not sent,
        return that alert so notification can be retried.
        """
        alert = self.build_alert(indicators)
        if alert is None:
            return None

        history = self._history()
        for old in reversed(history[-20:]):
            if self._is_duplicate(old, alert):
                if not bool(old.get("email_sent", False)):
                    return Alert.from_dict(old)
                return None

        history.append(alert.to_dict())
        self._save(history)
        return alert

    def mark_email_sent(self, alert_id, sent=True):
        history = self._history()
        changed = False
        for item in reversed(history):
            if item.get("alert_id") == alert_id:
                item["email_sent"] = bool(sent)
                changed = True
                break
        if changed:
            self._save(history)
        return changed

    def acknowledge(self, alert_id):
        history = self._history()
        changed = False
        for item in reversed(history):
            if item.get("alert_id") == alert_id:
                item["acknowledged"] = True
                item["status"] = "ACKNOWLEDGED"
                changed = True
                break
        if changed:
            self._save(history)
        return changed

    def history(self):
        return self._history()

    def latest_alert(self):
        history = self._history()
        return history[-1] if history else None


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    engine = AlertEngine(root=root)
    alert = engine.raise_alert()
    print(
        "ALERT ENGINE: PASSED"
        if alert or engine.latest_alert()
        else "No forecast available"
    )
