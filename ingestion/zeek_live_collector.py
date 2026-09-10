"""
ARJUN Z6 - LIVE ZEEK COLLECTOR

Continuously watches a Zeek conn.log and connects the live telemetry path to:

    Zeek conn.log
        -> ARJUN Zeek parser/state builder
        -> 5-second network states + communication graphs
        -> trained Hybrid World Model
        -> recursive K-step forecast
        -> risk + MITRE + feature contributions
        -> optional email alert

This collector is designed for the current ARJUN project structure.
It does NOT require a folder restructure.

Important implementation note
-----------------------------
The current ARJUN Zeek-to-state builder is a batch builder. To preserve the
existing, tested feature engineering exactly, this live collector re-processes
the current conn.log only when the file changes, then advances the downstream
pipeline only when a NEW completed state window appears. This gives live
behaviour without changing the already-tested Zeek feature code.

For very large production deployments, replace the rebuild step with a
streaming/window aggregator (Kafka/Redis/etc.). The external interface can
remain the same.

Usage (Windows)
---------------
    python ingestion\\zeek_live_collector.py --input C:\\zeek\\logs\\current\\conn.log

Useful options:
    --interval 2       Poll every 2 seconds
    --steps 5          Forecast five future states
    --window-seconds 5 Use ARJUN's 5-second state windows
    --once             Process once and exit
    --no-email         Do not send email even if SMTP is configured

The collector writes Zeek-specific artifacts only:
    data/processed/zeek_network_states.npz
    data/processed/zeek_network_graphs.pkl
    data/processed/zeek_world_model_forecast.npz
    data/processed/zeek_forecast_risk.npz
    data/processed/zeek_forecast_risk.json
    alerts/zeek_alert_history.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"
CHECKPOINT = ROOT / "saved_models" / "hybrid_world_model.pt"
TRAINING = PROCESSED / "training_states.npz"
STATE_FILE = PROCESSED / "zeek_network_states.npz"
GRAPH_FILE = PROCESSED / "zeek_network_graphs.pkl"
FORECAST_FILE = PROCESSED / "zeek_world_model_forecast.npz"
RISK_FILE = PROCESSED / "zeek_forecast_risk.npz"
RISK_JSON = PROCESSED / "zeek_forecast_risk.json"
ALERT_HISTORY = ROOT / "alerts" / "zeek_alert_history.json"
LIVE_STATUS_FILE = PROCESSED / "zeek_live_status.json"


# ---------------------------------------------------------------------------
# Existing ARJUN components
# ---------------------------------------------------------------------------

from ingestion.zeek_to_state import build_zeek_states
from forecasting.hybrid_k_step_forecaster import (
    load_checkpoint,
    load_state_scaler,
    load_model,
    forecast_k_steps,
)


@dataclass
class LiveSnapshot:
    """State kept between polling cycles."""

    signature: Optional[tuple[int, int]] = None
    processed_window_count: int = 0
    last_window_id: Optional[int] = None
    last_window_start: Optional[str] = None


class ZeekLiveCollector:
    """Poll a Zeek conn.log and run the ARJUN predictive pipeline."""

    def __init__(
        self,
        input_path: str | Path,
        interval: float = 2.0,
        window_seconds: int = 5,
        forecast_steps: int = 5,
        send_email: bool = True,
    ) -> None:
        self.input_path = Path(input_path).expanduser().resolve()
        self.interval = max(0.5, float(interval))
        self.window_seconds = int(window_seconds)
        self.forecast_steps = int(forecast_steps)
        self.send_email = bool(send_email)
        self.snapshot = LiveSnapshot()
        self._model_cache: Optional[dict[str, Any]] = None

        if self.window_seconds <= 0:
            raise ValueError("--window-seconds must be greater than zero.")
        if self.forecast_steps <= 0:
            raise ValueError("--steps must be greater than zero.")

    # ---------------------------------------------------------------------
    # File state
    # ---------------------------------------------------------------------

    def file_signature(self) -> Optional[tuple[int, int]]:
        """Return (mtime_ns, size), or None if the file is unavailable."""
        try:
            stat = self.input_path.stat()
        except OSError:
            return None
        return int(stat.st_mtime_ns), int(stat.st_size)

    def wait_for_input(self) -> bool:
        """Wait until the Zeek file exists."""
        if self.input_path.exists():
            return True
        print(f"WAITING: Zeek log not found: {self.input_path}")
        return False

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------

    def load_model_once(self, state_dimension: int, graph_dimension: int):
        """Load the trained World Model once and reuse it across cycles."""
        if self._model_cache is not None:
            cached = self._model_cache
            if cached["state_dimension"] == state_dimension and cached["graph_dimension"] == graph_dimension:
                return cached["model"], cached["scaler"], cached["config"]

        if not CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Trained World Model checkpoint not found: {CHECKPOINT}"
            )

        checkpoint = load_checkpoint()
        scaler = load_state_scaler(checkpoint)
        model, config = load_model(
            checkpoint,
            state_dimension,
            graph_dimension,
        )

        self._model_cache = {
            "model": model,
            "scaler": scaler,
            "config": config,
            "state_dimension": state_dimension,
            "graph_dimension": graph_dimension,
        }
        return model, scaler, config

    # ---------------------------------------------------------------------
    # Artifact writing
    # ---------------------------------------------------------------------

    @staticmethod
    def _feature_names() -> list[str]:
        if TRAINING.exists():
            with np.load(TRAINING, allow_pickle=True) as data:
                if "feature_names" in data:
                    return [str(x) for x in data["feature_names"]]
        return [f"feature_{i}" for i in range(33)]

    def save_forecast(
        self,
        forecast: np.ndarray,
        input_states: np.ndarray,
        window_ids: np.ndarray,
    ) -> None:
        PROCESSED.mkdir(parents=True, exist_ok=True)

        forecast = np.asarray(forecast, dtype=np.float32)
        input_states = np.asarray(input_states, dtype=np.float32)
        window_ids = np.asarray(window_ids, dtype=np.int64)

        if forecast.ndim != 2:
            raise ValueError(f"Forecast must be 2-D, got {forecast.shape}.")
        if not np.isfinite(forecast).all():
            raise ValueError("Forecast contains NaN or infinite values.")

        np.savez_compressed(
            FORECAST_FILE,
            forecast=forecast,
            forecast_states=forecast,
            input_states=input_states,
            input_window_ids=window_ids,
            feature_names=np.asarray(self._feature_names(), dtype=object),
            source="live_zeek_collector",
            generated_at=np.asarray(time.time(), dtype=np.float64),
        )

    # ---------------------------------------------------------------------
    # Risk + alert integration
    # ---------------------------------------------------------------------

    def run_risk_pipeline(self) -> dict[str, Any]:
        """Run the already-tested Z4 adapter against the fresh Zeek artifacts."""
        risk_script = ROOT / "risk" / "zeek_forecast_risk.py"
        if not risk_script.exists():
            raise FileNotFoundError(
                f"Zeek risk adapter not found: {risk_script}"
            )

        spec = importlib.util.spec_from_file_location(
            "arjun_zeek_forecast_risk",
            risk_script,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load risk/zeek_forecast_risk.py")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()

        if not RISK_FILE.exists():
            raise RuntimeError("Z4 completed without producing zeek_forecast_risk.npz")

        with np.load(RISK_FILE, allow_pickle=True) as data:
            probabilities = np.asarray(
                data["attack_probability"], dtype=np.float32
            )
            scores = np.asarray(data["risk_score"], dtype=np.float32)
            stages = [str(x) for x in data["mitre_stage"]]
            overall = str(data["overall_risk_level"].item()) if "overall_risk_level" in data else "UNKNOWN"

        return {
            "probabilities": probabilities,
            "scores": scores,
            "stages": stages,
            "overall": overall,
        }

    def run_alert_pipeline(self) -> str:
        """Create a Zeek-specific alert and optionally send its email."""
        try:
            from alerts.alert_engine import AlertEngine
            from alerts.notification_service import EmailNotificationService
        except Exception as exc:
            return f"ALERT SYSTEM UNAVAILABLE: {exc}"

        engine = AlertEngine(RISK_FILE, ALERT_HISTORY)
        indicators = [
            "Source: live Zeek conn.log",
            "ARJUN: 5-second network state windows",
            "ARJUN: trained Hybrid World Model recursive forecast",
        ]

        try:
            alert = engine.raise_alert(indicators=indicators)
        except Exception as exc:
            return f"ALERT ENGINE FAILED: {exc}"

        if alert is None:
            return "NO NEW ALERT: duplicate suppressed or no risk forecast available."

        if not self.send_email:
            return (
                f"ALERT CREATED: {alert.alert_id} | {alert.severity} | "
                f"{alert.attack_probability:.2%} | {alert.mitre_stage} | EMAIL DISABLED"
            )

        try:
            result = EmailNotificationService().send(
                alert.to_dict(),
                dry_run=False,
            )
            sent = bool(result.get("sent", False))
            engine.mark_email_sent(alert.alert_id, sent)
            if sent:
                return (
                    f"EMAIL SENT: {alert.alert_id} | {alert.severity} | "
                    f"{alert.attack_probability:.2%} | {alert.mitre_stage}"
                )
            return f"ALERT CREATED: {alert.alert_id} | EMAIL NOT SENT"
        except Exception as exc:
            engine.mark_email_sent(alert.alert_id, False)
            return f"EMAIL FAILED: {exc} (alert retained for retry)"

    # ---------------------------------------------------------------------
    # One live processing cycle
    # ---------------------------------------------------------------------

    def write_live_status(self, **updates: Any) -> None:
        """Persist a small dashboard-safe heartbeat/status artifact."""
        PROCESSED.mkdir(parents=True, exist_ok=True)
        status = {
            "status": "RUNNING",
            "updated_at": time.time(),
            "input": str(self.input_path),
            "interval_seconds": self.interval,
            "window_seconds": self.window_seconds,
            "forecast_steps": self.forecast_steps,
            "email_enabled": self.send_email,
        }
        status.update(updates)
        tmp = LIVE_STATUS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
        tmp.replace(LIVE_STATUS_FILE)

    def process_once(self, force: bool = False) -> bool:
        """
        Process one poll cycle.

        Returns True when new telemetry produced at least one new state window.
        """
        self.write_live_status(status="WAITING_FOR_INPUT")
        if not self.wait_for_input():
            return False

        signature = self.file_signature()
        if signature is None:
            return False

        # No file change -> no need to rebuild the existing batch feature set.
        if not force and self.snapshot.signature == signature:
            print("NO CHANGE: Zeek conn.log unchanged; waiting...")
            return False

        print()
        print("-" * 78)
        print(f"ZEEK UPDATE DETECTED | size={signature[1]:,} bytes")
        print(f"Input: {self.input_path}")

        try:
            _, states, window_ids, graphs = build_zeek_states(
                self.input_path,
                PROCESSED,
                window_seconds=self.window_seconds,
            )
        except TypeError:
            # Backward-compatible fallback if the current builder does not
            # expose window_seconds as a keyword.
            _, states, window_ids, graphs = build_zeek_states(
                self.input_path,
                PROCESSED,
            )

        states = np.asarray(states, dtype=np.float32)
        window_ids = np.asarray(window_ids, dtype=np.int64)

        if states.ndim != 2 or states.shape[1] != 33:
            raise RuntimeError(
                f"Expected Zeek state shape (N,33), got {states.shape}."
            )
        if len(graphs) != len(states):
            raise RuntimeError(
                f"State/graph mismatch: {len(states)} states vs {len(graphs)} graphs."
            )
        if len(states) == 0:
            print("NO COMPLETE WINDOWS: waiting for more Zeek telemetry...")
            self.snapshot.signature = signature
            return False

        latest_window_id = int(window_ids[-1])
        latest_window_start = None
        if STATE_FILE.exists():
            try:
                with np.load(STATE_FILE, allow_pickle=True) as data:
                    if "window_starts" in data and len(data["window_starts"]):
                        latest_window_start = str(data["window_starts"][-1])
            except Exception:
                latest_window_start = None

        new_window = (
            self.snapshot.last_window_id is None
            or latest_window_id > self.snapshot.last_window_id
        )

        print(f"Zeek rows -> states : {len(states):,} -> {len(states):,}")
        print(f"State dimension     : {states.shape[1]}")
        print(f"Graph count         : {len(graphs):,}")
        print(f"Latest window ID    : {latest_window_id}")
        if latest_window_start:
            print(f"Latest window start : {latest_window_start}")
        self.write_live_status(
            status="STATE_READY",
            state_count=int(len(states)),
            latest_window_id=int(latest_window_id),
            latest_window_start=str(latest_window_start),
        )

        # Record file signature even if the final line is currently incomplete.
        self.snapshot.signature = signature

        if not new_window:
            print("NO NEW COMPLETE WINDOW: waiting for the next Zeek window...")
            return False

        if len(states) < 10:
            self.snapshot.last_window_id = latest_window_id
            self.snapshot.processed_window_count = len(states)
            self.snapshot.last_window_start = latest_window_start
            print(
                f"TEMPORAL WARM-UP: {len(states)}/10 windows available; "
                "World Model inference will start after 10 windows."
            )
            return True

        # ---------------------------------------------------------------
        # World Model inference
        # ---------------------------------------------------------------
        graph_dimension = int(
            np.asarray(graphs[-1]["node_features"]).shape[-1]
        )
        model, scaler, config = self.load_model_once(
            states.shape[1],
            graph_dimension,
        )

        initial_states = states[-10:].astype(np.float32)
        graph_history = graphs[-10:]

        print()
        print("WORLD MODEL: running live 10-state context...")
        forecast = forecast_k_steps(
            model=model,
            initial_states=initial_states,
            graph_sequence=graph_history,
            scaler=scaler,
            k=self.forecast_steps,
        )
        forecast = np.asarray(forecast, dtype=np.float32)

        expected = (self.forecast_steps, 33)
        if forecast.shape != expected:
            raise RuntimeError(
                f"Unexpected forecast shape: expected {expected}, got {forecast.shape}."
            )
        if not np.isfinite(forecast).all():
            raise RuntimeError("Live forecast contains NaN or infinite values.")

        self.save_forecast(
            forecast=forecast,
            input_states=initial_states,
            window_ids=window_ids[-10:],
        )

        print(f"Model residual scale : {config.get('residual_scale', 'n/a')}")
        print(f"Forecast shape       : {forecast.shape}")
        for step, prediction in enumerate(forecast, start=1):
            previous = initial_states[-1] if step == 1 else forecast[step - 2]
            delta = prediction - previous
            print(
                f"  t+{step}: mean |delta|={np.mean(np.abs(delta)):.6f} "
                f"max |delta|={np.max(np.abs(delta)):.6f}"
            )

        # ---------------------------------------------------------------
        # Risk + MITRE + explainability + alert
        # ---------------------------------------------------------------
        print()
        print("RISK ENGINE: evaluating live forecast...")
        risk = self.run_risk_pipeline()

        if len(risk["probabilities"]):
            p = float(risk["probabilities"][-1])
            r = float(risk["scores"][np.argmax(risk["probabilities"])]) if len(risk["scores"]) else 0.0
            stage = risk["stages"][-1] if risk["stages"] else "Unknown"
            print(
                f"LIVE FORECAST: t+{len(risk['probabilities'])} "
                f"attack={p:.2%} risk={r:.2%} stage={stage} "
                f"overall={risk['overall'].upper()}"
            )

        self.write_live_status(
            status="FORECAST_READY",
            state_count=int(len(states)),
            latest_window_id=int(latest_window_id),
            latest_window_start=str(latest_window_start),
            forecast_steps=int(len(risk.get("probabilities", []))),
            peak_attack_probability=float(np.max(risk["probabilities"])) if len(risk.get("probabilities", [])) else None,
            peak_risk_score=float(np.max(risk["scores"])) if len(risk.get("scores", [])) else None,
            overall_risk=str(risk.get("overall", "UNKNOWN")),
            predicted_stage=str(risk["stages"][-1]) if risk.get("stages") else "Unknown",
        )

        alert_status = self.run_alert_pipeline()
        print(f"ALERT: {alert_status}")
        self.write_live_status(alert_status=str(alert_status), status="ONLINE")

        self.snapshot.last_window_id = latest_window_id
        self.snapshot.processed_window_count = len(states)
        self.snapshot.last_window_start = latest_window_start

        print()
        print("LIVE ZEEK CYCLE: PASSED")
        return True

    # ---------------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------------

    def run(self, once: bool = False) -> None:
        print("=" * 78)
        print("ARJUN Z6: LIVE ZEEK COLLECTOR")
        print("=" * 78)
        print(f"Zeek input       : {self.input_path}")
        print(f"Poll interval    : {self.interval:g}s")
        print(f"State window     : {self.window_seconds}s")
        print(f"Forecast steps   : {self.forecast_steps}")
        print(f"Email alerts     : {'ON' if self.send_email else 'OFF'}")
        print(f"World Model      : {CHECKPOINT}")
        print()
        print("Pipeline:")
        print("  Zeek -> states/graphs -> World Model -> forecast -> risk -> alert")
        print("=" * 78)

        if once:
            self.process_once(force=True)
            return

        try:
            while True:
                try:
                    self.process_once()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    print(f"LIVE CYCLE ERROR: {type(exc).__name__}: {exc}")
                    self.write_live_status(status="ERROR", error=f"{type(exc).__name__}: {exc}")
                    print("Collector remains alive and will retry on the next poll.")
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("\nARJUN LIVE ZEEK COLLECTOR STOPPED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ARJUN live Zeek conn.log collector and predictive pipeline"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to Zeek conn.log (classic TSV or JSON Lines).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds (default: 2).",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=5,
        help="ARJUN state window size in seconds (default: 5).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of recursive forecast steps (default: 5).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process the current file once and exit.",
    )
    parser.add_argument(
        "--no-email",
        action="store_true",
        help="Create alerts but do not send email.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    collector = ZeekLiveCollector(
        input_path=args.input,
        interval=args.interval,
        window_seconds=args.window_seconds,
        forecast_steps=args.steps,
        send_email=not args.no_email,
    )
    collector.run(once=args.once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
