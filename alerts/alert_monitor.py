"""Automatic ARJUN predictive-alert monitor.

Watches the current Zeek forecast-risk artifact and sends an email when a new
predictive alert is produced. Exact duplicates are suppressed by AlertEngine.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from alert_engine import AlertEngine
from notification_service import EmailNotificationService


DEFAULT_INTERVAL = 30


class AlertMonitor:
    def __init__(self, root: Path, interval: int = DEFAULT_INTERVAL):
        self.root = Path(root)
        self.interval = max(1, int(interval))

        self.risk_file = (
            self.root / "data" / "processed" / "zeek_forecast_risk.json"
        )
        self.history_file = self.root / "alerts" / "zeek_alert_history.json"

        self.engine = AlertEngine(self.risk_file, self.history_file)
        self.notifier = EmailNotificationService()
        self._last_mtime = None

    def _mtime(self):
        if not self.risk_file.exists():
            return None
        return self.risk_file.stat().st_mtime_ns

    def check_once(self, force_check: bool = False):
        mtime = self._mtime()
        if mtime is None:
            print(f"WAITING: forecast not found: {self.risk_file}")
            return None

        if not force_check and self._last_mtime == mtime:
            print("NO CHANGE: forecast artifact unchanged; waiting...")
            return False

        self._last_mtime = mtime

        try:
            alert = self.engine.raise_alert()
        except Exception as exc:
            print(f"ALERT FAILED: {exc}")
            return False

        if alert is None:
            print("NO NEW ALERT: duplicate suppressed or no forecast available.")
            return False

        print(
            "NEW ALERT: "
            f"{alert.severity} | "
            f"probability={alert.attack_probability:.2%} | "
            f"risk={alert.risk_score:.2%} | "
            f"t+{alert.forecast_step} | "
            f"stage={alert.mitre_stage}"
        )

        try:
            result = self.notifier.send(alert.to_dict(), dry_run=False)
        except Exception as exc:
            print(f"EMAIL FAILED: {exc}")
            print(
                "Alert remains unsent and can be retried after fixing "
                "email configuration."
            )
            return False

        self.engine.mark_email_sent(
            alert.alert_id, bool(result.get("sent"))
        )
        print(f"EMAIL SENT: {result.get('recipient', 'configured recipient')}")
        return True

    def run(self, once: bool = False):
        print("ARJUN AUTOMATIC ALERT MONITOR")
        print(f"Forecast : {self.risk_file}")
        print(f"History  : {self.history_file}")
        print(f"Interval : {self.interval}s")
        print("Mode     : ONCE" if once else "Mode     : CONTINUOUS")
        print("Press Ctrl+C to stop.\n")

        try:
            self.check_once(force_check=True)

            if once:
                return 0

            while True:
                time.sleep(self.interval)
                self.check_once(force_check=False)
        except KeyboardInterrupt:
            print("\nARJUN ALERT MONITOR STOPPED")
            return 0


def main():
    parser = argparse.ArgumentParser(
        description="ARJUN automatic predictive-alert monitor"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Seconds between forecast checks (default: {DEFAULT_INTERVAL}).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Perform one check and exit.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    return AlertMonitor(root, args.interval).run(once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
