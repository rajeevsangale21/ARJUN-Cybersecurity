"""
ARJUN Z5 - Zeek Risk -> Automatic Alert -> Email

Connects the Z4 Zeek forecast-risk artifact to the existing ARJUN
AlertEngine and EmailNotificationService.

Normal flow:
    Zeek -> World Model -> Z4 risk -> Z5 alert -> SMTP email

This file does NOT modify the existing generic alert monitor.
It uses a separate history file for Zeek-originated alerts so that
Zeek and CIC forecast alerts cannot suppress one another.

Usage:
    python alerts/zeek_alert_monitor.py --once
    python alerts/zeek_alert_monitor.py --interval 30

The --dry-run option validates/builds the email without sending it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alerts.alert_engine import AlertEngine
from alerts.notification_service import EmailNotificationService


DEFAULT_INTERVAL = 30

ZEEK_RISK_FILE = ROOT / "data" / "processed" / "zeek_forecast_risk.npz"
ZEEK_HISTORY_FILE = ROOT / "alerts" / "zeek_alert_history.json"


class ZeekAlertMonitor:
    """Monitor the Z4 Zeek risk artifact and send one email per new alert."""

    def __init__(self, root: Path = ROOT, interval: int = DEFAULT_INTERVAL):
        self.root = Path(root)
        self.interval = max(1, int(interval))

        self.risk_file = self.root / "data" / "processed" / "zeek_forecast_risk.npz"
        self.history_file = self.root / "alerts" / "zeek_alert_history.json"

        self.engine = AlertEngine(self.risk_file, self.history_file)
        self.notifier = EmailNotificationService()

        self._last_mtime = None

    def _mtime(self):
        if not self.risk_file.exists():
            return None
        return self.risk_file.stat().st_mtime_ns

    def check_once(self, force_check: bool = False, dry_run: bool = False):
        """
        Check Z4 output once.

        Returns:
            True  = an alert was generated and email operation succeeded
            False = nothing sent
            None  = Z4 artifact is unavailable
        """
        mtime = self._mtime()

        if mtime is None:
            print(f"WAITING: Z4 risk artifact not found: {self.risk_file}")
            return None

        if not force_check and self._last_mtime == mtime:
            print("NO CHANGE: Zeek risk artifact unchanged; waiting...")
            return False

        self._last_mtime = mtime

        try:
            alert = self.engine.raise_alert(
                indicators=[
                    {
                        "source": "Zeek conn.log",
                        "artifact": "zeek_forecast_risk.npz",
                        "type": "world_model_forecast",
                    }
                ]
            )
        except Exception as exc:
            print(f"ALERT FAILED: {exc}")
            return False

        if alert is None:
            print("NO NEW ALERT: duplicate suppressed or no Zeek forecast available.")
            return False

        print(
            "NEW ZEEK ALERT: "
            f"{alert.severity} | "
            f"probability={alert.attack_probability:.2%} | "
            f"risk={alert.risk_score:.2%} | "
            f"t+{alert.forecast_step} | "
            f"stage={alert.mitre_stage}"
        )

        try:
            result = self.notifier.send(
                alert.to_dict(),
                dry_run=dry_run,
            )
        except Exception as exc:
            print(f"EMAIL FAILED: {exc}")
            print("Alert remains unsent and can be retried after fixing email configuration.")
            return False

        if dry_run:
            print("EMAIL DRY RUN PASSED")
            # Do not mark a dry run as sent. A real run should still email.
            return True

        sent = bool(result.get("sent"))
        self.engine.mark_email_sent(alert.alert_id, sent)

        if sent:
            print(f"EMAIL SENT: {result.get('recipient', 'configured recipient')}")
            return True

        print("EMAIL NOT SENT: notification service returned sent=False")
        return False

    def run(self, once: bool = False, dry_run: bool = False):
        print("=" * 72)
        print("ARJUN Z5: ZEEK -> AUTOMATIC ALERT -> EMAIL")
        print("=" * 72)
        print(f"Z4 risk artifact : {self.risk_file}")
        print(f"Alert history    : {self.history_file}")
        print(f"Interval         : {self.interval}s")
        print(f"Mode             : {'ONCE' if once else 'CONTINUOUS'}")
        print(f"Email mode       : {'DRY RUN' if dry_run else 'LIVE SMTP'}")
        print("Press Ctrl+C to stop.\n")

        try:
            # First check is immediate.
            self.check_once(force_check=True, dry_run=dry_run)

            if once:
                return 0

            while True:
                time.sleep(self.interval)
                self.check_once(force_check=False, dry_run=dry_run)

        except KeyboardInterrupt:
            print("\nARJUN Z5 MONITOR STOPPED")
            return 0


def main():
    parser = argparse.ArgumentParser(
        description="Monitor Z4 Zeek forecast-risk output and send predictive alerts."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Perform one check and exit.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help="Seconds between checks in continuous mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build/validate the email without sending SMTP mail or marking it sent.",
    )
    args = parser.parse_args()

    monitor = ZeekAlertMonitor(interval=args.interval)
    return monitor.run(once=args.once, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
