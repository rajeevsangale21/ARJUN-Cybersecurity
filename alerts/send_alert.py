"""Generate and send one ARJUN predictive alert."""
from __future__ import annotations

import argparse
from pathlib import Path

from alert_engine import AlertEngine
from notification_service import EmailNotificationService


def main():
    parser = argparse.ArgumentParser(
        description="ARJUN predictive alert email sender"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Test alert generation and email message creation locally. "
            "SMTP credentials are NOT required."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Build a fresh alert even if the current forecast "
            "was already recorded."
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    engine = AlertEngine(root=root)

    try:
        alert = (
            engine.build_alert()
            if args.force
            else engine.raise_alert()
        )
    except Exception as exc:
        print(f"ALERT FAILED: {exc}")
        return 1

    if alert is None:
        print(
            "No new alert: duplicate suppressed or no forecast available."
        )
        return 0

    try:
        result = EmailNotificationService().send(
            alert.to_dict(),
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"EMAIL FAILED: {exc}")
        if not args.force:
            print(
                "The alert remains unsent and can be retried after "
                "fixing the email configuration."
            )
        return 1

    # A dry-run must never mark an alert as emailed.
    if not args.force and not args.dry_run:
        engine.mark_email_sent(
            alert.alert_id,
            bool(result.get("sent")),
        )

    print("\nARJUN ALERT TEST PASSED" if args.dry_run else "\nARJUN ALERT SENT")
    print(f"Alert:       {alert.alert_id}")
    print(f"Severity:    {alert.severity}")
    print(f"Probability: {alert.attack_probability:.4f}")
    print(f"Risk:        {alert.risk_score:.4f}")
    print(f"Forecast:    t+{alert.forecast_step}")
    print(f"Stage:       {alert.mitre_stage}")

    if args.dry_run:
        configured = result.get("configuration", {}).get(
            "smtp_configured", False
        )
        print(
            "SMTP config: "
            + ("CONFIGURED" if configured else "NOT CONFIGURED")
        )
        print("Email:       DRY RUN ONLY — no email was sent")
    else:
        print(f"Email:       {result.get('recipient', 'SENT')}")

    if args.force:
        print("Mode:        FORCE TEST")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
