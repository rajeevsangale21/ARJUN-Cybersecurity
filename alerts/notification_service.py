"""SMTP email notifications for ARJUN alerts.

Normal sends require SMTP configuration from environment variables.
Dry-run mode deliberately does NOT require credentials; it validates and
renders the alert message locally so the alert pipeline can be tested before
email is configured.
"""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage


class EmailConfigurationError(RuntimeError):
    pass


class EmailNotificationService:
    def __init__(self):
        self.host = os.getenv("ARJUN_EMAIL_HOST", "")
        self.port = int(os.getenv("ARJUN_EMAIL_PORT", "587"))
        self.username = os.getenv("ARJUN_EMAIL_USERNAME", "")
        self.password = os.getenv("ARJUN_EMAIL_PASSWORD", "")
        self.recipient = os.getenv("ARJUN_ALERT_RECIPIENT", "")
        self.use_tls = os.getenv(
            "ARJUN_EMAIL_USE_TLS", "true"
        ).lower() not in {"0", "false", "no", "off"}

    def validate(self):
        missing = [
            key
            for key, value in [
                ("ARJUN_EMAIL_HOST", self.host),
                ("ARJUN_EMAIL_USERNAME", self.username),
                ("ARJUN_EMAIL_PASSWORD", self.password),
                ("ARJUN_ALERT_RECIPIENT", self.recipient),
            ]
            if not value
        ]
        if missing:
            raise EmailConfigurationError(
                "Missing email configuration: " + ", ".join(missing)
            )

    @staticmethod
    def pct(value):
        try:
            return f"{float(value) * 100:.2f}%"
        except (TypeError, ValueError):
            return "N/A"

    def message(self, alert, validate=True):
        if validate:
            self.validate()

        severity = str(alert.get("severity", "UNKNOWN")).upper()
        stage = alert.get("mitre_stage", "Unknown")

        lines = []
        for item in (alert.get("indicators") or [])[:8]:
            if isinstance(item, dict):
                feature = item.get(
                    "feature",
                    item.get("name", "Unknown")
                )
                change = item.get(
                    "change",
                    item.get("value", "N/A")
                )
                lines.append(f"- {feature}: {change}")
            else:
                lines.append(f"- {item}")

        if not lines:
            lines = [
                "- See ARJUN forecast/risk output for contributing indicators."
            ]

        body = f"""ARJUN CYBER DEFENCE SYSTEM
========================================

PREDICTIVE SECURITY ALERT

Alert ID: {alert.get('alert_id', 'N/A')}
Status: PREDICTED / PRE-ATTACK
Severity: {severity}
Attack Probability: {self.pct(alert.get('attack_probability'))}
Risk Score: {self.pct(alert.get('risk_score'))}
Forecast: t+{alert.get('forecast_step', 'N/A')}
Predicted MITRE ATT&CK Stage: {stage}

TOP PREDICTIVE INDICATORS
----------------------------------------
{chr(10).join(lines)}

RECOMMENDED ACTION
----------------------------------------
{alert.get('recommendation', 'Validate the forecast against live telemetry.')}

IMPORTANT
----------------------------------------
This is a predictive/pre-attack signal. It is not proof that an attack is
currently occurring. Validate against live network and endpoint evidence.

ARJUN
World Models for Proactive Cyber Defence
"""

        message = EmailMessage()
        message["Subject"] = f"[ARJUN] PRE-ATTACK ALERT | {severity} | {stage}"
        message["From"] = self.username or "arjun@localhost"
        message["To"] = self.recipient or "dry-run@localhost"
        message.set_content(body)
        return message

    def send(self, alert, dry_run=False):
        if dry_run:
            # No SMTP credentials are required for a local pipeline test.
            message = self.message(alert, validate=False)
            return {
                "sent": False,
                "dry_run": True,
                "subject": message["Subject"],
                "recipient": message["To"],
                "configuration": {
                    "smtp_configured": bool(
                        self.host
                        and self.username
                        and self.password
                        and self.recipient
                    )
                },
            }

        self.validate()
        message = self.message(alert, validate=False)

        with smtplib.SMTP(
            self.host,
            self.port,
            timeout=20
        ) as server:
            server.ehlo()
            if self.use_tls:
                server.starttls()
                server.ehlo()
            server.login(self.username, self.password)
            server.send_message(message)

        return {
            "sent": True,
            "dry_run": False,
            "subject": message["Subject"],
            "recipient": message["To"],
        }
