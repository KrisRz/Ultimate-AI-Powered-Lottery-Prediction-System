"""Email, and nothing else.

The PLAY alert is the one output of this project that has to survive
everything, and it used to borrow its sender from nightly_backtest.py - which
imports the backtest, which imports the legacy LSTM predictor, which imports
TensorFlow. Every PLAY email therefore loaded 1.2 s of a machine-learning
stack it never used, and any import failure anywhere in that legacy code
would have taken the alert down with it. Standard library only, on purpose.
"""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage


def maybe_send_email(subject: str, body: str) -> None:
    """Send when SMTP_SERVER / SMTP_USER / SMTP_PASS / EMAIL_TO are set; say so
    when they are not; never raise."""
    server = os.environ.get('SMTP_SERVER')
    user = os.environ.get('SMTP_USER')
    password = os.environ.get('SMTP_PASS')
    to_addr = os.environ.get('EMAIL_TO')
    from_addr = os.environ.get('EMAIL_FROM', user)
    if not (server and user and password and to_addr and from_addr):
        print("[email] SMTP env not configured (SMTP_SERVER/SMTP_USER/SMTP_PASS/EMAIL_TO) - not sending")
        return
    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = from_addr
        msg['To'] = to_addr
        msg.set_content(body)
        with smtplib.SMTP_SSL(server, 465) as s:
            s.login(user, password)
            s.send_message(msg)
        print(f"[email] SENT to {to_addr}: {subject}")
    except Exception as e:
        # Loud failure - an alert that silently fails is worse than none
        print(f"[email] FAILED to send ({type(e).__name__}): {e}")
