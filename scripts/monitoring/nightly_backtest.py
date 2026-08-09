#!/usr/bin/env python3
"""
Nightly backtest runner:
- Updates data
- Runs backtests for multiple ensemble methods (and random baseline)
- Picks best method and saves to outputs/results/best_ensemble.json
- Emits alert file if best underperforms random baseline by threshold
- Optionally sends email if SMTP env is configured
"""

from __future__ import annotations

import json
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Dict

from scripts.validations.backtest import run_backtest


RESULTS_DIR = Path('outputs/results')
ALERTS_DIR = Path('outputs/monitoring/alerts')


def pick_best(metrics: Dict[str, Dict[str, float]], methods: list[str]) -> str:
    # Rank by partial_3plus_rate, then avg_matches
    best = None
    best_key = None
    for m in methods:
        mm = metrics.get(m, {})
        key = (mm.get('partial_3plus_rate', 0.0), mm.get('avg_matches', 0.0))
        if best is None or key > best:
            best = key
            best_key = m
    return best_key or methods[0]


def maybe_send_email(subject: str, body: str) -> None:
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


def main() -> int:
    methods = ['frequency', 'weighted', 'consensus', 'probmap']
    lookback = int(os.environ.get('BACKTEST_LOOKBACK', '200'))
    step = int(os.environ.get('BACKTEST_STEP', '1'))
    threshold = float(os.environ.get('ALERT_THRESHOLD', '0.02'))  # 2% default

    # Run backtests and gather metrics
    results_by_method: Dict[str, Dict] = {}
    for m in methods:
        res = run_backtest(lookback=lookback, method=m, step=step, compare=['random'], no_plot=True)
        results_by_method[m] = res

    # Build a combined metrics view
    combined_metrics: Dict[str, Dict[str, float]] = {}
    for m, res in results_by_method.items():
        for k, v in res.get('metrics', {}).items():
            combined_metrics.setdefault(k, v)

    # Decide best method
    best = pick_best(combined_metrics, methods)

    # Persist best
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = RESULTS_DIR / 'best_ensemble.json'
    with open(best_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'lookback': lookback,
            'methods_considered': methods,
            'best_method': best,
            'metrics': combined_metrics,
        }, f, indent=2)

    # Alert if best underperforms random
    best_m = combined_metrics.get(best, {})
    rand_m = combined_metrics.get('random', {})
    if best_m and rand_m:
        drop = rand_m.get('partial_3plus_rate', 0.0) - best_m.get('partial_3plus_rate', 0.0)
        if drop > threshold:
            ALERTS_DIR.mkdir(parents=True, exist_ok=True)
            alert_path = ALERTS_DIR / f'alert_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            body = (
                f"Alert: Best ensemble '{best}' underperformed random by {drop:.4f} (threshold {threshold})\n"
                f"Random 3+ rate: {rand_m.get('partial_3plus_rate', 0.0):.4f}\n"
                f"Best   3+ rate: {best_m.get('partial_3plus_rate', 0.0):.4f}\n"
            )
            with open(alert_path, 'w') as a:
                a.write(body)
            # Silenced with the rest: only a PLAY verdict earns the inbox.
            # The alert file in outputs/monitoring/alerts/ still lands.

    print(f"Best ensemble method: {best} (saved to {best_path})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


