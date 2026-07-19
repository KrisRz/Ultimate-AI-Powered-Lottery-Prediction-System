#!/usr/bin/env python3
"""
Rolling backtest for UK Lotto predictions with baseline comparisons.
Generates next-draw predictions across a sliding window and computes metrics.
Outputs JSON summary and optional plot.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from scripts.fetch_data import load_data, DATA_DIR, download_fresh_data
from scripts.new_predict import (
    load_models,
    create_ensemble,
    prepare_input_data,
    predict_with_ensemble,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calc_match_counts(pred: List[int], actual: List[int]) -> int:
    return len(set(pred).intersection(set(actual)))


def _rand_pred() -> List[int]:
    import random
    return sorted(random.sample(range(1, 60), 6))


# A no-skill pick of 6 from 59 matches the drawn 6 with a hypergeometric
# distribution: mean 6*6/59, P(3+) etc. computable exactly
RANDOM_EXPECTED_AVG = 6.0 * 6.0 / 59.0


def significance_vs_random(
    matches: List[int], n_sim: int = 10_000, seed: int | None = None
) -> Dict:
    """Monte-Carlo p-values for a match series against the no-skill null.

    Under the null every prediction is an arbitrary 6-set, so each draw's
    match count is hypergeometric(6 good, 53 bad, 6 sampled). We simulate
    n_sim replicates of the whole series and report the fraction that do at
    least as well as the observed series (one-sided).
    """
    if not matches:
        return {}
    rng = np.random.default_rng(seed)
    steps = len(matches)
    obs_avg = float(np.mean(matches))
    obs_3plus = float(np.mean([m >= 3 for m in matches]))

    sim = rng.hypergeometric(ngood=6, nbad=53, nsample=6, size=(n_sim, steps))
    sim_avg = sim.mean(axis=1)
    sim_3plus = (sim >= 3).mean(axis=1)

    # Bootstrap CI of the observed average (resampling the observed series)
    boot = rng.choice(matches, size=(n_sim, steps), replace=True).mean(axis=1)

    return {
        'expected_random_avg': RANDOM_EXPECTED_AVG,
        'observed_avg': obs_avg,
        'observed_avg_ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        'p_value_avg': float((sim_avg >= obs_avg).mean()),
        'observed_3plus_rate': obs_3plus,
        'p_value_3plus': float((sim_3plus >= obs_3plus).mean()),
        'n_sim': n_sim,
    }


def run_backtest(
    lookback: int = 200,
    method: str = 'frequency',
    step: int = 1,
    compare: Sequence[str] | None = None,
    no_plot: bool = False,
    offline: bool = False,
    seed: int | None = None,
    min_steps: int = 30,
) -> Dict:
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)

    if not offline:
        try:
            download_fresh_data()
        except Exception as e:
            logger.warning(f"Data update failed, proceeding with existing data: {e}")

    df = load_data(DATA_DIR / 'merged_lottery_data.csv')
    if len(df) < lookback + 1:
        new_lookback = max(10, len(df) - 1)
        logger.info(f"Adjusting lookback from {lookback} to {new_lookback} due to limited data (have {len(df)})")
        lookback = new_lookback

    projected_steps = len(range(lookback, len(df), step))
    if projected_steps < min_steps:
        raise SystemExit(
            f"Backtest would evaluate only {projected_steps} draws (< {min_steps}). "
            f"A result on so few draws is statistically meaningless - lower --lookback "
            f"or --step, or extend the data with scripts/backfill_history.py."
        )

    models = load_models()
    if not models:
        raise RuntimeError('No trained models found')
    ensemble = create_ensemble(models)

    series: Dict[str, List[int]] = {method: []}
    dates: List[str] = []
    baselines = list(compare or [])
    for b in baselines:
        series[b] = []

    # Sliding window: use [i-lookback, i) to predict draw i
    for i in range(lookback, len(df), step):
        window = df.iloc[i - lookback:i]
        X = prepare_input_data(window)
        pred = predict_with_ensemble(ensemble, X, combination_method=method)
        actual = df['Main_Numbers'].iloc[i]
        series[method].append(calc_match_counts(pred, actual))

        for b in baselines:
            if b == 'random':
                bp = _rand_pred()
            elif b in {'frequency', 'weighted', 'consensus', 'probmap'}:
                bp = predict_with_ensemble(ensemble, X, combination_method=b)
            else:
                continue
            series[b].append(calc_match_counts(bp, actual))
        dates.append(df['Draw Date'].iloc[i].strftime('%Y-%m-%d'))

    def _rates(ms: List[int]) -> Dict[str, float]:
        return {
            'avg_matches': float(np.mean(ms)) if ms else 0.0,
            'partial_3plus_rate': float(np.mean([m >= 3 for m in ms])) if ms else 0.0,
            'partial_4plus_rate': float(np.mean([m >= 4 for m in ms])) if ms else 0.0,
            'partial_5plus_rate': float(np.mean([m >= 5 for m in ms])) if ms else 0.0,
        }

    results = {
        'timestamp': datetime.now().isoformat(),
        'method': method,
        'lookback': lookback,
        'steps': len(dates),
        'seed': seed,
        'offline': offline,
        'metrics': {k: _rates(v) for k, v in series.items()},
        'significance': {k: significance_vs_random(v, seed=seed) for k, v in series.items()},
        'series': {k: [{'date': d, 'matches': int(m)} for d, m in zip(dates, v)] for k, v in series.items()},
    }

    # Honest human-readable verdict
    print(f"\n{'=' * 62}\nBACKTEST SUMMARY  ({len(dates)} draws, lookback {lookback})\n{'=' * 62}")
    for k, v in series.items():
        sig = results['significance'][k]
        verdict = (
            "significant edge (p<0.05) - verify before believing it"
            if sig.get('p_value_avg', 1.0) < 0.05
            else "no edge over random (as expected for a fair lottery)"
        )
        print(
            f"{k:>12}: avg {sig['observed_avg']:.3f} "
            f"(CI95 {sig['observed_avg_ci95'][0]:.3f}-{sig['observed_avg_ci95'][1]:.3f}) "
            f"vs random {RANDOM_EXPECTED_AVG:.3f} | p={sig['p_value_avg']:.3f} | {verdict}"
        )
    print('=' * 62)

    out_dir = Path('outputs/validation')
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f'validation_{method}_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Saved backtest results to {out_path}')

    # Plot
    if not no_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 7))
            x = list(range(len(dates)))
            # Matches over time
            ax1 = plt.subplot(2, 1, 1)
            for k, v in series.items():
                ax1.plot(x, v, label=k)
            ax1.set_title('Matches per draw over time')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Matches (0-6)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Cumulative average
            ax2 = plt.subplot(2, 1, 2)
            for k, v in series.items():
                if v:
                    cumavg = np.cumsum(v) / (np.arange(len(v)) + 1)
                    ax2.plot(x, cumavg, label=k)
            ax2.set_title('Cumulative average matches')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Avg matches')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            plot_path = out_dir / f'validation_plot_{method}_{ts}.png'
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close()
            logger.info(f'Saved backtest plot to {plot_path}')
            results['plot'] = str(plot_path)
        except Exception as e:
            logger.warning(f'Failed to create plot: {e}')

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Backtest Lotto predictions')
    parser.add_argument('--lookback', type=int, default=200)
    parser.add_argument('--method', choices=['frequency', 'weighted', 'consensus', 'probmap'], default='frequency')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--compare', type=str, default='random', help='Comma-separated baselines (e.g., random,frequency,probmap)')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--offline', action='store_true', help='Skip data download (reproducible runs)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--min-steps', type=int, default=30, help='Refuse to run with fewer evaluation draws')
    args = parser.parse_args()

    compare = [s.strip() for s in args.compare.split(',')] if args.compare else []
    run_backtest(
        lookback=args.lookback, method=args.method, step=args.step, compare=compare,
        no_plot=args.no_plot, offline=args.offline, seed=args.seed, min_steps=args.min_steps,
    )


if __name__ == '__main__':
    main()


