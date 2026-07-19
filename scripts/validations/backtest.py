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


def run_backtest(
    lookback: int = 200,
    method: str = 'frequency',
    step: int = 1,
    compare: Sequence[str] | None = None,
    no_plot: bool = False,
) -> Dict:
    # Ensure we have freshest data
    try:
        download_fresh_data()
    except Exception as e:
        logger.warning(f"Data update failed, proceeding with existing data: {e}")

    df = load_data(DATA_DIR / 'merged_lottery_data.csv')
    if len(df) < lookback + 1:
        new_lookback = max(10, len(df) - 1)
        logger.info(f"Adjusting lookback from {lookback} to {new_lookback} due to limited data (have {len(df)})")
        lookback = new_lookback

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
        'metrics': {k: _rates(v) for k, v in series.items()},
        'series': {k: [{'date': d, 'matches': int(m)} for d, m in zip(dates, v)] for k, v in series.items()},
    }

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
    args = parser.parse_args()

    compare = [s.strip() for s in args.compare.split(',')] if args.compare else []
    run_backtest(lookback=args.lookback, method=args.method, step=args.step, compare=compare, no_plot=args.no_plot)


if __name__ == '__main__':
    main()


