#!/usr/bin/env python3
"""
Evaluate 10 AI predictions vs 10 random and a 'repeating/hot' baseline
against the most recent actual draw.

- AI: uses current ensemble (auto-select best if available)
- Random: uniform random combinations
- Hot (repeating numbers): top-6 most frequent numbers in recent history

Outputs a JSON summary and prints a concise table.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np

from scripts.fetch_data import DATA_DIR, load_data, download_fresh_data
from scripts.new_predict import (
    load_models,
    create_ensemble,
    prepare_input_data,
    generate_predictions,
)


OUT_DIR = Path('outputs/validation')


def match_count(pred: List[int], actual: List[int]) -> int:
    return len(set(pred).intersection(set(actual)))


def top6_hot_numbers(df, window: int | None = None) -> List[int]:
    if window is None:
        window = min(200, len(df))
    recent = df.tail(window)
    counts: Dict[int, int] = {}
    for row in recent['Main_Numbers']:
        for n in row:
            counts[int(n)] = counts.get(int(n), 0) + 1
    # Pick top 6
    hot = [n for n, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:6]]
    return sorted(hot)


def choose_best_ensemble_default() -> str:
    best_path = Path('outputs/results/best_ensemble.json')
    if best_path.exists():
        try:
            best = json.load(open(best_path, 'r'))
            m = best.get('best_method')
            if m in {'frequency', 'weighted', 'consensus', 'probmap'}:
                return m
        except Exception:
            pass
    return 'frequency'


def main() -> int:
    # Ensure freshest data
    try:
        download_fresh_data()
    except Exception:
        pass

    df = load_data(DATA_DIR / 'merged_lottery_data.csv')
    last_draw = df['Main_Numbers'].iloc[-1]

    # AI predictions
    models = load_models()
    if not models:
        raise SystemExit('No trained models available')
    ensemble = create_ensemble(models)
    method = choose_best_ensemble_default()
    ai_preds = generate_predictions(ensemble, df, count=10, diversity=0.35, combination_method=method)

    # Random predictions
    rnd_preds = [sorted(random.sample(range(1, 60), 6)) for _ in range(10)]

    # Hot/repeating baseline (top-6 most frequent)
    hot = top6_hot_numbers(df)
    hot_preds = [hot[:] for _ in range(10)]

    # Evaluate
    ai_matches = [match_count(p, last_draw) for p in ai_preds]
    rnd_matches = [match_count(p, last_draw) for p in rnd_preds]
    hot_matches = [match_count(p, last_draw) for p in hot_preds]

    def summary(ms: List[int]) -> Dict[str, float]:
        arr = np.array(ms, dtype=float)
        return {
            'avg_matches': float(arr.mean()),
            'partial_3plus_rate': float((arr >= 3).mean()),
            'partial_4plus_rate': float((arr >= 4).mean()),
            'partial_5plus_rate': float((arr >= 5).mean()),
        }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'evaluate_latest_{ts}.json'
    payload = {
        'timestamp': datetime.now().isoformat(),
        'last_draw': last_draw,
        'ensemble_method': method,
        'ai': {
            'predictions': ai_preds,
            'matches': ai_matches,
            'metrics': summary(ai_matches),
        },
        'random': {
            'predictions': rnd_preds,
            'matches': rnd_matches,
            'metrics': summary(rnd_matches),
        },
        'hot_repeating': {
            'hot_numbers': hot,
            'predictions': hot_preds,
            'matches': hot_matches,
            'metrics': summary(hot_matches),
        },
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)

    def fmt(d: Dict[str, float]) -> str:
        return (
            f"avg={d['avg_matches']:.2f}, 3+={d['partial_3plus_rate']:.2f}, "
            f"4+={d['partial_4plus_rate']:.2f}, 5+={d['partial_5plus_rate']:.2f}"
        )

    print("\nEVALUATE LATEST (10 predictions vs last draw)")
    print(f"Last draw: {sorted(last_draw)}\n")
    print(f"AI ({method}):   {fmt(payload['ai']['metrics'])}")
    print(f"Random:       {fmt(payload['random']['metrics'])}")
    print(f"Hot/repeating:{fmt(payload['hot_repeating']['metrics'])}")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


