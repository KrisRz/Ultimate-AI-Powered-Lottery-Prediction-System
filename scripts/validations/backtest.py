"""Significance of a match series against the no-skill null.

What is left of the rolling backtest after the LSTM predictor it drove was
removed (audit item C1): the statistics, which scripts/validations/
ensemble_score.py now answers the same question with - does any way of
picking numbers beat a random pick? It has not (full-pipeline null p=0.600).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def calc_match_counts(pred: List[int], actual: List[int]) -> int:
    return len(set(pred).intersection(set(actual)))


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
