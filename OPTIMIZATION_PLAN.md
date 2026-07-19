## UK Lotto Prediction System – Optimization Plan

### Objectives
- Faster, deterministic predictions on real UK Lotto data only
- Robust data updates and schema integrity
- Clean model loading, shape-safe inputs, configurable ensembles
- Reproducible runs and measurable performance

### Phase 0 — Quick fixes (immediate) — Completed
- Fix QA report writer
  - Add `import json` at top of `scripts/fetch_data.py` to enable QA report write.
- Persist run metadata per execution
  - After each run, write `outputs/results/run_meta_YYYYMMDD_HHMMSS.json` with:
    - Data date range, row count, ensemble method, model names, model input shape, seed, duration, output paths

### Phase 1 — Data pipeline hardening — Completed
- Smarter download to avoid redundant fetches
  - Use `If-Modified-Since` and/or ETag; save headers to `data/.download_state.json`
  - Retries with backoff and strict timeouts
- Schema enforcement (already largely in place)
  - 6 unique ints [1..59] per draw; drop invalid rows; dedupe on `Draw Date`
  - Log conflicts during merge
- Directory hygiene
  - Move caches under `data/cache/` (or consistently under `outputs/results/`)
- QA JSON (per run)
  - `outputs/results/data_quality.json`: record count, date range, dropped counts, schema_ok

### Phase 2 — Model loading + input alignment — Completed
- Prefer SavedModel/H5 over pickle
  - Deprioritize `trained_models.pkl` due to Keras version coupling
- Model config file
  - `models/checkpoints/model_config.json`: `expected_timesteps`, `expected_features`, `normalize_by`, `name`
- Shape-safe inference (default)
  - Build `X` to match `expected_timesteps` and `expected_features` directly (no runtime padding where possible)
- Determinism
  - Add/propagate `--seed` in all entry scripts for numpy/random/tf

### Phase 3 — Ensemble & prediction strategy — Partially Completed
- CLI flag (already added): `--ensemble {frequency, weighted, consensus}`
- Probability-map combiner
  - Produce 59-length per-number scores, aggregate with weights, select top-6 with uniqueness + spacing constraints
- Portfolio diversification constraints
  - Maintain sum/odd-even/gap bands; reject perturbs violating bands

### Phase 4 — Backtesting and monitoring — Completed
- Rolling backtest (implemented)
  - `scripts/validations/backtest.py --lookback 200 --method frequency`
- Nightly cron (local)
  - Update, backtest K weeks, export plots/artifacts
- Metrics
  - Avg matches; partial_3+/4+/5+ rates; compare vs random baseline

### Phase 5 — Environment & dependency slimming — Completed
- Lightweight predict environment (Conda py3.11)
  - `tensorflow-macos==2.13.0`, `keras==2.13.1`, `numpy==1.24.3`, `pandas==2.0.3`, `scikit-learn==1.3.0`
- Avoid training-only imports in prediction path (optuna/lightgbm/catboost)

### Phase 6 — Run metadata & logs — Completed
- Save `outputs/results/run_meta_*.json` per run with: data dates, ensemble, models, input shape, seed, duration, output paths
- Include the last 3 draws for traceability

### Phase 7 — Optional improvements — Ongoing
 - Performance: cache prepared `X`, vectorize feature engineering
 - Validation: pytest for schema, input-shape builder, ensemble integrity
 - Artifact mgmt: versioned model dirs `models/checkpoints/vYYYYMMDD_HHMMSS/`

## Next Tasks (Actionable)

1) Real data backfill (priority)
- Expand `data/merged_lottery_data.csv` to full UK Lotto history.

2) Retrain + add a lightweight tree model
- Retrain LSTM with longer lookback (30–60), early stopping; save H5.
- Train XGBoost/LightGBM on engineered features; include in ensemble; compare via backtest.
- Persist and auto-load ensemble weights from recent window performance into `outputs/training/ensemble_weights.json`.

3) Portfolio optimizer (solver-grade)
- Replace greedy optimizer with OR-Tools CP-SAT constraints:
  - Per-number cap (e.g., ≤ 6/10)
  - Per-line: ≥1 high (≥50), odd/even balance, sum bands (e.g., 120–210), gap limits
  - Portfolio: decade quotas, pair/trio penalties, min Hamming distance
  - Keep wildcard lines configurable

4) Auto-use best ensemble method
- Main/start scripts should prefer `outputs/results/best_ensemble.json` unless overridden by CLI.

5) Multi-seed hedge (disjoint portfolios)
- Add `--multi-seed K` to generate K disjoint portfolios and enforce cross-portfolio non-overlap.

6) One-liners & docs
- Make targets: `evaluate-latest`, `predict-probmap`.
- README: cron/launchd snippet 30–60 min pre-draw.

7) Continuous evaluation & ROI
- After each draw, auto-ingest results, compute exact/partial hits, update rolling metrics, optionally track cost vs returns.


