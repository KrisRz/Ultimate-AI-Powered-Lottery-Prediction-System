/**
 * Shape of site/public/data/site.json.
 *
 * Written by scripts/export_site_data.py. Keep this in step with the schema
 * there - the exporter is the source of truth, and its tests assert the
 * invariants these types cannot (that `b` stays unrounded, that break-even is
 * really -a/b, that the popularity examples match the live model).
 */

export interface Snapshot {
  as_of_draw_date: string;
  data_through: string;
  draws_all_time: number;
  draws_59_ball_era: number;
}

export interface Odds {
  key: string;
  label: string;
  probability: number;
  one_in: number;
}

export interface Hook {
  total_combinations: number;
  n_balls: number;
  n_pick: number;
  rounds_per_draw: number;
  ticket_price_gbp: number;
  odds: Odds[];
}

export interface BacktestMethod {
  name: string;
  is_baseline: boolean;
  observed_avg: number;
  ci95: [number, number];
  p_value_avg: number;
  rate_3plus: number;
  p_value_3plus: number;
  beats_random: boolean;
}

export interface Backtest {
  steps: number;
  lookback: number;
  n_sim: number;
  expected_random_avg: number;
  date_from: string;
  date_to: string;
  methods: BacktestMethod[];
  series: {
    dates: string[];
    cumulative_avg: Record<string, number[]>;
  };
}

/** EV(J) = a + b*J. Exact, not a sampled curve - see `affine` in the exporter. */
export interface Regime {
  key: 'ordinary' | 'mbw';
  label: string;
  tickets_sold: number;
  roll_down: boolean;
  rounds: number;
  a: number;
  b: number;
  break_even_jackpot: number;
}

export interface SalesBandPoint {
  uplift: number;
  tickets_sold: number;
  a: number;
  b: number;
  break_even_jackpot: number;
}

export interface Ev {
  reference_line: number[];
  reference_popularity: number;
  fixed_prizes: {
    match_5_bonus: number;
    match_5: number;
    match_4: number;
    match_3: number;
    match_2: number;
    source: string;
    ev_per_round: number;
  };
  slider: { min_gbp: number; max_gbp: number; step_gbp: number };
  regimes: Regime[];
  mbw_sales_band: { p25: SalesBandPoint; p75: SalesBandPoint };
  live: {
    draw_date: string;
    jackpot_gbp: number;
    tickets_sold: number;
    roll_down: boolean;
    rollover_count: number;
    rollover_cap: number;
    mbw_type: string | null;
    ev_best_line: number;
    break_even_jackpot: number;
    verdict: 'PLAY' | 'SKIP';
    robust: boolean;
  };
}

export interface Popularity {
  recovered: number[];
  recovered_range: [number, number];
  n_observations: number;
  installed_step: { from: number; to: number; weight: number }[];
  model: {
    mean_weight: number;
    normalization: number;
    arithmetic_mult: number;
    consecutive_mult: number;
    birthday_mult: number;
    consecutive_run: number;
    birthday_max: number;
  };
  most_played: { number: number; weight: number }[];
  least_played: { number: number; weight: number }[];
  examples: { line: number[]; ratio: number; note: string }[];
  match3_multiplier_by_low31: {
    n_low31: number;
    draws: number;
    mean_multiplier: number;
  }[];
}

export interface LastDraw {
  draw_number: number;
  draw_date: string;
  numbers: number[];
  bonus: number | null;
  jackpot_winners: number;
  jackpot_per_winner_gbp: number;
  jackpot_total_gbp: number;
  was_shared: boolean;
}

export interface Ledger {
  first_ticket_date: string;
  last_draw_date: string;
  lines: number;
  settled: number;
  spent_gbp: number;
  won_gbp: number;
  net_gbp: number;
  roi: number | null;
  match_histogram: Record<string, number>;
  source: string;
}

export interface SiteData {
  schema_version: number;
  snapshot: Snapshot;
  hook: Hook;
  backtest: Backtest;
  ev: Ev;
  popularity: Popularity;
  last_draw: LastDraw;
  ledger: Ledger | null;
  rolldown: Rolldown;
  wheel: Wheel;
  built: Built;
}

export interface Rolldown {
  rule: string;
  rollover_cap: number;
  split: {
    basis: string;
    jackpot_gbp: number;
    tickets_sold: number;
    rounds: number;
    match_2_boost: number;
    match_3_boost: number;
    expected_match_2_winners: number;
    expected_match_3_winners: number;
    match_2_total_gbp: number;
    match_3_total_gbp: number;
  };
  history: {
    detected: number;
    cap_driven: number;
    special_event: number;
    window: [string, string];
    per_year: number;
    positive_ev: number;
    positive_ev_share: number;
    median_ev: number;
    ev_quartiles: [number, number];
    median_pool_gbp: number;
    median_tickets: number;
    caveat: string;
    draws: {
      draw_number: number;
      date: string;
      pool_gbp: number;
      tickets_sold: number;
      cap_driven: boolean;
      ev: number;
    }[];
  };
}

export interface Wheel {
  pool: number[];
  pool_size: number;
  lines: number[][];
  line_popularity: number;
  guarantees: Record<string, number>;
  line_ev: number;
  backtest: {
    draws: number;
    hits_total: number;
    hits_by_tier: Record<string, number>;
    cash_gbp: number;
    cost_gbp: number;
    return_pct: number | null;
    guarantee_violations: number;
    draws_with_win: number;
    clump_variance: number;
  };
}

export interface Built {
  workflows: { name: string; file: string; schedule: string[]; does: string }[];
  tests: { count: number; files: number };
  datastore: string;
  alerts: { transport: string; default: string; fires_on: string };
  self_healing: { window_days: number; how: string };
  freshness_gate: string;
  scheduler_caveat: string;
  hosting: Record<string, string>;
}
