#!/usr/bin/env python3
"""Generate the decision dashboard: outputs/dashboard.html.

A self-contained static page (no external assets, light/dark aware) that
answers, in order: should we play the next draw, with which lines, what has
actually happened to our money, and does any method beat random.

Data sources (all optional - missing ones render as "no data yet"):
  data/merged_lottery_data.csv           draw history / freshness
  data/prize_tiers.csv                   next-draw jackpot & roll-down flag
  outputs/predictions/latest.json        current EV portfolio
  data/ledger.csv                        real money ledger
  outputs/validation/validation_*.json   newest backtest + significance
"""

import html
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from lottery.ev import DrawConditions, line_ev, popularity_ratio, should_play  # noqa: E402
from scripts.ev_play import next_draw_conditions  # noqa: E402

OUT_FILE = Path("outputs/dashboard.html")
MERGED_FILE = Path("data/merged_lottery_data.csv")
PRIZE_TIERS_FILE = Path("data/prize_tiers.csv")
LATEST_PREDICTIONS = Path("outputs/predictions/latest.json")
LEDGER_FILE = Path("data/ledger.csv")
VALIDATION_DIR = Path("outputs/validation")


def esc(v) -> str:
    return html.escape(str(v))


def gather() -> dict:
    d: dict = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M")}

    if MERGED_FILE.exists():
        df = pd.read_csv(MERGED_FILE)
        d["draws"] = len(df)
        d["last_draw"] = df["Draw Date"].max()
        d["stale_days"] = (datetime.now() - pd.to_datetime(d["last_draw"])).days

    # Single source of truth for next-draw conditions - the same function the
    # advisor and the email alert use (jackpot, MBW flag, estimated sales).
    cond = next_draw_conditions()
    d["cond"] = cond
    d["verdict"] = should_play(cond, threshold=0.0)

    if LATEST_PREDICTIONS.exists():
        payload = json.load(open(LATEST_PREDICTIONS))
        d["portfolio_date"] = payload.get("date", "?")
        d["portfolio_method"] = payload.get("metadata", {}).get("method", "?")
        # Recompute EV/popularity under TODAY's conditions - stored values may
        # come from a what-if run with a different jackpot
        d["portfolio"] = [{
            "line": sorted(int(n) for n in p),
            "ev": line_ev(p, cond),
            "popularity_ratio": popularity_ratio(p),
        } for p in payload.get("predictions", [])]

    if LEDGER_FILE.exists():
        ledger = pd.read_csv(LEDGER_FILE)
        settled = ledger[ledger["settled"] == True]  # noqa: E712
        d["ledger"] = {
            "lines": len(ledger),
            "settled": len(settled),
            "spent": float(ledger["cost"].sum()),
            "won": float(settled["prize"].fillna(0).sum()),
        }

    # Newest run BY MTIME that carries the significance format (older files
    # sort after newer ones alphabetically and lack the field)
    runs = sorted(VALIDATION_DIR.glob("validation_*.json"),
                  key=lambda p: p.stat().st_mtime,
                  reverse=True) if VALIDATION_DIR.exists() else []
    for run in runs:
        bt = json.load(open(run))
        if bt.get("significance") and bt.get("series"):
            d["backtest"] = bt
            break

    return d


def _cumavg(values):
    out, total = [], 0.0
    for i, v in enumerate(values, 1):
        total += v
        out.append(total / i)
    return out


def chart_svg(bt: dict) -> str:
    """Cumulative-average-matches line chart: method (blue) vs random (gray)."""
    series = bt.get("series", {})
    method = bt.get("method", "method")
    if method not in series or "random" not in series:
        return ""
    m_vals = _cumavg([p["matches"] for p in series[method]])
    r_vals = _cumavg([p["matches"] for p in series["random"]])
    dates = [p["date"] for p in series[method]]
    n = len(m_vals)
    if n < 2:
        return ""

    W, H, PAD_L, PAD_R, PAD_T, PAD_B = 640, 220, 40, 90, 16, 26
    y_max = max(1.0, max(m_vals + r_vals) * 1.15)
    expected = 36.0 / 59.0

    def x(i):
        return PAD_L + i * (W - PAD_L - PAD_R) / (n - 1)

    def y(v):
        return PAD_T + (1 - v / y_max) * (H - PAD_T - PAD_B)

    def pts(vals):
        return " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))

    gridlines = "".join(
        f'<line x1="{PAD_L}" y1="{y(g):.1f}" x2="{W - PAD_R}" y2="{y(g):.1f}" class="grid"/>'
        f'<text x="{PAD_L - 6}" y="{y(g) + 4:.1f}" class="tick" text-anchor="end">{g:.1f}</text>'
        for g in (0.0, 0.5, 1.0) if g <= y_max
    )
    data_json = esc(json.dumps({"dates": dates, method: [round(v, 3) for v in m_vals],
                                "random": [round(v, 3) for v in r_vals]}))
    return f"""
<div class="chart-wrap" data-chart='{data_json}'>
<svg viewBox="0 0 {W} {H}" role="img" aria-label="Cumulative average matches: {esc(method)} vs random">
  {gridlines}
  <line x1="{PAD_L}" y1="{y(expected):.1f}" x2="{W - PAD_R}" y2="{y(expected):.1f}" class="expected"/>
  <polyline points="{pts(r_vals)}" class="line-random"/>
  <polyline points="{pts(m_vals)}" class="line-method"/>
  <text x="{W - PAD_R + 6}" y="{y(m_vals[-1]) + 4:.1f}" class="lbl-method">{esc(method)}</text>
  <text x="{W - PAD_R + 6}" y="{y(r_vals[-1]) + 16:.1f}" class="lbl-muted">random</text>
  <line class="cursor" x1="0" y1="{PAD_T}" x2="0" y2="{H - PAD_B}" style="display:none"/>
</svg>
<div class="tooltip" style="display:none"></div>
<div class="legend">
  <span><i class="swatch sw-method"></i>{esc(method)}</span>
  <span><i class="swatch sw-random"></i>random baseline</span>
  <span><i class="swatch sw-expected"></i>no-skill mean ({expected:.2f})</span>
  <span class="muted">cumulative average matches per draw</span>
</div>
</div>
<script>
(function() {{
  const wrap = document.querySelector('.chart-wrap');
  if (!wrap) return;
  const data = JSON.parse(wrap.dataset.chart);
  const svg = wrap.querySelector('svg');
  const tip = wrap.querySelector('.tooltip');
  const cursor = wrap.querySelector('.cursor');
  const n = data.dates.length, PAD_L = {PAD_L}, PAD_R = {PAD_R}, W = {W};
  const keys = Object.keys(data).filter(k => k !== 'dates');
  svg.addEventListener('mousemove', (e) => {{
    const r = svg.getBoundingClientRect();
    const xs = (e.clientX - r.left) * (W / r.width);
    let i = Math.round((xs - PAD_L) / ((W - PAD_L - PAD_R) / (n - 1)));
    i = Math.max(0, Math.min(n - 1, i));
    const px = PAD_L + i * (W - PAD_L - PAD_R) / (n - 1);
    cursor.setAttribute('x1', px); cursor.setAttribute('x2', px);
    cursor.style.display = '';
    tip.style.display = '';
    tip.style.left = Math.min(e.clientX - r.left + 12, r.width - 150) + 'px';
    tip.style.top = '10px';
    tip.innerHTML = '<b>' + data.dates[i] + '</b><br>' +
      keys.map(k => k + ': ' + data[k][i].toFixed(3)).join('<br>');
  }});
  svg.addEventListener('mouseleave', () => {{
    tip.style.display = 'none'; cursor.style.display = 'none';
  }});
}})();
</script>"""


def render(d: dict) -> str:
    cond: DrawConditions = d["cond"]
    v = d["verdict"]
    play = v["play"]
    verdict_cls = "good" if play else "bad"
    verdict_icon = "&#9654;" if play else "&#9209;"
    verdict_word = "PLAY" if play else "SKIP"
    verdict_note = (
        "EV clears the threshold - rare conditions, see portfolio below"
        if play else
        "negative EV at your threshold (normal for ordinary draws) - do not play"
    )

    tiles = f"""
  <div class="tile {verdict_cls}">
    <div class="tile-label">Next draw verdict</div>
    <div class="tile-value">{verdict_icon} {verdict_word}</div>
    <div class="tile-note">{esc(verdict_note)}</div>
  </div>
  <div class="tile">
    <div class="tile-label">Best-line EV (2 rounds, &pound;2 ticket)</div>
    <div class="tile-value">&pound;{v['ev_best_line']:+.2f}</div>
    <div class="tile-note">jackpot &pound;{cond.jackpot:,.0f} (event pool) &middot; {cond.tickets_sold:,} lines sold &middot; {'Must-Be-Won' if cond.roll_down else 'normal draw'}</div>
  </div>"""

    if "ledger" in d:
        led = d["ledger"]
        net = led["won"] - led["spent"]
        roi = f"{100 * net / led['spent']:+.1f}%" if led["spent"] else "-"
        tiles += f"""
  <div class="tile {'good' if net >= 0 else 'bad'}">
    <div class="tile-label">Real money (ledger)</div>
    <div class="tile-value">&pound;{net:+.2f}</div>
    <div class="tile-note">spent &pound;{led['spent']:.2f} &middot; won &pound;{led['won']:.2f} &middot; ROI {roi} &middot; {led['settled']}/{led['lines']} settled</div>
  </div>"""
    else:
        tiles += """
  <div class="tile">
    <div class="tile-label">Real money (ledger)</div>
    <div class="tile-value">-</div>
    <div class="tile-note">no lines recorded yet - scripts/roi_ledger.py add</div>
  </div>"""

    if "draws" in d:
        stale = d.get("stale_days", 0)
        tiles += f"""
  <div class="tile {'warn' if stale > 4 else ''}">
    <div class="tile-label">Data</div>
    <div class="tile-value">{d['draws']}</div>
    <div class="tile-note">draws (59-ball era) &middot; last {esc(d['last_draw'])}{f' &middot; {stale}d old' if stale > 4 else ''}</div>
  </div>"""

    portfolio_rows = ""
    for i, p in enumerate(d.get("portfolio", []), 1):
        nums = " ".join(f"{n:02d}" for n in p["line"])
        ev = f"&pound;{p['ev']:+.3f}" if p.get("ev") is not None else "-"
        pop = f"&times;{p['popularity_ratio']:.2f}" if p.get("popularity_ratio") is not None else "-"
        portfolio_rows += f"<tr><td>{i}</td><td class='mono'>{nums}</td><td>{ev}</td><td>{pop}</td></tr>"
    portfolio_html = f"""
<h2>Current portfolio <span class="muted">({esc(d.get('portfolio_method', '?'))}, {esc(d.get('portfolio_date', '?'))})</span></h2>
<table>
  <thead><tr><th>#</th><th>Line</th><th>EV</th><th>Popularity vs avg player</th></tr></thead>
  <tbody>{portfolio_rows or '<tr><td colspan=4>no portfolio - SKIP verdict builds none (python scripts/ev_play.py --force to preview lines)</td></tr>'}</tbody>
</table>"""

    backtest_html = "<h2>Backtest vs random</h2><p class='muted'>no backtest yet - run: make backtest</p>"
    if "backtest" in d:
        bt = d["backtest"]
        rows = ""
        for name, sig in (bt.get("significance") or {}).items():
            if not sig:
                continue
            p_val = sig.get("p_value_avg")
            verdict = "no edge" if (p_val is None or p_val >= 0.05) else "check it!"
            rows += (
                f"<tr><td>{esc(name)}</td><td>{sig['observed_avg']:.3f}</td>"
                f"<td>{sig['observed_avg_ci95'][0]:.3f}&ndash;{sig['observed_avg_ci95'][1]:.3f}</td>"
                f"<td>{sig['expected_random_avg']:.3f}</td><td>{p_val:.3f}</td><td>{verdict}</td></tr>"
            )
        backtest_html = f"""
<h2>Backtest vs random <span class="muted">({bt.get('steps', '?')} draws, lookback {bt.get('lookback', '?')})</span></h2>
{chart_svg(bt)}
<table>
  <thead><tr><th>Method</th><th>Avg matches</th><th>CI95</th><th>No-skill mean</th><th>p-value</th><th>Verdict</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
<p class="muted">The honest baseline: a fair lottery admits no predictive edge. This table exists to keep us honest, not to find magic.</p>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lotto EV Dashboard</title>
<style>
  :root {{
    color-scheme: light dark;
    --surface: #fcfcfb; --surface-2: #f1f1ee;
    --ink: #0b0b0b; --ink-2: #52514e; --border: #dddcd6;
    --method: #2a78d6; --random: #8a887f;
    --good: #008300; --bad: #b3261e; --warn: #a06000;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --surface: #1a1a19; --surface-2: #242422;
      --ink: #ffffff; --ink-2: #c3c2b7; --border: #3a3936;
      --method: #3987e5; --random: #8a887f;
      --good: #58b158; --bad: #e66767; --warn: #c98500;
    }}
  }}
  body {{ margin: 0 auto; max-width: 900px; padding: 24px 16px 60px;
         font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: var(--surface); color: var(--ink); }}
  h1 {{ font-size: 22px; margin: 0 0 2px; }}
  h2 {{ font-size: 17px; margin: 34px 0 10px; }}
  .muted {{ color: var(--ink-2); font-weight: normal; font-size: 13px; }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 10px; margin-top: 18px; }}
  .tile {{ background: var(--surface-2); border: 1px solid var(--border);
           border-radius: 10px; padding: 12px 14px; }}
  .tile-label {{ font-size: 12px; color: var(--ink-2); text-transform: uppercase;
                 letter-spacing: 0.04em; }}
  .tile-value {{ font-size: 26px; font-weight: 650; margin: 2px 0; }}
  .tile-note {{ font-size: 12.5px; color: var(--ink-2); }}
  .tile.good .tile-value {{ color: var(--good); }}
  .tile.bad .tile-value {{ color: var(--bad); }}
  .tile.warn .tile-value {{ color: var(--warn); }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }}
  th {{ font-size: 12px; color: var(--ink-2); text-transform: uppercase; letter-spacing: 0.04em; }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .chart-wrap {{ position: relative; margin: 6px 0 4px; }}
  svg {{ width: 100%; height: auto; display: block; }}
  .grid {{ stroke: var(--border); stroke-width: 1; }}
  .tick, .lbl-muted {{ font-size: 11px; fill: var(--ink-2); }}
  .lbl-method {{ font-size: 11px; fill: var(--method); font-weight: 600; }}
  .expected {{ stroke: var(--ink-2); stroke-width: 1; stroke-dasharray: 2 4; }}
  .line-method {{ fill: none; stroke: var(--method); stroke-width: 2;
                  stroke-linejoin: round; stroke-linecap: round; }}
  .line-random {{ fill: none; stroke: var(--random); stroke-width: 2;
                  stroke-dasharray: 5 4; stroke-linejoin: round; }}
  .cursor {{ stroke: var(--ink-2); stroke-width: 1; }}
  .tooltip {{ position: absolute; background: var(--surface-2); color: var(--ink);
              border: 1px solid var(--border); border-radius: 6px; padding: 6px 9px;
              font-size: 12px; pointer-events: none; }}
  .legend {{ display: flex; gap: 18px; font-size: 12.5px; color: var(--ink-2);
             margin-top: 2px; flex-wrap: wrap; }}
  .swatch {{ display: inline-block; width: 14px; height: 3px; border-radius: 2px;
             vertical-align: middle; margin-right: 5px; }}
  .sw-method {{ background: var(--method); }}
  .sw-random {{ background: var(--random); }}
  .sw-expected {{ background: repeating-linear-gradient(90deg, var(--ink-2) 0 2px, transparent 2px 5px); }}
</style>
</head>
<body>
<h1>Lotto EV Dashboard</h1>
<div class="muted">UK Lotto 6/59, two rounds per draw &middot; generated {esc(d['generated'])}</div>
<div class="tiles">{tiles}
</div>
{portfolio_html}
{backtest_html}
<h2>How to use this</h2>
<p class="muted">
1. <span class="mono">make play</span> - EV verdict &amp; portfolio for the next draw.
2. If (and only if) it says PLAY and you choose to spend the money:
<span class="mono">python scripts/roi_ledger.py add --from-latest</span>.
3. After the draw: <span class="mono">make roi</span> settles lines against real results.
4. <span class="mono">make dashboard</span> regenerates this page.
No system can predict lottery numbers; this page optimizes when and what to play, and never lies about results.
</p>
</body>
</html>"""


def main() -> None:
    d = gather()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(render(d))
    print(f"Dashboard written to {OUT_FILE}")


if __name__ == "__main__":
    main()
