"""A/B backtest driver: OR-levels stop vs 1.25x ATR(14) stop, both with 60/40 partial.

Runs the full 2-year backtest under both stop methods in sequence by mutating
CONFIG between runs. Saves a side-by-side comparison widget with both equity
curves overlaid and a metrics table.
"""
from __future__ import annotations

import json
import time
from copy import deepcopy

import pandas as pd

from strategies.orb_nq_vp.config import CONFIG
from strategies.orb_nq_vp.backtest import run_backtest
from strategies.orb_nq_vp.metrics import fmt_report


START_DATE = "2024-05-26"
END_DATE = "2026-05-22"

SCENARIOS = [
    {
        "name": "OR-levels stop",
        "label": "A: OR stop",
        "overrides": {"initial_stop_method": "or_levels"},
        "color": "#60a5fa",   # blue
    },
    {
        "name": "1.25 x ATR(14) stop",
        "label": "B: 1.25xATR",
        "overrides": {"initial_stop_method": "atr", "atr_multiple": 1.25, "atr_period": 14},
        "color": "#fbbf24",   # amber
    },
]


def _progress(scenario_name):
    def cb(i: int, n: int, d: str):
        print(f"  [{scenario_name}] [{i:4d}/{n}] {d}", flush=True)
    return cb


def _apply_overrides(overrides: dict):
    """Mutate CONFIG in place; return prior values for restore."""
    prior = {}
    for k, v in overrides.items():
        prior[k] = CONFIG.get(k)
        CONFIG[k] = v
    return prior


def _restore(prior: dict):
    for k, v in prior.items():
        CONFIG[k] = v


def _kpi_dict(result) -> dict:
    m = result.metrics
    atr_skips = sum(1 for s in result.sessions if "atr_unavailable" in s.outcome or "atr_extreme" in s.outcome)
    return {
        "trades": m.n_trades,
        "wins": m.n_wins,
        "losses": m.n_losses,
        "win_rate": m.win_rate,
        "net_pnl": m.net_pnl,
        "gross_profit": m.gross_profit,
        "gross_loss": m.gross_loss,
        "profit_factor": m.profit_factor,
        "avg_win": m.avg_win,
        "avg_loss": m.avg_loss,
        "expectancy": m.expectancy,
        "max_dd": m.max_drawdown_dollar,
        "max_dd_pct": m.max_drawdown_pct_of_account,
        "sharpe": m.sharpe_daily,
        "calmar": m.calmar,
        "avg_dur_min": m.avg_trade_minutes,
        "pct_confirming": m.pct_confirming_hit,
        "pct_partial": m.pct_partial_executed,
        "skip_no_signal": m.n_skip_no_signal,
        "skip_risk_too_small": m.n_skip_risk_too_small,
        "skip_atr": atr_skips,
    }


def _equity_series(result) -> tuple[list, list, list]:
    daily = result.daily_pnl.copy()
    labels = daily.index.strftime("%Y-%m-%d").tolist()
    equity = (CONFIG["account_size"] + daily.cumsum()).round(2).tolist()
    peak = pd.Series(equity).cummax()
    dd = (pd.Series(equity) - peak).round(2).tolist()
    return labels, equity, dd


def _fmt_money(v: float) -> str:
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.0f}"


def _print_comparison(results: list):
    rows = [
        ("Trades",                lambda k: f"{k['trades']}"),
        ("Wins / Losses",         lambda k: f"{k['wins']} / {k['losses']}"),
        ("Win rate",              lambda k: f"{k['win_rate']*100:.1f}%"),
        ("Gross profit",          lambda k: _fmt_money(k['gross_profit'])),
        ("Gross loss",            lambda k: _fmt_money(k['gross_loss'])),
        ("Net P&L",               lambda k: _fmt_money(k['net_pnl'])),
        ("Profit factor",         lambda k: f"{k['profit_factor']:.2f}"),
        ("Avg win",               lambda k: _fmt_money(k['avg_win'])),
        ("Avg loss",              lambda k: _fmt_money(k['avg_loss'])),
        ("Expectancy / trade",    lambda k: _fmt_money(k['expectancy'])),
        ("Max drawdown ($)",      lambda k: _fmt_money(k['max_dd'])),
        ("Max drawdown (% acct)", lambda k: f"{k['max_dd_pct']*100:.1f}%"),
        ("Sharpe (ann.)",         lambda k: f"{k['sharpe']:.2f}"),
        ("Calmar",                lambda k: f"{k['calmar']:.2f}"),
        ("Avg trade dur (min)",   lambda k: f"{k['avg_dur_min']:.1f}"),
        ("% w/ confirming",       lambda k: f"{k['pct_confirming']*100:.1f}%"),
        ("% w/ partial fill",     lambda k: f"{k['pct_partial']*100:.1f}%"),
        ("Skip: no_signal",       lambda k: f"{k['skip_no_signal']}"),
        ("Skip: risk_too_small",  lambda k: f"{k['skip_risk_too_small']}"),
        ("Skip: ATR-related",     lambda k: f"{k['skip_atr']}"),
    ]
    width_label = 26
    width_col = 18
    headers = ["Metric".ljust(width_label)] + [r["label"].ljust(width_col) for r in results]
    print()
    print("=" * (width_label + len(results) * width_col + 4))
    print(" " + " | ".join(headers))
    print("-" * (width_label + len(results) * width_col + 4))
    for label, fn in rows:
        cells = [label.ljust(width_label)] + [fn(r["kpis"]).ljust(width_col) for r in results]
        print(" " + " | ".join(cells))
    print("=" * (width_label + len(results) * width_col + 4))


def _build_widget(results: list):
    """Two-line equity curve + DD overlay + metrics table."""
    base_labels = results[0]["labels"]
    series = []
    for r in results:
        # Align all series to the first scenario's date index (they should match anyway).
        eq_map = dict(zip(r["labels"], r["equity"]))
        dd_map = dict(zip(r["labels"], r["drawdown"]))
        eq = [eq_map.get(d, None) for d in base_labels]
        dd = [dd_map.get(d, None) for d in base_labels]
        series.append({"label": r["label"], "color": r["color"], "equity": eq, "drawdown": dd,
                       "kpis": r["kpis"]})

    blob = json.dumps({
        "labels": base_labels,
        "series": series,
        "start": START_DATE,
        "end": END_DATE,
        "account": CONFIG["account_size"],
    })

    html = """
<style>
 .grid { display:grid; grid-template-columns: 2fr 1fr; gap:12px; }
 .panel { padding:10px; border:1px solid var(--color-border, #444); border-radius:6px; }
 h3 { margin: 4px 0 8px 0; color: var(--color-text-primary, #ddd); font-size:14px; }
 table { width:100%; border-collapse: collapse; font-size:12px;
         color: var(--color-text-primary, #ddd); font-family: ui-monospace, monospace; }
 th, td { padding:4px 8px; border-bottom: 1px solid rgba(128,128,128,0.18); text-align:right; }
 th:first-child, td:first-child { text-align:left; color: var(--color-text-secondary, #aaa); }
 .pos { color: #4ade80; } .neg { color: #f87171; }
</style>
<div class="grid">
  <div class="panel">
    <h3>Equity curves: OR-stop vs 1.25xATR-stop</h3>
    <div style="position:relative;width:100%"><canvas id="eq"></canvas></div>
  </div>
  <div class="panel">
    <h3>Side-by-side metrics</h3>
    <table id="kpi"></table>
  </div>
  <div class="panel" style="grid-column: 1 / span 2;">
    <h3>Drawdown ($)</h3>
    <div style="position:relative;width:100%"><canvas id="dd"></canvas></div>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script>
const D = __BLOB__;
const txt = getComputedStyle(document.body).getPropertyValue('--color-text-primary') || '#ddd';
const grid = 'rgba(128,128,128,0.18)';
Chart.defaults.color = txt;
Chart.defaults.borderColor = grid;

const fmtMoney = (v) => (v < 0 ? '-' : '') + '$' + Math.abs(v).toLocaleString(undefined,{maximumFractionDigits:0});
const fmtPct = (v) => (v*100).toFixed(1) + '%';
const fmtNum = (v, d=2) => Number(v).toFixed(d);

const rows = [
  ['Trades',              s => s.kpis.trades],
  ['Wins / Losses',       s => s.kpis.wins + ' / ' + s.kpis.losses],
  ['Win rate',            s => fmtPct(s.kpis.win_rate)],
  ['Net P&L',             s => fmtMoney(s.kpis.net_pnl), true],
  ['Profit factor',       s => fmtNum(s.kpis.profit_factor)],
  ['Avg win',             s => fmtMoney(s.kpis.avg_win)],
  ['Avg loss',            s => fmtMoney(s.kpis.avg_loss)],
  ['Expectancy / trade',  s => fmtMoney(s.kpis.expectancy), true],
  ['Max DD ($)',          s => fmtMoney(s.kpis.max_dd)],
  ['Max DD (% acct)',     s => fmtPct(s.kpis.max_dd_pct)],
  ['Sharpe (ann.)',       s => fmtNum(s.kpis.sharpe), true],
  ['Calmar',              s => fmtNum(s.kpis.calmar), true],
  ['Avg dur (min)',       s => fmtNum(s.kpis.avg_dur_min, 1)],
  ['% confirming',        s => fmtPct(s.kpis.pct_confirming)],
  ['% partial',           s => fmtPct(s.kpis.pct_partial)],
  ['Skip: no_signal',     s => s.kpis.skip_no_signal],
  ['Skip: risk_too_small',s => s.kpis.skip_risk_too_small],
  ['Skip: ATR-related',   s => s.kpis.skip_atr],
];

let html = '<thead><tr><th>Metric</th>';
for (const s of D.series) html += `<th style="color:${s.color}">${s.label}</th>`;
html += '</tr></thead><tbody>';
for (const [label, fn, signed] of rows) {
  html += `<tr><td>${label}</td>`;
  for (const s of D.series) {
    const v = fn(s);
    let cls = '';
    if (signed) {
      const num = (typeof v === 'string') ? parseFloat(v.replace(/[^0-9.-]/g,'')) : v;
      cls = num > 0 ? 'pos' : (num < 0 ? 'neg' : '');
    }
    html += `<td class="${cls}">${v}</td>`;
  }
  html += '</tr>';
}
html += '</tbody>';
document.getElementById('kpi').innerHTML = html;

new Chart(document.getElementById('eq'), {
  type: 'line',
  data: {
    labels: D.labels,
    datasets: D.series.map(s => ({
      label: s.label, data: s.equity, borderColor: s.color, borderWidth: 1.5,
      pointRadius: 0, tension: 0.05, fill: false,
    })),
  },
  options: { aspectRatio: 2.2, plugins: { legend: { display: true }},
    scales: { x: { ticks: { maxTicksLimit: 8 }, grid:{color:grid}},
              y: { grid:{color:grid}, ticks: { callback: v => '$'+v.toLocaleString() }}}}
});
new Chart(document.getElementById('dd'), {
  type: 'line',
  data: {
    labels: D.labels,
    datasets: D.series.map(s => ({
      label: s.label, data: s.drawdown, borderColor: s.color, borderWidth: 1.2,
      pointRadius: 0, tension: 0, fill: false,
    })),
  },
  options: { aspectRatio: 3.5, plugins: { legend: { display: true }},
    scales: { x: { ticks: { maxTicksLimit: 8 }, grid:{color:grid}},
              y: { grid:{color:grid}, ticks: { callback: v => '$'+v.toLocaleString() }}}}
});
</script>
"""
    save_widget(html.replace("__BLOB__", blob), "ab_stop_method_comparison")


def main():
    snapshot = deepcopy(CONFIG)
    print(f"[A/B] window: {START_DATE} -> {END_DATE}")
    print(f"[A/B] common config: account=${CONFIG['account_size']:,.0f} risk={CONFIG['risk_pct_per_trade']*100:.1f}% "
          f"trail={CONFIG['trail_method']} stop_trig={CONFIG['stop_trigger_mode']} "
          f"vp_bin={CONFIG['vp_bin_size_ticks']}t partial={CONFIG['partial_exit_fraction']*100:.0f}%")

    results = []
    for sc in SCENARIOS:
        print(f"\n[A/B] --- running scenario: {sc['name']} ---")
        print(f"[A/B] overrides: {sc['overrides']}")
        prior = _apply_overrides(sc["overrides"])
        t0 = time.time()
        result = run_backtest(START_DATE, END_DATE, progress_callback=_progress(sc["label"]))
        dt = time.time() - t0
        print(f"[A/B] {sc['label']} complete in {dt:.1f}s")
        print(fmt_report(result.metrics))

        labels, equity, dd = _equity_series(result)
        results.append({
            "name": sc["name"], "label": sc["label"], "color": sc["color"],
            "kpis": _kpi_dict(result),
            "labels": labels, "equity": equity, "drawdown": dd,
            "trade_results": result.trade_results,  # kept for diagnostic phase
        })
        _restore(prior)

    _print_comparison(results)
    _build_widget(results)
    _save_scenario_trade_csvs(results)
    _print_diagnostics(results)
    _build_diagnostic_widget(results)
    print("\n[A/B] comparison widget saved.")
    # Sanity: restore original CONFIG state (paranoia)
    for k, v in snapshot.items():
        CONFIG[k] = v


if __name__ == "__main__":
    main()