"""Full backtest driver.

Usage (edit constants at top, then `execute_file run_backtest.py`):
  START_DATE, END_DATE — backtest window (ISO)
  Outputs:
    - text report printed to stdout
    - .runs/.../artifacts/equity_curve.html — interactive widget
    - .runs/.../artifacts/per_session.csv  — session-level CSV
    - .runs/.../artifacts/trades.csv       — trade-level CSV
"""
from __future__ import annotations

import json
import time

import pandas as pd

from strategies.orb_nq_vp.backtest import run_backtest
from strategies.orb_nq_vp.metrics import fmt_report
from strategies.orb_nq_vp.config import CONFIG


START_DATE = "2024-05-26"   # full 2-year window
END_DATE = "2026-05-22"     # last full session in sanity window


def _progress(i: int, n: int, d: str):
    print(f"  [{i:4d}/{n}] {d}", flush=True)


def _save_csvs(result, out_dir: str):
    import os
    os.makedirs(out_dir, exist_ok=True)
    sess_df = result.to_dataframe()
    sess_path = f"{out_dir}/per_session.csv"
    sess_df.to_csv(sess_path, index=False)

    trade_rows = []
    for t in result.trade_results:
        trade_rows.append({
            "date": t.signal.entry_ts.tz_convert(CONFIG["session_tz"]).strftime("%Y-%m-%d"),
            "side": t.signal.side,
            "entry_ts": t.signal.entry_ts.isoformat(),
            "entry_price": t.signal.entry_price,
            "or_high": t.signal.or_high,
            "or_low": t.signal.or_low,
            "or_range": t.signal.or_range,
            "contracts": t.sizing.contracts,
            "partial_contracts": t.sizing.partial_contracts,
            "exit_reason": t.exit_reason,
            "pnl_gross": t.pnl_gross,
            "pnl_net": t.pnl_net,
            "confirming_hit": t.confirming is not None,
            "n_fills": len(t.fills),
            "final_stop": t.final_stop_at_exit,
            "final_state": t.final_state_at_exit,
        })
    trades_df = pd.DataFrame(trade_rows)
    trades_path = f"{out_dir}/trades.csv"
    trades_df.to_csv(trades_path, index=False)
    print(f"  saved: {sess_path}  ({len(sess_df)} rows)")
    print(f"  saved: {trades_path} ({len(trades_df)} rows)")
    return sess_df, trades_df


def _build_widget(result, sess_df: pd.DataFrame, trades_df: pd.DataFrame):
    """Equity curve + drawdown + per-trade P&L bars + diagnostic panel."""
    daily = result.daily_pnl.copy()
    daily.index = daily.index.strftime("%Y-%m-%d")
    equity = CONFIG["account_size"] + daily.cumsum()
    peak = equity.cummax()
    dd = equity - peak

    labels = list(daily.index)
    equity_data = [round(float(v), 2) for v in equity.values]
    dd_data = [round(float(v), 2) for v in dd.values]
    trade_dates = trades_df["date"].tolist() if len(trades_df) else []
    trade_pnls = [round(float(v), 2) for v in trades_df["pnl_net"].tolist()] if len(trades_df) else []
    trade_sides = trades_df["side"].tolist() if len(trades_df) else []

    # POC filter audit: avg rejection rate on signal-eligible days
    elig = sess_df[(sess_df["raw_long"] + sess_df["raw_short"]) > 0]
    raw_total = int((elig["raw_long"] + elig["raw_short"]).sum())
    comp_total = int((elig["comp_long"] + elig["comp_short"]).sum())
    poc_rejection_rate = 1.0 - (comp_total / raw_total) if raw_total else 0.0

    m = result.metrics
    report_lines = fmt_report(m).split("\n")

    blob = json.dumps({
        "labels": labels,
        "equity": equity_data,
        "drawdown": dd_data,
        "trade_dates": trade_dates,
        "trade_pnls": trade_pnls,
        "trade_sides": trade_sides,
        "report_lines": report_lines,
        "poc_audit": {
            "raw_breakouts": raw_total,
            "compound_passed": comp_total,
            "rejection_rate": round(poc_rejection_rate, 4),
            "eligible_sessions": int(len(elig)),
        },
        "account_size": CONFIG["account_size"],
        "start": result.start_date,
        "end": result.end_date,
    })

    html = """
<style>
 .grid { display:grid; grid-template-columns: 2fr 1fr; gap:12px; }
 .panel { padding:10px; border:1px solid var(--color-border, #444); border-radius:6px; }
 pre { font-family: ui-monospace, monospace; font-size: 12px; line-height:1.4; margin:0;
       color: var(--color-text-primary, #ddd); }
 h3 { margin: 4px 0 8px 0; color: var(--color-text-primary, #ddd); font-size:14px; }
 .kpi { display:flex; gap:18px; flex-wrap:wrap; margin-bottom:8px; }
 .kpi div { font-size:12px; color: var(--color-text-secondary, #aaa); }
 .kpi b { color: var(--color-text-primary, #ddd); font-size:14px; }
</style>
<div class="kpi" id="kpis"></div>
<div class="grid">
  <div class="panel">
    <h3>Equity curve</h3>
    <div style="position:relative;width:100%"><canvas id="eq"></canvas></div>
  </div>
  <div class="panel">
    <h3>Performance report</h3>
    <pre id="report"></pre>
  </div>
  <div class="panel">
    <h3>Drawdown ($)</h3>
    <div style="position:relative;width:100%"><canvas id="dd"></canvas></div>
  </div>
  <div class="panel">
    <h3>Per-trade P&L</h3>
    <div style="position:relative;width:100%"><canvas id="pnl"></canvas></div>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script>
const D = __BLOB__;
const txt = getComputedStyle(document.body).getPropertyValue('--color-text-primary') || '#ddd';
const grid = 'rgba(128,128,128,0.18)';
Chart.defaults.color = txt;
Chart.defaults.borderColor = grid;

document.getElementById('report').textContent = D.report_lines.join("\\n");
document.getElementById('kpis').innerHTML = `
  <div>Window<br><b>${D.start} \u2192 ${D.end}</b></div>
  <div>Sessions<br><b>${D.labels.length}</b></div>
  <div>Trades<br><b>${D.trade_pnls.length}</b></div>
  <div>POC filter rejection<br><b>${(D.poc_audit.rejection_rate*100).toFixed(1)}% (${D.poc_audit.compound_passed}/${D.poc_audit.raw_breakouts})</b></div>
`;

new Chart(document.getElementById('eq'), {
  type: 'line',
  data: { labels: D.labels, datasets: [
    { label: 'Equity ($)', data: D.equity, borderColor:'#4ade80', borderWidth:1.5,
      pointRadius: 0, tension: 0.05 }
  ]},
  options: { aspectRatio: 2.2, plugins: { legend: { display: true }},
    scales: { x: { ticks: { maxTicksLimit: 8 }, grid:{color:grid}},
              y: { grid:{color:grid}, ticks: { callback: v => '$'+v.toLocaleString() }}}}
});
new Chart(document.getElementById('dd'), {
  type: 'line',
  data: { labels: D.labels, datasets: [
    { label: 'Drawdown ($)', data: D.drawdown, borderColor:'#f87171', backgroundColor:'rgba(248,113,113,0.25)',
      borderWidth:1, fill: true, pointRadius: 0, tension: 0 }
  ]},
  options: { aspectRatio: 2.2,
    scales: { x: { ticks: { maxTicksLimit: 8 }, grid:{color:grid}},
              y: { grid:{color:grid}, ticks: { callback: v => '$'+v.toLocaleString() }}}}
});
const barColors = D.trade_pnls.map(v => v >= 0 ? '#4ade80' : '#f87171');
new Chart(document.getElementById('pnl'), {
  type: 'bar',
  data: { labels: D.trade_dates, datasets: [
    { label: 'P&L ($)', data: D.trade_pnls, backgroundColor: barColors,
      borderColor: barColors, borderWidth: 0 }
  ]},
  options: { aspectRatio: 2.2,
    scales: { x: { ticks: { maxTicksLimit: 12 }, grid:{color:grid}},
              y: { grid:{color:grid}, ticks: { callback: v => '$'+v.toLocaleString() }}}}
});
</script>
"""
    save_widget(html.replace("__BLOB__", blob), "backtest_report")


def main():
    print(f"[backtest] window: {START_DATE} -> {END_DATE}")
    print(f"[backtest] config: account=${CONFIG['account_size']:,.0f} risk={CONFIG['risk_pct_per_trade']*100:.1f}% "
          f"trail={CONFIG['trail_method']} stop_trig={CONFIG['stop_trigger_mode']} "
          f"vp_bin={CONFIG['vp_bin_size_ticks']}t partial={CONFIG['partial_exit_fraction']*100:.0f}%")
    t0 = time.time()
    result = run_backtest(START_DATE, END_DATE, progress_callback=_progress)
    dt = time.time() - t0
    print(f"\n[backtest] complete in {dt:.1f}s")
    print(fmt_report(result.metrics))

    # CSVs land in ./_bt_out beside cwd; widget is saved via the save_widget builtin.
    sess_df, trades_df = _save_csvs(result, "./_bt_out")
    _build_widget(result, sess_df, trades_df)
    print("[backtest] widget + CSVs saved.")


if __name__ == "__main__":
    main()