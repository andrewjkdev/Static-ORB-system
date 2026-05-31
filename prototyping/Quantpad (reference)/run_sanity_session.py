"""Phase-1 sanity check: load one /NQ RTH session, build OR + developing POC,
render an interactive widget, and print signal-candidate diagnostics.

This script verifies Modules A (data) and B (volume_profile) before we wire
up entry signals, stops, and the backtest harness.
"""

from __future__ import annotations

import json

import pandas as pd
import exchange_calendars as xcals

from strategies.orb_nq_vp.config import CONFIG
from strategies.orb_nq_vp.data import (
    session_rth_bounds,
    load_session_bars,
    load_session_ticks,
    compute_OR,
)
from strategies.orb_nq_vp.volume_profile import build_developing_profile


def pick_recent_session() -> str:
    nyse = xcals.get_calendar("XNYS")
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    cur = today - pd.Timedelta(days=5)
    while not nyse.is_session(cur):
        cur -= pd.Timedelta(days=1)
    return cur.strftime("%Y-%m-%d")


def main() -> None:
    date_iso = pick_recent_session()
    print(f"[sanity] target session: {date_iso}")

    bounds = session_rth_bounds(date_iso)
    assert bounds is not None, f"{date_iso} is not a trading day"
    print(
        f"[sanity] RTH NY: {bounds.open_ny.strftime('%H:%M')} -> "
        f"{bounds.close_ny.strftime('%H:%M')}  half_day={bounds.is_half_day}"
    )

    bars = load_session_bars(bounds)
    print(f"[sanity] 1m bars rows: {len(bars)}")
    if bars.empty:
        raise RuntimeError("No 1m bars returned for the session.")

    try:
        ticks = load_session_ticks(bounds)
    except Exception as e:
        print(f"[sanity] tick fetch failed ({e}); falling back to bar_approx")
        ticks = pd.DataFrame()
    print(f"[sanity] tick rows: {len(ticks)}")

    OR = compute_OR(bars, bounds)
    print(
        f"[sanity] OR: high={OR.high:.2f}  low={OR.low:.2f}  "
        f"range={OR.range_price:.2f} ({OR.range_ticks} ticks)  "
        f"skip={OR.skip}  reason={OR.skip_reason}"
    )
    if OR.skip:
        print("[sanity] OR flagged for skip; aborting visualization")
        return

    profile, mode_used = build_developing_profile(bars, ticks if not ticks.empty else None, mode="auto")
    print(f"[sanity] VP mode: {mode_used}")
    print(f"[sanity] POC@OR_close: {profile['poc'].iloc[0]:.2f}")
    print(f"[sanity] POC@session_end: {profile['poc'].iloc[-1]:.2f}")

    # Restrict signal scanning to the 90-min entry window
    n_window = min(CONFIG["entry_window_minutes"], len(bars))
    win_bars = bars.iloc[:n_window]
    win_poc = profile["poc"].iloc[:n_window]

    long_mask = (win_bars["c"] > OR.high) & (win_bars["c"] > win_poc)
    short_mask = (win_bars["c"] < OR.low) & (win_bars["c"] < win_poc)
    raw_long = win_bars["c"] > OR.high
    raw_short = win_bars["c"] < OR.low

    print(
        f"[sanity] raw price-breaks  long={int(raw_long.sum())}  short={int(raw_short.sum())}"
    )
    print(
        f"[sanity] POC-passed breaks long={int(long_mask.sum())}  short={int(short_mask.sum())}"
    )
    first_long = win_bars.index[long_mask].min() if long_mask.any() else None
    first_short = win_bars.index[short_mask].min() if short_mask.any() else None
    print(f"[sanity] first long signal:  {first_long}")
    print(f"[sanity] first short signal: {first_short}")

    # Build the widget
    tz = CONFIG["session_tz"]
    labels = [t.tz_convert(tz).strftime("%H:%M") for t in bars.index]
    close = bars["c"].astype(float).tolist()
    high = bars["h"].astype(float).tolist()
    low = bars["l"].astype(float).tolist()
    poc = profile["poc"].astype(float).tolist()

    long_pts = [
        {"x": labels[bars.index.get_loc(t)], "y": float(bars.loc[t, "c"])}
        for t in win_bars.index[long_mask]
    ]
    short_pts = [
        {"x": labels[bars.index.get_loc(t)], "y": float(bars.loc[t, "c"])}
        for t in win_bars.index[short_mask]
    ]

    entry_window_end = labels[n_window - 1]

    payload = json.dumps({
        "labels": labels,
        "close": close,
        "high": high,
        "low": low,
        "poc": poc,
        "or_high": OR.high,
        "or_low": OR.low,
        "long_pts": long_pts,
        "short_pts": short_pts,
        "entry_end": entry_window_end,
        "date_iso": date_iso,
        "mode": mode_used,
    })

    html = (
        '<div style="font:13px var(--font-mono, monospace);color:var(--color-text-primary);margin-bottom:8px">'
        f'<b>/NQ 1m ORB sanity</b> &mdash; session {date_iso} &mdash; VP mode: <code>{mode_used}</code> &mdash; '
        f'OR high {OR.high:.2f} / low {OR.low:.2f} (range {OR.range_price:.2f}, {OR.range_ticks} ticks) &mdash; '
        f'entry window cutoff <code>{entry_window_end}</code>'
        '</div>'
        '<div style="position:relative;width:100%">'
        '<canvas id="orbSanity"></canvas>'
        '</div>'
        '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>'
        '<script>'
        f'const D = {payload};'
        'const txt = getComputedStyle(document.documentElement).getPropertyValue("--color-text-primary") || "#222";'
        'const grid = getComputedStyle(document.documentElement).getPropertyValue("--color-border") || "rgba(128,128,128,0.2)";'
        'const ctx = document.getElementById("orbSanity").getContext("2d");'
        'new Chart(ctx, {'
        '  type: "line",'
        '  data: {'
        '    labels: D.labels,'
        '    datasets: ['
        '      {label:"Close",data:D.close,borderColor:"#4c8bf5",borderWidth:1.5,pointRadius:0,tension:0,order:5},'
        '      {label:"High",data:D.high,borderColor:"rgba(76,139,245,0.25)",borderWidth:0.5,pointRadius:0,fill:false,order:6},'
        '      {label:"Low",data:D.low,borderColor:"rgba(76,139,245,0.25)",borderWidth:0.5,pointRadius:0,fill:"-1",backgroundColor:"rgba(76,139,245,0.08)",order:6},'
        '      {label:"Developing POC",data:D.poc,borderColor:"#f5a623",borderDash:[4,3],borderWidth:1.5,pointRadius:0,order:4},'
        '      {label:"OR High",data:D.labels.map(()=>D.or_high),borderColor:"#2ecc71",borderWidth:1.2,pointRadius:0,order:3},'
        '      {label:"OR Low",data:D.labels.map(()=>D.or_low),borderColor:"#e74c3c",borderWidth:1.2,pointRadius:0,order:3},'
        '      {label:"Long signals",data:D.long_pts,borderColor:"#2ecc71",backgroundColor:"#2ecc71",pointRadius:5,pointStyle:"triangle",showLine:false,parsing:false,order:1},'
        '      {label:"Short signals",data:D.short_pts,borderColor:"#e74c3c",backgroundColor:"#e74c3c",pointRadius:5,pointStyle:"triangle",rotation:180,showLine:false,parsing:false,order:1}'
        '    ]'
        '  },'
        '  options: {'
        '    responsive: true,'
        '    maintainAspectRatio: true,'
        '    aspectRatio: 2.2,'
        '    interaction: {mode:"index", intersect:false},'
        '    plugins: {'
        '      legend: {labels:{color:txt}},'
        '      tooltip: {mode:"index", intersect:false}'
        '    },'
        '    scales: {'
        '      x: {ticks:{color:txt,maxTicksLimit:12,autoSkip:true}, grid:{color:grid}},'
        '      y: {ticks:{color:txt}, grid:{color:grid}}'
        '    }'
        '  }'
        '});'
        '</script>'
    )

    save_widget(html, "sanity_session")
    print("[sanity] widget saved")


if __name__ == "__main__":
    main()