"""Phase-2 sanity: walk last ~25 trading days, run full signal+trade pipeline, print summary."""
from __future__ import annotations

import sys
import traceback
import pandas as pd
import exchange_calendars as xcals

from strategies.orb_nq_vp.data import session_rth_bounds, load_session_bars, load_session_ticks, compute_OR
from strategies.orb_nq_vp.volume_profile import build_developing_profile
from strategies.orb_nq_vp.signals import find_first_signal
from strategies.orb_nq_vp.sizing import compute_position_size
from strategies.orb_nq_vp.trade_engine import run_trade


def recent_sessions(n_back: int = 25) -> list[str]:
    """Return the last `n_back` completed NYSE sessions (ISO date strings)."""
    nyse = xcals.get_calendar("XNYS")
    # Use yesterday UTC as the end-bound so today's incomplete session is excluded.
    today_str = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    end = pd.Timestamp(today_str) - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=n_back * 2)
    sessions = nyse.sessions_in_range(
        start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    )
    return [s.strftime("%Y-%m-%d") for s in sessions[-n_back:]]


def run_one(date_iso: str) -> dict:
    """Process a single session end-to-end. Returns a dict row."""
    row = {"date": date_iso, "outcome": "?", "side": "", "pnl_net": 0.0, "contracts": 0,
           "partial": 0, "fills": 0, "exit_reason": "", "note": ""}
    bounds = session_rth_bounds(date_iso)
    if bounds is None:
        row["outcome"] = "skip_no_session"
        return row
    bars = load_session_bars(bounds)
    if bars.empty or len(bars) < 5:
        row["outcome"] = "skip_no_bars"
        return row
    # Tick fetch (may fail / be slow on very old days; fall back to bar_approx)
    try:
        ticks = load_session_ticks(bounds)
    except Exception as e:  # noqa: BLE001
        ticks = pd.DataFrame()
        row["note"] = f"tick_fetch_err:{type(e).__name__}"
    profile, vp_mode = build_developing_profile(bars, ticks, mode="auto")
    OR = compute_OR(bars, bounds)
    if OR.skip:
        row["outcome"] = f"skip_{OR.skip_reason}"
        return row
    scan = find_first_signal(bars, profile, OR)
    row["note"] = (row["note"] + f"|raw L{scan.raw_long_count}/S{scan.raw_short_count}"
                   f" comp L{scan.compound_long_count}/S{scan.compound_short_count} vp={vp_mode}").lstrip("|")
    if scan.signal is None:
        row["outcome"] = f"no_trade:{scan.skip_reason}"
        return row
    sig = scan.signal
    initial_stop = OR.low if sig.side == "long" else OR.high
    sz = compute_position_size(sig.entry_price, initial_stop, sig.side)
    if sz.skip:
        row["outcome"] = f"no_trade:{sz.skip_reason}"
        row["side"] = sig.side
        return row
    trade = run_trade(bars, profile, sig, sz)
    row["outcome"] = "trade"
    row["side"] = sig.side
    row["pnl_net"] = trade.pnl_net
    row["contracts"] = sz.contracts
    row["partial"] = sz.partial_contracts
    row["fills"] = len(trade.fills)
    row["exit_reason"] = trade.exit_reason
    if trade.confirming is not None:
        row["note"] += "|confirming_hit"
    return row


def main():
    dates = recent_sessions(25)
    print(f"[sanity-2] processing {len(dates)} sessions: {dates[0]} -> {dates[-1]}")
    rows = []
    for d in dates:
        try:
            r = run_one(d)
        except Exception as e:  # noqa: BLE001
            r = {"date": d, "outcome": f"ERROR:{type(e).__name__}", "side": "",
                 "pnl_net": 0.0, "contracts": 0, "partial": 0, "fills": 0,
                 "exit_reason": "", "note": str(e)[:80]}
            traceback.print_exc(file=sys.stderr)
        rows.append(r)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 100)
    print(df.to_string(index=False))
    trades = df[df["outcome"] == "trade"]
    print()
    print(f"[summary] sessions={len(df)} trades_taken={len(trades)}")
    if len(trades):
        wins = (trades["pnl_net"] > 0).sum()
        print(f"[summary] wins={wins}/{len(trades)}  total_pnl_net=${trades['pnl_net'].sum():,.2f}")
        print(f"[summary] avg_pnl_per_trade=${trades['pnl_net'].mean():,.2f}")
        print(f"[summary] exit_reasons:\n{trades['exit_reason'].value_counts()}")
    print(f"[summary] skip/no-trade breakdown:\n{df[df['outcome'] != 'trade']['outcome'].value_counts()}")


if __name__ == "__main__":
    main()