"""Module C: Entry signal detection.

Compound POC filter (Andrew's tightening of the spec):
  LONG:  close > OR.high  AND  POC > OR_mid  AND  close > VAH
  SHORT: close < OR.low   AND  POC < OR_mid  AND  close < VAL

Behaviour:
  - Scan only the first `entry_window_minutes` bars of the session.
  - First qualifying bar (in time order) wins; one-trade-per-session.
  - If both directions qualify on different bars, the EARLIER bar wins.
  - Entry fills on the OPEN of the bar IMMEDIATELY AFTER the signal bar,
    with adverse slippage of `slippage_ticks_per_side` ticks.
  - If the signal lands on the last RTH bar, the trade is skipped
    (no next bar to fill on); logged as `signal_on_last_bar`.

Returns a SignalScan with diagnostic counts so Module G can audit the
filter's actual rejection rate across the full backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .config import CONFIG, slippage_price
from .data import OpeningRange


@dataclass(frozen=True)
class Signal:
    side: str
    signal_bar_idx: int
    signal_ts: pd.Timestamp
    signal_close: float
    or_high: float
    or_low: float
    or_mid: float
    or_range: float
    poc_at_signal: float
    vah_at_signal: float
    val_at_signal: float
    entry_bar_idx: int
    entry_ts: pd.Timestamp
    entry_price: float


@dataclass(frozen=True)
class SignalScan:
    signal: Optional[Signal]
    raw_long_count: int
    raw_short_count: int
    compound_long_count: int
    compound_short_count: int
    skip_reason: Optional[str]


def find_first_signal(
    session_bars: pd.DataFrame,
    profile: pd.DataFrame,
    OR: OpeningRange,
) -> SignalScan:
    """Scan the entry window and return the first compound-filter-passing bar."""
    if OR.skip:
        return SignalScan(None, 0, 0, 0, 0, f"or_{OR.skip_reason}")

    n_window = min(CONFIG["entry_window_minutes"], len(session_bars))
    or_mid = 0.5 * (OR.high + OR.low)

    win_bars = session_bars.iloc[:n_window]
    win_poc = profile["poc"].iloc[:n_window]
    win_vah = profile["vah"].iloc[:n_window]
    win_val = profile["val"].iloc[:n_window]
    closes = win_bars["c"]

    raw_long = closes > OR.high
    raw_short = closes < OR.low
    comp_long = raw_long & (win_poc > or_mid) & (closes > win_vah)
    comp_short = raw_short & (win_poc < or_mid) & (closes < win_val)

    first_long_ts = comp_long[comp_long].index.min() if comp_long.any() else None
    first_short_ts = comp_short[comp_short].index.min() if comp_short.any() else None

    chosen_side: Optional[str] = None
    chosen_ts: Optional[pd.Timestamp] = None
    if first_long_ts is not None and first_short_ts is not None:
        if first_long_ts <= first_short_ts:
            chosen_side, chosen_ts = "long", first_long_ts
        else:
            chosen_side, chosen_ts = "short", first_short_ts
    elif first_long_ts is not None:
        chosen_side, chosen_ts = "long", first_long_ts
    elif first_short_ts is not None:
        chosen_side, chosen_ts = "short", first_short_ts

    raw_long_count = int(raw_long.sum())
    raw_short_count = int(raw_short.sum())
    comp_long_count = int(comp_long.sum())
    comp_short_count = int(comp_short.sum())

    if chosen_side is None or chosen_ts is None:
        return SignalScan(None, raw_long_count, raw_short_count,
                          comp_long_count, comp_short_count, "no_signal")

    sig_iloc = session_bars.index.get_loc(chosen_ts)
    if sig_iloc + 1 >= len(session_bars):
        return SignalScan(None, raw_long_count, raw_short_count,
                          comp_long_count, comp_short_count, "signal_on_last_bar")

    entry_iloc = sig_iloc + 1
    entry_ts = session_bars.index[entry_iloc]
    open_next = float(session_bars["o"].iloc[entry_iloc])
    slip = slippage_price()
    entry_price = open_next + slip if chosen_side == "long" else open_next - slip

    signal = Signal(
        side=chosen_side,
        signal_bar_idx=sig_iloc,
        signal_ts=chosen_ts,
        signal_close=float(session_bars["c"].iloc[sig_iloc]),
        or_high=OR.high, or_low=OR.low, or_mid=or_mid, or_range=OR.range_price,
        poc_at_signal=float(profile["poc"].iloc[sig_iloc]),
        vah_at_signal=float(profile["vah"].iloc[sig_iloc]),
        val_at_signal=float(profile["val"].iloc[sig_iloc]),
        entry_bar_idx=entry_iloc, entry_ts=entry_ts, entry_price=entry_price,
    )
    return SignalScan(signal, raw_long_count, raw_short_count,
                      comp_long_count, comp_short_count, None)