"""Initial-stop dispatcher.

Supports three methods (switched via CONFIG["initial_stop_method"]):

  "or_levels":    Original spec. Long stop = OR.low ; short stop = OR.high.
                  Stop distance is dictated by the OR's range; can be wide.
  "fixed_points": Fixed offset from entry, in NQ points (CONFIG["hard_stop_points"]).
                  Constant risk per contract regardless of session vol.
  "atr":          Volatility-adaptive. Stop = entry +/- (atr_multiple * ATR(period))
                  where ATR is Wilder's ATR on 1m bars from session open
                  up to and including the signal bar. If fewer than `period`
                  bars are available, uses the rolling mean of available TRs.
                  Returns ("atr_extreme", None) if computed stop distance
                  falls outside CONFIG["atr_min_points"] / CONFIG["atr_max_points"].
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import CONFIG
from .data import OpeningRange


def _wilder_atr(bars: pd.DataFrame, period: int) -> float:
    """Wilder's ATR over the supplied bars (uses all rows, ending at the last).

    Returns ATR at the last bar. If fewer than `period` bars, returns the
    simple mean of available TRs (best-effort fallback).
    """
    n = len(bars)
    if n == 0:
        return float("nan")
    h = bars["h"].to_numpy(dtype=np.float64)
    lo = bars["l"].to_numpy(dtype=np.float64)
    c = bars["c"].to_numpy(dtype=np.float64)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = h[0] - lo[0]
    if n > 1:
        prev_c = c[:-1]
        a = h[1:] - lo[1:]
        b = np.abs(h[1:] - prev_c)
        d = np.abs(lo[1:] - prev_c)
        tr[1:] = np.maximum(np.maximum(a, b), d)
    if n < period:
        return float(np.mean(tr))
    atr = float(np.mean(tr[:period]))
    for i in range(period, n):
        atr = (atr * (period - 1) + tr[i]) / period
    return atr


def compute_initial_stop_for_signal(
    side: str,
    entry_price: float,
    signal_iloc: int,
    session_bars: pd.DataFrame,
    OR: OpeningRange,
) -> Tuple[Optional[float], Optional[str]]:
    """Dispatch on CONFIG["initial_stop_method"]. Returns (stop_price, skip_reason).

    On success: (stop_price, None).
    On skip:    (None, reason) -- caller logs reason and skips the trade.
    """
    method = CONFIG["initial_stop_method"]

    if method == "or_levels":
        stop = OR.low if side == "long" else OR.high
        return float(stop), None

    if method == "fixed_points":
        d = CONFIG["hard_stop_points"]
        stop = entry_price - d if side == "long" else entry_price + d
        return float(stop), None

    if method == "atr":
        period = int(CONFIG["atr_period"])
        mult = float(CONFIG["atr_multiple"])
        # ATR window: bars from session open up to AND INCLUDING signal bar.
        # signal_iloc is the bar where the breakout candle closed.
        atr_bars = session_bars.iloc[: signal_iloc + 1]
        atr = _wilder_atr(atr_bars, period)
        if not np.isfinite(atr) or atr <= 0:
            return None, "atr_unavailable"
        d = mult * atr
        if d < CONFIG["atr_min_points"] or d > CONFIG["atr_max_points"]:
            return None, f"atr_extreme:{d:.2f}"
        stop = entry_price - d if side == "long" else entry_price + d
        return float(stop), None

    return None, f"unknown_method:{method}"