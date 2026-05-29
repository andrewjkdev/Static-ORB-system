"""Module E: Partial-exit / confirming-breakout detection.

A "confirming breakout" is a 1m bar after entry that:

  LONG:  closes ABOVE the most-recent swing HIGH in the post-entry window
         AND passes the same compound POC filter used at entry
         ( POC > OR_mid AND close > VAH )

  SHORT: closes BELOW the most-recent swing LOW
         ( POC < OR_mid AND close < VAL )

Swing definition (fractal): a bar at index i is a swing high if its high
is >= the highs of `strength` bars on each side (default strength=1).
Same for swing low. The post-entry lookback window for finding a swing
is `swing_lookback_bars` (default 5).

Only ONE confirming breakout per trade; the trade engine schedules a
partial exit at the NEXT bar's open and tightens the trail to the
confirming bar's low (long) / high (short).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import CONFIG
from .signals import Signal


@dataclass(frozen=True)
class ConfirmingBreakout:
    bar_idx: int
    bar_ts: pd.Timestamp
    breakout_close: float
    breakout_bar_high: float
    breakout_bar_low: float
    swing_level: float


def _find_swings(bars_slice: pd.DataFrame, strength: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for fractal swing highs/lows in a window."""
    n = len(bars_slice)
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)
    h = bars_slice["h"].to_numpy()
    lo = bars_slice["l"].to_numpy()
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(strength, n - strength):
        is_sh = True
        is_sl = True
        for k in range(1, strength + 1):
            if not (h[i] >= h[i - k] and h[i] >= h[i + k]):
                is_sh = False
            if not (lo[i] <= lo[i - k] and lo[i] <= lo[i + k]):
                is_sl = False
            if not is_sh and not is_sl:
                break
        sh[i] = is_sh
        sl[i] = is_sl
    return sh, sl


def find_confirming_breakout(
    session_bars: pd.DataFrame,
    profile: pd.DataFrame,
    signal: Signal,
    scan_from_iloc: int,
) -> Optional[ConfirmingBreakout]:
    """First bar at or after `scan_from_iloc` satisfying all confirming conditions."""
    n = len(session_bars)
    lookback = CONFIG["swing_lookback_bars"]
    strength = CONFIG["swing_fractal_strength"]
    or_mid = signal.or_mid

    poc = profile["poc"].to_numpy()
    vah = profile["vah"].to_numpy()
    val = profile["val"].to_numpy()
    closes = session_bars["c"].to_numpy()
    highs = session_bars["h"].to_numpy()
    lows = session_bars["l"].to_numpy()

    for j in range(scan_from_iloc, n):
        win_start = max(signal.entry_bar_idx, j - lookback)
        if j - win_start < 2 * strength + 1:
            continue
        win_slice = session_bars.iloc[win_start:j]
        sh, sl = _find_swings(win_slice, strength)
        if signal.side == "long":
            if not sh.any():
                continue
            swing_high = float(win_slice["h"].to_numpy()[sh].max())
            if closes[j] > swing_high and poc[j] > or_mid and closes[j] > vah[j]:
                return ConfirmingBreakout(
                    bar_idx=j, bar_ts=session_bars.index[j],
                    breakout_close=float(closes[j]),
                    breakout_bar_high=float(highs[j]),
                    breakout_bar_low=float(lows[j]),
                    swing_level=swing_high,
                )
        else:
            if not sl.any():
                continue
            swing_low = float(win_slice["l"].to_numpy()[sl].min())
            if closes[j] < swing_low and poc[j] < or_mid and closes[j] < val[j]:
                return ConfirmingBreakout(
                    bar_idx=j, bar_ts=session_bars.index[j],
                    breakout_close=float(closes[j]),
                    breakout_bar_high=float(highs[j]),
                    breakout_bar_low=float(lows[j]),
                    swing_level=swing_low,
                )
    return None