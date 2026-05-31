"""Module B: Session-anchored developing Volume Profile + POC.

Two construction modes (tick is preferred, bar_approx is fallback for the
>1-year-old portion of the backtest where tick history is capped):

  - "tick":       True print-by-print histogram from get_ticks("trades").
  - "bar_approx": Distribute each 1m bar's volume across the price bins
                  covering [bar.low, bar.high]. Uniform by default;
                  "triangular_typical" weights toward (h+l+c)/3.

Output: per 1m bar a developing snapshot of (POC, VAH, VAL, total_vol),
indexed by the bar OPEN timestamp (matching session_bars.index).

POC tie-break rule: when multiple bins share the max volume, choose the bin
whose midpoint is closest to that bar's close price (stability heuristic).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .config import CONFIG, vp_bin_size_price


VP_VALUE_AREA_FRACTION = 0.70  # standard "70% around POC" definition


def _bin_grid(price_lo: float, price_hi: float, bin_size: float) -> tuple[np.ndarray, float]:
    """Build a price-bin grid that covers [price_lo - pad, price_hi + pad].

    Returns (bin_left_edges_array, grid_min). bin index = floor((p - grid_min) / bin_size).
    Pads 5 bins on each side to absorb intraday extension after the grid is built.
    """
    pad = 5 * bin_size
    grid_min = (np.floor((price_lo - pad) / bin_size)) * bin_size
    grid_max = (np.ceil((price_hi + pad) / bin_size)) * bin_size
    n_bins = int(round((grid_max - grid_min) / bin_size)) + 1
    edges = grid_min + np.arange(n_bins) * bin_size
    return edges, grid_min


def _value_area(row_cum_vol: np.ndarray, poc_idx: int, total: float) -> tuple[int, int]:
    """Expand around poc_idx until cumulative volume >= 70%. Returns (val_idx, vah_idx)."""
    if total <= 0:
        return poc_idx, poc_idx
    target = VP_VALUE_AREA_FRACTION * total
    lo = hi = poc_idx
    cum = row_cum_vol[poc_idx]
    n = len(row_cum_vol)
    while cum < target and (lo > 0 or hi < n - 1):
        up_vol = row_cum_vol[hi + 1] if hi < n - 1 else -1.0
        dn_vol = row_cum_vol[lo - 1] if lo > 0 else -1.0
        if up_vol >= dn_vol:
            hi += 1
            cum += max(up_vol, 0.0)
        else:
            lo -= 1
            cum += max(dn_vol, 0.0)
    return lo, hi


def _poc_with_tiebreak(row_cum_vol: np.ndarray, bin_mids: np.ndarray, close_price: float) -> int:
    """argmax with tie-break = bin midpoint closest to close_price."""
    max_vol = row_cum_vol.max()
    if max_vol <= 0:
        # No volume yet (shouldn't happen after bar 0); fall back to nearest bin to close.
        return int(np.argmin(np.abs(bin_mids - close_price)))
    cand = np.flatnonzero(row_cum_vol == max_vol)
    if cand.size == 1:
        return int(cand[0])
    distances = np.abs(bin_mids[cand] - close_price)
    return int(cand[int(np.argmin(distances))])


def build_developing_profile_tick(
    session_bars: pd.DataFrame,
    session_ticks: pd.DataFrame,
) -> pd.DataFrame:
    """Tick-based developing volume profile, one row per 1m bar."""
    bin_size = vp_bin_size_price()
    # Price range needs to cover both bar wicks and any tick prints (usually they agree).
    price_lo = min(session_bars["l"].min(), session_ticks["price"].min())
    price_hi = max(session_bars["h"].max(), session_ticks["price"].max())
    edges, grid_min = _bin_grid(float(price_lo), float(price_hi), bin_size)
    n_bins = len(edges)
    bin_mids = edges + bin_size / 2.0

    # Map each tick to its bin index
    tick_bin = np.floor((session_ticks["price"].to_numpy() - grid_min) / bin_size).astype(int)
    tick_bin = np.clip(tick_bin, 0, n_bins - 1)
    tick_size_arr = session_ticks["size"].to_numpy(dtype=np.float64)

    # Map each tick to its containing 1m bar index in session_bars
    bar_starts = session_bars.index.values.astype("datetime64[ns]")  # bar OPEN times (UTC)
    tick_t = session_ticks["t"].dt.tz_convert("UTC").values.astype("datetime64[ns]") \
        if session_ticks["t"].dt.tz is not None else session_ticks["t"].values.astype("datetime64[ns]")
    # searchsorted on bar open times; right-side so a tick exactly at bar_start belongs to that bar.
    bar_idx_per_tick = np.searchsorted(bar_starts, tick_t, side="right") - 1
    valid = bar_idx_per_tick >= 0
    tick_bin = tick_bin[valid]
    tick_size_arr = tick_size_arr[valid]
    bar_idx_per_tick = bar_idx_per_tick[valid]

    n_bars = len(session_bars)
    contrib = np.zeros((n_bars, n_bins), dtype=np.float64)
    flat = bar_idx_per_tick * n_bins + tick_bin
    np.add.at(contrib.reshape(-1), flat, tick_size_arr)

    cum = np.cumsum(contrib, axis=0)  # row i = volume in each bin from session open through bar i
    return _finalize_profile(session_bars, cum, bin_mids, bin_size)


def build_developing_profile_bar_approx(
    session_bars: pd.DataFrame,
    distribution: Literal["uniform", "triangular_typical"] = "uniform",
) -> pd.DataFrame:
    """Bar-approximated developing profile when tick data isn't available."""
    bin_size = vp_bin_size_price()
    price_lo = float(session_bars["l"].min())
    price_hi = float(session_bars["h"].max())
    edges, grid_min = _bin_grid(price_lo, price_hi, bin_size)
    n_bins = len(edges)
    bin_mids = edges + bin_size / 2.0

    n_bars = len(session_bars)
    contrib = np.zeros((n_bars, n_bins), dtype=np.float64)

    lows = session_bars["l"].to_numpy()
    highs = session_bars["h"].to_numpy()
    closes = session_bars["c"].to_numpy()
    vols = session_bars["v"].to_numpy(dtype=np.float64)

    for i in range(n_bars):
        lo_bin = int(np.floor((lows[i] - grid_min) / bin_size))
        hi_bin = int(np.floor((highs[i] - grid_min) / bin_size))
        lo_bin = max(0, lo_bin)
        hi_bin = min(n_bins - 1, hi_bin)
        span = hi_bin - lo_bin + 1
        if span <= 0 or vols[i] <= 0:
            continue
        if distribution == "uniform":
            contrib[i, lo_bin:hi_bin + 1] += vols[i] / span
        else:  # triangular_typical: weight toward typical price (h+l+c)/3
            typ = (highs[i] + lows[i] + closes[i]) / 3.0
            mids = bin_mids[lo_bin:hi_bin + 1]
            # Triangle peaked at typ, falling linearly to bar edges
            edge_dist = max(highs[i] - lows[i], bin_size)
            w = np.maximum(0.0, 1.0 - np.abs(mids - typ) / edge_dist)
            if w.sum() <= 0:
                w = np.ones_like(w)
            w = w / w.sum()
            contrib[i, lo_bin:hi_bin + 1] += vols[i] * w

    cum = np.cumsum(contrib, axis=0)
    return _finalize_profile(session_bars, cum, bin_mids, bin_size)


def _finalize_profile(
    session_bars: pd.DataFrame,
    cum: np.ndarray,
    bin_mids: np.ndarray,
    bin_size: float,
) -> pd.DataFrame:
    """Common reduction: row-wise POC (with tie-break), VAH, VAL, total_vol."""
    closes = session_bars["c"].to_numpy()
    n_bars = cum.shape[0]
    poc = np.empty(n_bars, dtype=np.float64)
    vah = np.empty(n_bars, dtype=np.float64)
    val = np.empty(n_bars, dtype=np.float64)
    total = cum.sum(axis=1)

    for i in range(n_bars):
        row = cum[i]
        poc_idx = _poc_with_tiebreak(row, bin_mids, float(closes[i]))
        poc[i] = bin_mids[poc_idx]
        lo_idx, hi_idx = _value_area(row, poc_idx, float(total[i]))
        val[i] = bin_mids[lo_idx] - bin_size / 2.0
        vah[i] = bin_mids[hi_idx] + bin_size / 2.0

    out = pd.DataFrame(
        {"poc": poc, "vah": vah, "val": val, "total_vol": total},
        index=session_bars.index,
    )
    return out


def build_developing_profile(
    session_bars: pd.DataFrame,
    session_ticks: pd.DataFrame | None,
    mode: str = "auto",
) -> tuple[pd.DataFrame, str]:
    """Dispatch to tick or bar_approx based on data availability and mode.

    Returns (profile_df, mode_used). Falls back to bar_approx if mode == "auto"
    and ticks are empty/None.
    """
    if mode == "tick":
        if session_ticks is None or session_ticks.empty:
            raise ValueError("mode='tick' requested but session_ticks is empty.")
        return build_developing_profile_tick(session_bars, session_ticks), "tick"
    if mode == "bar_approx":
        return build_developing_profile_bar_approx(
            session_bars, CONFIG["vp_bar_approx_distribution"]
        ), "bar_approx"
    # auto
    if session_ticks is not None and not session_ticks.empty:
        return build_developing_profile_tick(session_bars, session_ticks), "tick"
    return build_developing_profile_bar_approx(
        session_bars, CONFIG["vp_bar_approx_distribution"]
    ), "bar_approx"