"""Single source of truth for all tunable parameters.

Edit values here without touching strategy logic. Every module imports CONFIG
from this file. Anything not in CONFIG is a hard-coded constant of the
instrument (tick size, contract specs) and lives below the CONFIG dict.
"""

from __future__ import annotations

CONFIG: dict = {
    # Instrument
    "instrument": "NQ.FUT",
    "session_tz": "America/New_York",
    "session_open": "09:30",
    "session_close": "16:00",

    # ORB rules
    "or_lookback_minutes": 1,
    "entry_window_minutes": 90,
    "doji_tolerance_ticks": 0,            # OR.high == OR.low => skip
    "skip_if_or_range_ticks_gt": None,    # No hard cap per Andrew; sizing skips on its own

    # Strength validation (Volume-Profile POC bias)
    "vp_bin_size_ticks": 4,               # 4 ticks = $1.00 NQ-point bin
    "vp_data_mode": "auto",               # "tick" | "bar_approx" | "auto" (tick when available, else bar_approx)
    "vp_bar_approx_distribution": "uniform",  # "uniform" | "triangular_typical"

    # Stops
    "trail_method": "swing",              # Swing trail per Andrew
    "stop_trigger_mode": "close_through", # 1m close-through only per Andrew (no intra-bar touch)
    "validation_multiple": 1.0,           # Breakeven trigger = stop_distance * this multiple, beyond entry (1R-to-BE)
    "initial_stop_method": "atr",         # "or_levels" | "fixed_points" | "atr"
    "hard_stop_points": 30.0,             # used when initial_stop_method == "fixed_points"
    "atr_period": 14,                     # used when initial_stop_method == "atr"
    "atr_multiple": 1.25,                 # used when initial_stop_method == "atr"
    "atr_min_points": 1.0,                # skip trade if 1.25*ATR < this many points (degenerate stop)
    "atr_max_points": 100.0,              # skip trade if 1.25*ATR > this many points (vol-explosion guard)

    # Partial exit (confirming-breakout 30/70)
    "partial_exit_fraction": 0.60,        # 60/40 split per Andrew: lock 60% at confirming bar, ride 40%
    "partial_qty_rounding": "round",      # "round" | "floor" -- "round" gives cleanest 60/40 at small qty
    "swing_lookback_bars": 5,             # window to find most-recent swing high/low after entry
    "swing_fractal_strength": 1,          # bars on each side that must be lower/higher

    # Sizing
    "account_size": 100_000.0,
    "risk_pct_per_trade": 0.02,

    # Costs (per side)
    "commission_per_side": 1.50,          # $/contract per fill
    "slippage_ticks_per_side": 1,         # adverse, per fill
}

# /NQ contract constants (not user-tunable; reflect CME spec)
TICK_SIZE: float = 0.25
TICK_VALUE: float = 5.00                  # $ per tick per contract
POINT_VALUE: float = TICK_VALUE / TICK_SIZE  # $20 / point / contract

# Derived helpers (kept here so modules don't recompute)
def vp_bin_size_price() -> float:
    """Bin width in price units (NQ points)."""
    return CONFIG["vp_bin_size_ticks"] * TICK_SIZE

def slippage_price() -> float:
    """Adverse slippage in price units, per fill."""
    return CONFIG["slippage_ticks_per_side"] * TICK_SIZE