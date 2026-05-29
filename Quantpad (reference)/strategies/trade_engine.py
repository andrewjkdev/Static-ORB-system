"""Per-trade orchestrator: from signal -> entry -> manage -> exit.

Walks bars sequentially after entry, calling:

  StopState         to track initial / breakeven / trailing stop levels
  find_confirming_breakout  to detect the 30%-partial trigger

Fills are modeled at NEXT BAR'S OPEN (with adverse slippage) for both
the entry, the partial, and any stop-out, consistent with Andrew's
close-through stop trigger rule. EOD flush at the last RTH bar's close.

Same-bar priority: if a stop-out and a confirming-breakout both trigger
on bar j, both schedule fills for j+1's open; on j+1 the stop fill runs
first and exits ALL remaining contracts, pre-empting the partial -- per
the spec edge-case rule ("Stop and partial trigger same bar -> stop
priority, worst-case fill assumption").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from .config import CONFIG, TICK_SIZE, TICK_VALUE, slippage_price
from .exits import ConfirmingBreakout, find_confirming_breakout
from .signals import Signal
from .sizing import PositionSize
from .stops import StopState

_POINT_VALUE = TICK_VALUE / TICK_SIZE  # $ per 1.0 NQ point per contract = $20


@dataclass
class FillEvent:
    ts: pd.Timestamp
    bar_idx: int
    price: float
    contracts: int
    kind: str        # "entry" | "partial" | "stop" | "eod"
    reason: str


@dataclass
class TradeResult:
    signal: Signal
    sizing: PositionSize
    fills: List[FillEvent] = field(default_factory=list)
    confirming: Optional[ConfirmingBreakout] = None
    final_stop_at_exit: float = 0.0
    final_state_at_exit: str = "INITIAL"
    pnl_points: float = 0.0
    pnl_gross: float = 0.0
    pnl_net: float = 0.0
    commission_total: float = 0.0
    exit_reason: str = ""
    breakeven_ts: Optional[pd.Timestamp] = None
    trailing_first_move_ts: Optional[pd.Timestamp] = None


def _pnl_pts_per_contract(side: str, entry: float, exit_price: float) -> float:
    return (exit_price - entry) if side == "long" else (entry - exit_price)


def run_trade(
    session_bars: pd.DataFrame,
    profile: pd.DataFrame,
    signal: Signal,
    sizing: PositionSize,
) -> TradeResult:
    side = signal.side
    n = len(session_bars)
    entry_iloc = signal.entry_bar_idx
    slip = slippage_price()

    result = TradeResult(signal=signal, sizing=sizing)
    result.fills.append(FillEvent(
        ts=signal.entry_ts, bar_idx=entry_iloc,
        price=signal.entry_price, contracts=sizing.contracts,
        kind="entry", reason="signal",
    ))

    stop = StopState(
        side=side,
        entry_price=signal.entry_price,
        hard_stop_distance=sizing.stop_distance_price,
    )

    confirming = find_confirming_breakout(
        session_bars, profile, signal, scan_from_iloc=entry_iloc + 1
    )
    if confirming is not None:
        result.confirming = confirming

    remaining = sizing.contracts
    partial_done = False
    pnl_pts_accum = 0.0
    pending_stop_at: Optional[int] = None
    pending_partial_at: Optional[int] = None

    j = entry_iloc
    while j < n and remaining > 0:
        bar_ts = session_bars.index[j]
        bar_open = float(session_bars["o"].iloc[j])
        bar_high = float(session_bars["h"].iloc[j])
        bar_low = float(session_bars["l"].iloc[j])
        bar_close = float(session_bars["c"].iloc[j])

        # 1) Pending fills at this bar's open. Stop has priority over partial.
        if pending_stop_at == j and remaining > 0:
            fill_px = (bar_open - slip) if side == "long" else (bar_open + slip)
            pnl_pts_accum += _pnl_pts_per_contract(side, signal.entry_price, fill_px) * remaining
            result.fills.append(FillEvent(
                ts=bar_ts, bar_idx=j, price=fill_px, contracts=remaining,
                kind="stop", reason="close_through_stop",
            ))
            remaining = 0
            result.exit_reason = "stop"
            break
        if pending_partial_at == j and not partial_done and remaining > 0 and sizing.partial_contracts > 0:
            qty = min(sizing.partial_contracts, remaining)
            fill_px = (bar_open - slip) if side == "long" else (bar_open + slip)
            pnl_pts_accum += _pnl_pts_per_contract(side, signal.entry_price, fill_px) * qty
            result.fills.append(FillEvent(
                ts=bar_ts, bar_idx=j, price=fill_px, contracts=qty,
                kind="partial", reason="confirming_breakout",
            ))
            remaining -= qty
            partial_done = True

        # 2) Trailing ratchet using PRIOR completed bar (inactive while INITIAL)
        if j > entry_iloc and remaining > 0:
            prev_h = float(session_bars["h"].iloc[j - 1])
            prev_l = float(session_bars["l"].iloc[j - 1])
            stop.update_trail(prev_h, prev_l, bar_ts)

        # 3) Breakeven arm on this bar's close
        if remaining > 0:
            stop.check_breakeven_arm(bar_close, bar_ts)

        # 4) Confirming breakout on this bar -> schedule partial + tighten stop
        if (
            confirming is not None
            and not partial_done
            and remaining > 0
            and sizing.partial_contracts > 0
            and j == confirming.bar_idx
        ):
            stop.tighten_to(
                level=confirming.breakout_bar_low if side == "long" else confirming.breakout_bar_high,
                bar_ts=bar_ts,
            )
            if j + 1 < n:
                pending_partial_at = j + 1

        # 5) Stop-hit on this bar's close -> schedule exit for next bar's open
        if remaining > 0 and stop.check_stop_hit_close(bar_close):
            if j + 1 < n:
                pending_stop_at = j + 1
            else:
                fill_px = (bar_close - slip) if side == "long" else (bar_close + slip)
                pnl_pts_accum += _pnl_pts_per_contract(side, signal.entry_price, fill_px) * remaining
                result.fills.append(FillEvent(
                    ts=bar_ts, bar_idx=j, price=fill_px, contracts=remaining,
                    kind="stop", reason="close_through_stop_on_eod_bar",
                ))
                remaining = 0
                result.exit_reason = "stop_eod"
                break

        j += 1
        # Silence unused-var lints for high/low (kept for future intra-bar fill logic)
        _ = (bar_high, bar_low)

    # 6) EOD flush if still holding
    if remaining > 0:
        last_iloc = n - 1
        last_close = float(session_bars["c"].iloc[last_iloc])
        fill_px = (last_close - slip) if side == "long" else (last_close + slip)
        pnl_pts_accum += _pnl_pts_per_contract(side, signal.entry_price, fill_px) * remaining
        result.fills.append(FillEvent(
            ts=session_bars.index[last_iloc], bar_idx=last_iloc,
            price=fill_px, contracts=remaining,
            kind="eod", reason="end_of_session",
        ))
        remaining = 0
        result.exit_reason = result.exit_reason or "eod"

    result.pnl_points = pnl_pts_accum
    result.pnl_gross = pnl_pts_accum * _POINT_VALUE
    result.commission_total = sum(f.contracts for f in result.fills) * CONFIG["commission_per_side"]
    result.pnl_net = result.pnl_gross - result.commission_total
    result.final_stop_at_exit = stop.stop
    result.final_state_at_exit = stop.state
    result.breakeven_ts = stop.breakeven_ts
    result.trailing_first_move_ts = stop.trailing_first_move_ts
    return result