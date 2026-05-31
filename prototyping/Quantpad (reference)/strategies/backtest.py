"""Module G (part 2): Multi-session backtest orchestrator.

Walks a date range, processes every NYSE session in sequence, and collects:
  - List[TradeResult] for performance metrics
  - Per-session diagnostic counters (raw vs compound POC breakouts, skips)
  - Daily P&L pd.Series for drawdown/Sharpe/equity-curve

Pure orchestration — visualization lives in run_backtest.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd
import exchange_calendars as xcals

from .data import session_rth_bounds, load_session_bars, load_session_ticks, compute_OR
from .volume_profile import build_developing_profile
from .signals import find_first_signal
from .sizing import compute_position_size
from .trade_engine import run_trade, TradeResult
from .metrics import compute_metrics, PerformanceMetrics
from .initial_stop import compute_initial_stop_for_signal


@dataclass
class SessionRecord:
    """Per-session record (one row per NYSE session in the requested range)."""
    date: str
    outcome: str                  # "trade" | "no_trade:<reason>" | "skip_no_session" | "skip_no_bars" | "ERROR:<type>"
    side: str                     # "long" | "short" | ""
    pnl_net: float
    contracts: int
    partial_contracts: int
    fills: int
    exit_reason: str
    vp_mode: str                  # "tick" | "bar_approx" | ""
    raw_long: int
    raw_short: int
    comp_long: int
    comp_short: int
    confirming_hit: bool
    error: str = ""


@dataclass
class BacktestResult:
    start_date: str
    end_date: str
    sessions: List[SessionRecord] = field(default_factory=list)
    trade_results: List[TradeResult] = field(default_factory=list)
    daily_pnl: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    metrics: Optional[PerformanceMetrics] = None

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([s.__dict__ for s in self.sessions])


def _nyse_sessions(start_date: str, end_date: str) -> list[str]:
    """Return ISO date strings for every NYSE session within [start, end]."""
    cal = xcals.get_calendar("XNYS")
    sessions = cal.sessions_in_range(start_date, end_date)
    return [s.strftime("%Y-%m-%d") for s in sessions]


def _process_session(date_iso: str) -> tuple[SessionRecord, Optional[TradeResult]]:
    """Run the full per-session pipeline. Returns (record, trade_or_None)."""
    rec = SessionRecord(
        date=date_iso, outcome="?", side="", pnl_net=0.0,
        contracts=0, partial_contracts=0, fills=0, exit_reason="",
        vp_mode="", raw_long=0, raw_short=0, comp_long=0, comp_short=0,
        confirming_hit=False,
    )
    bounds = session_rth_bounds(date_iso)
    if bounds is None:
        rec.outcome = "skip_no_session"
        return rec, None

    bars = load_session_bars(bounds)
    if bars.empty or len(bars) < 5:
        rec.outcome = "skip_no_bars"
        return rec, None

    try:
        ticks = load_session_ticks(bounds)
    except Exception:  # noqa: BLE001
        ticks = pd.DataFrame()
    profile, vp_mode = build_developing_profile(bars, ticks, mode="auto")
    rec.vp_mode = vp_mode

    OR = compute_OR(bars, bounds)
    if OR.skip:
        rec.outcome = f"no_trade:or_{OR.skip_reason}"
        return rec, None

    scan = find_first_signal(bars, profile, OR)
    rec.raw_long = scan.raw_long_count
    rec.raw_short = scan.raw_short_count
    rec.comp_long = scan.compound_long_count
    rec.comp_short = scan.compound_short_count
    if scan.signal is None:
        rec.outcome = f"no_trade:{scan.skip_reason}"
        return rec, None

    sig = scan.signal
    rec.side = sig.side
    # Dispatch initial-stop method via CONFIG (or_levels | fixed_points | atr).
    initial_stop, stop_skip = compute_initial_stop_for_signal(
        side=sig.side,
        entry_price=sig.entry_price,
        signal_iloc=sig.signal_bar_idx,
        session_bars=bars,
        OR=OR,
    )
    if stop_skip is not None:
        rec.outcome = f"no_trade:{stop_skip}"
        return rec, None
    sz = compute_position_size(sig.entry_price, initial_stop, sig.side)
    if sz.skip:
        rec.outcome = f"no_trade:{sz.skip_reason}"
        return rec, None

    trade = run_trade(bars, profile, sig, sz)
    rec.outcome = "trade"
    rec.pnl_net = trade.pnl_net
    rec.contracts = sz.contracts
    rec.partial_contracts = sz.partial_contracts
    rec.fills = len(trade.fills)
    rec.exit_reason = trade.exit_reason
    rec.confirming_hit = trade.confirming is not None
    return rec, trade


def run_backtest(
    start_date: str,
    end_date: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> BacktestResult:
    """Run the strategy day-by-day across [start_date, end_date] inclusive."""
    sessions = _nyse_sessions(start_date, end_date)
    result = BacktestResult(start_date=start_date, end_date=end_date)
    daily_pnl_dict: Dict[str, float] = {}

    for i, d in enumerate(sessions):
        try:
            rec, trade = _process_session(d)
        except Exception as e:  # noqa: BLE001
            rec = SessionRecord(
                date=d, outcome=f"ERROR:{type(e).__name__}", side="",
                pnl_net=0.0, contracts=0, partial_contracts=0, fills=0,
                exit_reason="", vp_mode="", raw_long=0, raw_short=0,
                comp_long=0, comp_short=0, confirming_hit=False,
                error=str(e)[:200],
            )
            trade = None

        result.sessions.append(rec)
        daily_pnl_dict[d] = rec.pnl_net
        if trade is not None:
            result.trade_results.append(trade)

        if progress_callback is not None and (i % 25 == 0 or i == len(sessions) - 1):
            progress_callback(i + 1, len(sessions), d)

    result.daily_pnl = pd.Series(daily_pnl_dict, dtype=float)
    result.daily_pnl.index = pd.to_datetime(result.daily_pnl.index)

    skip_counts: Dict[str, int] = {}
    for rec in result.sessions:
        if rec.outcome.startswith("no_trade:"):
            reason = rec.outcome.split(":", 1)[1]
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
        elif rec.outcome == "skip_no_session":
            skip_counts["no_session"] = skip_counts.get("no_session", 0) + 1
        elif rec.outcome == "skip_no_bars":
            skip_counts["no_bars"] = skip_counts.get("no_bars", 0) + 1

    result.metrics = compute_metrics(
        trade_results=result.trade_results,
        skip_counts=skip_counts,
        n_sessions=len(sessions),
        daily_pnl=result.daily_pnl,
    )
    return result