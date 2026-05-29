"""Module G (part 1): Performance metrics for a list of TradeResult objects.

All calcs are pure functions of the trade list + account size. No I/O,
no plotting — just numbers. The backtest module wires this into a report.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import pandas as pd

from .config import CONFIG
from .trade_engine import TradeResult


@dataclass
class PerformanceMetrics:
    n_sessions: int
    n_trades: int
    n_wins: int
    n_losses: int
    n_breakeven: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    net_pnl: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    expectancy: float
    max_drawdown_dollar: float
    max_drawdown_pct_of_account: float
    sharpe_daily: float
    calmar: float
    avg_trade_minutes: float
    pct_confirming_hit: float
    pct_partial_executed: float
    pct_long: float
    n_skip_or_doji: int
    n_skip_or_too_wide: int
    n_skip_risk_too_small: int
    n_skip_no_signal: int
    n_skip_no_session: int
    n_skip_no_bars: int
    n_skip_signal_on_last_bar: int

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_div(a: float, b: float, default: float = float("nan")) -> float:
    return a / b if b != 0 else default


def compute_metrics(
    trade_results: List[TradeResult],
    skip_counts: dict,
    n_sessions: int,
    daily_pnl: pd.Series,
) -> PerformanceMetrics:
    """Aggregate a list of TradeResult into a PerformanceMetrics struct."""
    pnls = np.array([t.pnl_net for t in trade_results], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    breakevens = pnls[pnls == 0]

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(losses.sum()) if len(losses) else 0.0  # negative
    net_pnl = float(pnls.sum()) if len(pnls) else 0.0
    profit_factor = _safe_div(gross_profit, -gross_loss, default=float("inf") if gross_profit > 0 else float("nan"))

    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0  # negative
    n = len(pnls)
    expectancy = _safe_div(net_pnl, n, default=0.0)

    # Equity curve + max drawdown on the chronological daily P&L
    equity = CONFIG["account_size"] + daily_pnl.cumsum()
    running_peak = equity.cummax()
    drawdown = equity - running_peak  # <= 0
    max_dd_dollar = float(drawdown.min()) if len(drawdown) else 0.0
    max_dd_pct = max_dd_dollar / CONFIG["account_size"]

    # Sharpe (daily): mean / std of daily P&L, annualized by sqrt(252)
    daily_returns = daily_pnl / CONFIG["account_size"]
    if daily_returns.std(ddof=1) > 0 and len(daily_returns) > 1:
        sharpe = float(np.sqrt(252) * daily_returns.mean() / daily_returns.std(ddof=1))
    else:
        sharpe = float("nan")

    # Calmar: annualized return / |max_dd_pct|
    if len(daily_pnl) > 0 and abs(max_dd_pct) > 0:
        ann_return = (net_pnl / CONFIG["account_size"]) * (252.0 / len(daily_pnl))
        calmar = ann_return / abs(max_dd_pct)
    else:
        calmar = float("nan")

    # Trade-level diagnostics
    durations = []
    for t in trade_results:
        if len(t.fills) >= 2:
            durations.append((t.fills[-1].ts - t.fills[0].ts).total_seconds() / 60.0)
    avg_dur = float(np.mean(durations)) if durations else 0.0

    pct_conf = _safe_div(sum(1 for t in trade_results if t.confirming is not None), n, default=0.0)
    pct_part = _safe_div(
        sum(1 for t in trade_results if any(f.kind == "partial" for f in t.fills)),
        n, default=0.0,
    )
    pct_long = _safe_div(sum(1 for t in trade_results if t.signal.side == "long"), n, default=0.0)

    return PerformanceMetrics(
        n_sessions=n_sessions, n_trades=n,
        n_wins=int(len(wins)), n_losses=int(len(losses)), n_breakeven=int(len(breakevens)),
        win_rate=_safe_div(len(wins), n, default=0.0),
        gross_profit=gross_profit, gross_loss=gross_loss, net_pnl=net_pnl,
        profit_factor=profit_factor,
        avg_win=avg_win, avg_loss=avg_loss, expectancy=expectancy,
        max_drawdown_dollar=max_dd_dollar,
        max_drawdown_pct_of_account=max_dd_pct,
        sharpe_daily=sharpe, calmar=calmar,
        avg_trade_minutes=avg_dur,
        pct_confirming_hit=pct_conf,
        pct_partial_executed=pct_part,
        pct_long=pct_long,
        n_skip_or_doji=skip_counts.get("or_doji", 0),
        n_skip_or_too_wide=skip_counts.get("or_or_range_too_wide", 0),
        n_skip_risk_too_small=skip_counts.get("risk_too_small", 0),
        n_skip_no_signal=skip_counts.get("no_signal", 0),
        n_skip_no_session=skip_counts.get("no_session", 0),
        n_skip_no_bars=skip_counts.get("no_bars", 0),
        n_skip_signal_on_last_bar=skip_counts.get("signal_on_last_bar", 0),
    )


def fmt_report(m: PerformanceMetrics) -> str:
    """Human-readable text report."""
    def pct(x: float) -> str:
        return "n/a" if x != x else f"{x*100:.1f}%"
    def dol(x: float) -> str:
        return f"${x:,.0f}"
    lines = [
        "=" * 60,
        "  BACKTEST PERFORMANCE REPORT — 1m ORB /NQ / VP-POC compound",
        "=" * 60,
        f"  Sessions processed     : {m.n_sessions}",
        f"  Trades taken           : {m.n_trades}",
        f"  Wins / Losses / BE     : {m.n_wins} / {m.n_losses} / {m.n_breakeven}",
        f"  Win rate               : {pct(m.win_rate)}",
        f"  Long / Short split     : {pct(m.pct_long)} / {pct(1-m.pct_long)}",
        "-" * 60,
        f"  Gross profit / loss    : {dol(m.gross_profit)} / {dol(m.gross_loss)}",
        f"  Net P&L                : {dol(m.net_pnl)}",
        f"  Profit factor          : {m.profit_factor:.2f}" if m.profit_factor == m.profit_factor else "  Profit factor          : n/a",
        f"  Avg win / avg loss     : {dol(m.avg_win)} / {dol(m.avg_loss)}",
        f"  Expectancy / trade     : {dol(m.expectancy)}",
        "-" * 60,
        f"  Max drawdown ($)       : {dol(m.max_drawdown_dollar)}",
        f"  Max drawdown (% acct)  : {pct(m.max_drawdown_pct_of_account)}",
        f"  Sharpe (daily, ann.)   : {m.sharpe_daily:.2f}" if m.sharpe_daily == m.sharpe_daily else "  Sharpe                 : n/a",
        f"  Calmar                 : {m.calmar:.2f}" if m.calmar == m.calmar else "  Calmar                 : n/a",
        "-" * 60,
        f"  Avg trade duration     : {m.avg_trade_minutes:.1f} minutes",
        f"  % trades w/ confirming : {pct(m.pct_confirming_hit)}",
        f"  % trades w/ partial    : {pct(m.pct_partial_executed)}",
        "-" * 60,
        f"  SKIP REASONS  doji={m.n_skip_or_doji}  too_wide={m.n_skip_or_too_wide}  "
        f"risk_too_small={m.n_skip_risk_too_small}  no_signal={m.n_skip_no_signal}  "
        f"no_session={m.n_skip_no_session}  no_bars={m.n_skip_no_bars}  "
        f"signal_last_bar={m.n_skip_signal_on_last_bar}",
        "=" * 60,
    ]
    return "\n".join(lines)