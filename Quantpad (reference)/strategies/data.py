"""Module A: Data ingestion + Opening Range calculation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Optional

import pandas as pd
import exchange_calendars as xcals
import quantpad_data as qpd

from .config import CONFIG, TICK_SIZE

_NYSE = xcals.get_calendar("XNYS")


@dataclass(frozen=True)
class SessionBounds:
    date_iso: str
    open_ny: pd.Timestamp
    close_ny: pd.Timestamp
    open_ms: int
    close_ms: int
    is_half_day: bool


@dataclass(frozen=True)
class OpeningRange:
    high: float
    low: float
    range_price: float
    range_ticks: int
    ts: pd.Timestamp
    skip: bool
    skip_reason: Optional[str]


def session_rth_bounds(date_iso: str) -> Optional[SessionBounds]:
    """RTH window for date_iso. None if market closed. Honors NYSE half-days."""
    ts = pd.Timestamp(date_iso).normalize()
    if not _NYSE.is_session(ts):
        return None
    open_utc = _NYSE.session_open(ts).tz_convert("UTC")
    close_utc = _NYSE.session_close(ts).tz_convert("UTC")
    open_ny = open_utc.tz_convert(CONFIG["session_tz"])
    close_ny = close_utc.tz_convert(CONFIG["session_tz"])
    is_half_day = close_ny.time() != time(16, 0)
    return SessionBounds(
        date_iso=date_iso,
        open_ny=open_ny,
        close_ny=close_ny,
        open_ms=int(open_utc.value // 1_000_000),
        close_ms=int(close_utc.value // 1_000_000),
        is_half_day=is_half_day,
    )


def load_session_bars(bounds: SessionBounds) -> pd.DataFrame:
    """Fetch 1m bars for one RTH session, clipped to [open_ms, close_ms)."""
    df = qpd.get_bars(
        symbol=CONFIG["instrument"],
        timeframe="1m",
        start_ms=bounds.open_ms,
        end_ms=bounds.close_ms,
    )
    if df.empty:
        return df
    start = pd.Timestamp(bounds.open_ms, unit="ms", tz="UTC")
    end = pd.Timestamp(bounds.close_ms, unit="ms", tz="UTC")
    df = df.loc[(df.index >= start) & (df.index < end)].copy()
    return df


def load_session_ticks(bounds: SessionBounds) -> pd.DataFrame:
    """Fetch tick trades for one RTH session via streaming chunks."""
    chunks = []
    for chunk in qpd.get_ticks(
        symbol=CONFIG["instrument"],
        schema="trades",
        start_ms=bounds.open_ms,
        end_ms=bounds.close_ms,
    ):
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=["t", "price", "size", "side"])
    df = pd.concat(chunks, ignore_index=True)
    if not pd.api.types.is_datetime64_any_dtype(df["t"]):
        df["t"] = pd.to_datetime(df["t"], unit="ns", utc=True)
    start = pd.Timestamp(bounds.open_ms, unit="ms", tz="UTC")
    end = pd.Timestamp(bounds.close_ms, unit="ms", tz="UTC")
    df = df.loc[(df["t"] >= start) & (df["t"] < end)].reset_index(drop=True)
    return df


def compute_OR(session_bars: pd.DataFrame, bounds: SessionBounds) -> OpeningRange:
    """Compute Opening Range from the first N=or_lookback_minutes bars."""
    n = CONFIG["or_lookback_minutes"]
    if len(session_bars) < n:
        return OpeningRange(
            high=float("nan"), low=float("nan"),
            range_price=float("nan"), range_ticks=0,
            ts=bounds.open_ny,
            skip=True, skip_reason="no_data",
        )
    or_slice = session_bars.iloc[:n]
    hi = float(or_slice["h"].max())
    lo = float(or_slice["l"].min())
    rng = hi - lo
    rng_ticks = int(round(rng / TICK_SIZE))
    or_close_utc = or_slice.index[-1] + pd.Timedelta(minutes=1)
    or_close_ny = or_close_utc.tz_convert(CONFIG["session_tz"])
    skip = False
    reason: Optional[str] = None
    if rng_ticks <= CONFIG["doji_tolerance_ticks"]:
        skip, reason = True, "doji"
    elif CONFIG["skip_if_or_range_ticks_gt"] is not None and rng_ticks > CONFIG["skip_if_or_range_ticks_gt"]:
        skip, reason = True, "or_range_too_wide"
    return OpeningRange(
        high=hi, low=lo, range_price=rng, range_ticks=rng_ticks,
        ts=or_close_ny,
        skip=skip, skip_reason=reason,
    )