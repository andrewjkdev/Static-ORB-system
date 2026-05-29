"""Module D: Stop-loss state machine.

States:  INITIAL -> BREAKEVEN -> TRAILING

  INITIAL    Initial stop = entry_price -/+ HARD_STOP_POINTS (fixed; per
             Andrew's revised spec, this REPLACES the OR-Low/OR-High initial
             stop).
  BREAKEVEN  Armed when a bar CLOSES at or past validation_level, where
             validation_level = entry +/- (hard_stop_distance * validation_multiple).
             With validation_multiple=1.0 this is a clean 1R-to-breakeven rule.
             Stop then moves to entry_price.
  TRAILING   After BE arm, on each new bar j, stop ratchets to the
             LOW of the prior completed bar j-1 (long) / HIGH (short).
             Never moves adversely.

Stop-hit detection: `check_stop_hit_close(bar_close)` -- bar must
CLOSE through the stop level (no intra-bar touches), per Andrew's pick.
The trade engine schedules the actual exit fill at NEXT bar's open
with adverse slippage.

Module E (exits) may call `tighten_to(level)` to force the trail to
a specific price after a confirming breakout. Like all stop moves,
it only ratchets favorably.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .config import CONFIG


@dataclass
class StopState:
    side: str
    entry_price: float
    hard_stop_distance: float    # positive number, in price units (e.g. 30.0 NQ points)

    stop: float = field(init=False)
    state: str = field(default="INITIAL", init=False)
    breakeven_ts: Optional[pd.Timestamp] = field(default=None, init=False)
    trailing_first_move_ts: Optional[pd.Timestamp] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.stop = (
            self.entry_price - self.hard_stop_distance
            if self.side == "long"
            else self.entry_price + self.hard_stop_distance
        )

    @property
    def validation_level(self) -> float:
        delta = self.hard_stop_distance * CONFIG["validation_multiple"]
        return (
            self.entry_price + delta
            if self.side == "long"
            else self.entry_price - delta
        )

    def check_breakeven_arm(self, bar_close: float, bar_ts: pd.Timestamp) -> bool:
        """Move stop to entry_price if bar closed past the validation level."""
        if self.state != "INITIAL":
            return False
        if self.side == "long" and bar_close >= self.validation_level:
            self.stop = self.entry_price
            self.state = "BREAKEVEN"
            self.breakeven_ts = bar_ts
            return True
        if self.side == "short" and bar_close <= self.validation_level:
            self.stop = self.entry_price
            self.state = "BREAKEVEN"
            self.breakeven_ts = bar_ts
            return True
        return False

    def update_trail(
        self,
        prev_completed_bar_high: float,
        prev_completed_bar_low: float,
        bar_ts: pd.Timestamp,
    ) -> bool:
        """Swing-trail ratchet using the most recent COMPLETED bar's extreme.

        Inactive while state == INITIAL (only meaningful after BE arm)."""
        if self.state == "INITIAL":
            return False
        moved = False
        if self.side == "long":
            new_stop = max(self.stop, prev_completed_bar_low)
            if new_stop > self.stop:
                self.stop = new_stop
                moved = True
        else:
            new_stop = min(self.stop, prev_completed_bar_high)
            if new_stop < self.stop:
                self.stop = new_stop
                moved = True
        if moved and self.state == "BREAKEVEN":
            self.state = "TRAILING"
            self.trailing_first_move_ts = bar_ts
        return moved

    def tighten_to(self, level: float, bar_ts: pd.Timestamp) -> bool:
        """Force tighten to `level` (e.g. confirming-breakout candle wick)."""
        moved = False
        if self.side == "long":
            if level > self.stop:
                self.stop = level
                moved = True
        else:
            if level < self.stop:
                self.stop = level
                moved = True
        if moved and self.state == "BREAKEVEN":
            self.state = "TRAILING"
            self.trailing_first_move_ts = bar_ts
        return moved

    def check_stop_hit_close(self, bar_close: float) -> bool:
        """Close-through stop trigger."""
        if self.side == "long":
            return bar_close <= self.stop
        return bar_close >= self.stop