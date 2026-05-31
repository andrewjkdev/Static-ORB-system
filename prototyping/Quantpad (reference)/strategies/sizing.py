"""Module F: Position sizing (fixed-fractional risk model).

contracts = floor( (account * risk_pct) / (stop_distance_ticks * tick_value) )

Edge cases:
  - contracts < 1 -> skip session (risk_too_small).
  - contracts == 1 -> partial=0; full position rides the trailer.
    (round(0.60 * 1) = 1, but we clamp partial <= contracts - 1 so the
    runner always has at least 1 contract. So single-contract trades
    always have partial=0 and full position rides the trail.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import CONFIG, TICK_SIZE, TICK_VALUE


@dataclass(frozen=True)
class PositionSize:
    contracts: int
    partial_contracts: int
    runner_contracts: int
    stop_distance_price: float
    stop_distance_ticks: int
    per_contract_risk_dollars: float
    risk_dollars_actual: float
    skip: bool
    skip_reason: Optional[str]


def compute_position_size(
    entry_price: float,
    initial_stop: float,
    side: str,
) -> PositionSize:
    """Compute contracts from entry, initial stop, and CONFIG risk budget."""
    risk_budget = CONFIG["account_size"] * CONFIG["risk_pct_per_trade"]
    dist = (entry_price - initial_stop) if side == "long" else (initial_stop - entry_price)
    if dist <= 0:
        return PositionSize(0, 0, 0, dist, 0, 0.0, 0.0, True, "invalid_stop_distance")
    dist_ticks = int(round(dist / TICK_SIZE))
    per_contract_risk = dist_ticks * TICK_VALUE
    if per_contract_risk <= 0:
        return PositionSize(0, 0, 0, dist, 0, 0.0, 0.0, True, "invalid_stop_distance")
    contracts = int(risk_budget // per_contract_risk)
    if contracts < 1:
        return PositionSize(0, 0, 0, dist, dist_ticks,
                            per_contract_risk, 0.0, True, "risk_too_small")
    frac = CONFIG["partial_exit_fraction"]
    mode = CONFIG.get("partial_qty_rounding", "floor")
    raw = frac * contracts
    partial = int(round(raw)) if mode == "round" else int(raw)
    # Always leave at least 1 contract on the runner (so we don't fully exit at partial).
    # This makes single-contract trades naturally skip the partial leg.
    partial = max(0, min(partial, contracts - 1))
    runner = contracts - partial
    actual_risk = contracts * per_contract_risk
    return PositionSize(
        contracts=contracts,
        partial_contracts=partial,
        runner_contracts=runner,
        stop_distance_price=dist,
        stop_distance_ticks=dist_ticks,
        per_contract_risk_dollars=per_contract_risk,
        risk_dollars_actual=actual_risk,
        skip=False,
        skip_reason=None,
    )