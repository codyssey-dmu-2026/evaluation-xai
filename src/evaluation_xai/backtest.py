"""Small reference ledger for testing contracts before the RL engine arrives.

Daily adjusted-price simulation with next-session-close execution. It is not
an order-book simulator or the team's Gymnasium training environment.
"""

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .baselines import Policy
from .config import SimulationConfig
from .validation import (
    DataContractError,
    validate_ledger,
    validate_prices,
    validate_weights,
)


@dataclass
class BacktestResult:
    ledger: pd.DataFrame
    weights: pd.DataFrame
    decisions: pd.DataFrame
    period_start: pd.Timestamp


def rebalance_with_costs(
    equity, existing_notional, target, commission_rate, slippage_rate
):
    """Solve post-trade equity = pre-trade equity - costs, avoiding negative cash."""
    rate = commission_rate + slippage_rate
    target = np.asarray(target, dtype=float)
    existing_notional = np.asarray(existing_notional, dtype=float)
    if (
        not np.isfinite(equity)
        or equity <= 0
        or not 0 <= rate < 1
        or commission_rate < 0
        or slippage_rate < 0
    ):
        raise DataContractError("invalid equity or combined cost rate")
    if target.shape != existing_notional.shape or not np.isfinite(target).all():
        raise DataContractError("invalid rebalance target")
    if (target < 0).any() or target.sum() > 1 + 1e-8:
        raise DataContractError("rebalance target requires leverage or short positions")
    if not np.isfinite(existing_notional).all() or (existing_notional < 0).any():
        raise DataContractError("invalid existing holdings")
    if existing_notional.sum() > equity + 1e-7:
        raise DataContractError("existing holdings imply borrowed cash")

    def balance(post):
        return post + rate * np.abs(post * target - existing_notional).sum() - equity

    post = brentq(balance, 0, equity, xtol=1e-8) if rate else equity
    notional = post * target
    traded = float(np.abs(notional - existing_notional).sum())
    cash = post - notional.sum()
    if cash < -1e-7:
        raise DataContractError("cost-aware rebalance created negative cash")
    return (
        notional,
        max(0.0, cash),
        traded,
        traded * commission_rate,
        traded * slippage_rate,
    )


def _validate_execution_schedule(prices, test_index, schedule, session_timezone):
    """Bind aware timestamps to daily session labels before calling any policy.

    Labels use the declared session timezone (index timezone, or UTC, by default).
    This validates ordering, not exchange close times or raw data availability.
    """
    if not schedule.index.equals(test_index) or not schedule.columns.is_unique:
        raise DataContractError("execution schedule dates must match test exactly")
    if not {"decision_at", "execution_at"}.issubset(schedule):
        raise DataContractError("schedule requires decision_at and execution_at")
    try:
        zone = ZoneInfo(session_timezone or str(prices.index.tz or "UTC"))
    except (ValueError, KeyError, TypeError) as exc:
        raise DataContractError("invalid session timezone") from exc
    validated, prior_execution = {}, None
    for day in test_index:
        previous_day = prices.index[prices.index.get_loc(day) - 1]
        try:
            decision = pd.Timestamp(schedule.loc[day, "decision_at"])
            execution = pd.Timestamp(schedule.loc[day, "execution_at"])
        except (ValueError, TypeError) as exc:
            raise DataContractError("invalid decision/execution timestamp") from exc
        if (
            pd.isna(decision)
            or pd.isna(execution)
            or decision.tzinfo is None
            or execution.tzinfo is None
            or decision >= execution
        ):
            raise DataContractError(
                "decision/execution timestamps must be aware and ordered"
            )
        if (
            decision.tz_convert(zone).date() != previous_day.date()
            or execution.tz_convert(zone).date() != day.date()
        ):
            raise DataContractError(
                "timestamps must match prior/current price sessions"
            )
        if prior_execution is not None and decision < prior_execution:
            raise DataContractError("decision precedes previous session execution")
        validated[day] = (decision.tz_convert("UTC"), execution.tz_convert("UTC"))
        prior_execution = execution
    return validated


def run_backtest(
    prices: pd.DataFrame,
    policy: Policy,
    *,
    test_start,
    test_end,
    config: SimulationConfig | None = None,
    execution_schedule: pd.DataFrame | None = None,
    session_timezone: str | None = None,
    risk_tags=(),
) -> BacktestResult:
    config = config or SimulationConfig()
    validate_prices(prices)
    start, end = pd.Timestamp(test_start), pd.Timestamp(test_end)
    test = prices.loc[start:end]
    if test.empty:
        raise DataContractError("no test prices in requested interval")
    registry = {tag.tag_id: tag for tag in risk_tags}
    if len(registry) != len(risk_tags):
        raise DataContractError("duplicate risk tag IDs")
    first_pos = prices.index.get_loc(test.index[0])
    if first_pos == 0:
        raise DataContractError("at least one pre-test observation is required")
    schedule_times = (
        _validate_execution_schedule(
            prices, test.index, execution_schedule, session_timezone
        )
        if execution_schedule is not None
        else None
    )
    if prices.shape[1] * config.max_asset_weight < 1 - 1e-10:
        raise DataContractError("asset count cannot satisfy target weight constraints")
    if policy.rebalance not in {"daily", "monthly"}:
        raise DataContractError("unknown rebalance schedule")
    period_start = prices.index[first_pos - 1]
    units = np.zeros(prices.shape[1])
    cash = previous_equity = peak = config.initial_equity
    stopped = False
    rows, weight_rows, decisions = [], [], []
    last_date = test.index[-1]

    for date, price_row in test.iterrows():
        pos = prices.index.get_loc(date)
        previous_date = prices.index[pos - 1]
        if execution_schedule is None:
            # Ordering placeholders for synthetic/reference runs, NOT exchange times.
            decision_at = (
                pd.Timestamp(previous_date).tz_localize("UTC")
                if previous_date.tzinfo is None
                else previous_date.tz_convert("UTC")
            )
            execution_at = (
                pd.Timestamp(date).tz_localize("UTC")
                if date.tzinfo is None
                else date.tz_convert("UTC")
            )
            time_status = "assumed_daily_order_not_exchange_clock"
        else:
            decision_at, execution_at = schedule_times[date]
            time_status = "caller_supplied_schedule"
        used_tag_ids = []
        previous_price = prices.iloc[pos - 1].to_numpy()
        current = units * previous_price / previous_equity
        is_month_start = previous_date.month != date.month
        regular = date == test.index[0] or policy.rebalance == "daily" or is_month_start
        final = date == last_date and config.liquidate_at_end
        target, reason, policy_status = None, None, "not_requested"
        policy_weights = None
        if final or stopped:
            target = np.zeros(prices.shape[1])
            reason = "test_end" if final and not stopped else "mdd_guard"
        elif regular:
            # A copy prevents a policy from modifying the shared market table.
            allocation = policy.target(prices.iloc[:pos].copy(), current.copy())
            used_tag_ids = list(allocation.risk_tag_ids)
            if len(set(used_tag_ids)) != len(used_tag_ids):
                raise DataContractError("duplicate decision risk tag IDs")
            for tag_id in used_tag_ids:
                if execution_schedule is None:
                    raise DataContractError(
                        "risk-tag attribution requires explicit timestamps"
                    )
                if tag_id not in registry or not registry[tag_id].usable_at(
                    decision_at
                ):
                    raise DataContractError("unknown, future or expired risk tag")
                if registry[tag_id].asset_id not in prices.columns:
                    raise DataContractError("risk tag asset not in portfolio")
            validate_weights(
                allocation.weights, prices.shape[1], config.max_asset_weight
            )
            target = allocation.weights
            policy_weights = target.copy()
            policy_status, reason = allocation.status, allocation.reason
        market_price = price_row.to_numpy()
        existing = units * market_price
        before_trade_equity = float(existing.sum() + cash)
        commission = slippage = traded = 0.0
        if target is not None:
            notionals, cash, traded, commission, slippage = rebalance_with_costs(
                before_trade_equity,
                existing,
                target,
                config.commission_rate,
                config.slippage_rate,
            )
            units = notionals / market_price
        equity = float(np.dot(units, market_price) + cash)
        if equity <= 0:
            raise DataContractError("account insolvent")
        peak = max(peak, equity)
        drawdown = 1 - equity / peak
        triggered = (
            config.apply_guard and not stopped and drawdown > config.drawdown_threshold
        )
        if triggered:
            stopped = True
        r = equity / previous_equity - 1
        rows.append(
            {
                "date": date,
                "equity_start": previous_equity,
                "equity_end": equity,
                "net_return_simple": r,
                "net_return_log": np.log1p(r),
                "traded_notional": traded,
                "commission": commission,
                "slippage": slippage,
                "drawdown": drawdown,
                "safeguard_triggered": bool(triggered),
                "guard_active": bool(stopped),
                "termination_reason": "mdd" if triggered else None,
                "cash": cash,
            }
        )
        actual = units * market_price / equity
        weight_rows.append(
            {"date": date, **dict(zip(prices.columns, actual)), "CASH": cash / equity}
        )
        if target is not None:
            decisions.append(
                {
                    "decision_id": f"d{date.strftime('%Y%m%d')}",
                    "decision_date": previous_date,
                    "execution_date": date,
                    "decision_at": decision_at.tz_convert("UTC").isoformat(),
                    "execution_at": execution_at.tz_convert("UTC").isoformat(),
                    "time_status": time_status,
                    "risk_tag_ids": used_tag_ids,
                    "policy_status": policy_status,
                    "reason": reason,
                    "weight_before": current.tolist(),
                    "weight_policy": (
                        None if policy_weights is None else policy_weights.tolist()
                    ),
                    "weight_executed": actual.tolist(),
                    "cash_weight": cash / equity,
                }
            )
        previous_equity = equity
    ledger = pd.DataFrame(rows).set_index("date")
    validate_ledger(ledger)
    return BacktestResult(
        ledger,
        pd.DataFrame(weight_rows).set_index("date"),
        pd.DataFrame(decisions),
        period_start,
    )
