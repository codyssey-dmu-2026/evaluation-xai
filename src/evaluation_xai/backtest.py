"""Small reference ledger for testing contracts before the RL engine arrives.

Daily adjusted-price simulation with next-session-close execution. It is not
an order-book simulator or the team's Gymnasium training environment.
"""

from dataclasses import dataclass

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


def run_backtest(
    prices: pd.DataFrame,
    policy: Policy,
    *,
    test_start,
    test_end,
    config: SimulationConfig | None = None,
) -> BacktestResult:
    config = config or SimulationConfig()
    validate_prices(prices)
    start, end = pd.Timestamp(test_start), pd.Timestamp(test_end)
    test = prices.loc[start:end]
    if test.empty:
        raise DataContractError("no test prices in requested interval")
    first_pos = prices.index.get_loc(test.index[0])
    if first_pos == 0:
        raise DataContractError("at least one pre-test observation is required")
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
