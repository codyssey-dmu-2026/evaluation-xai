"""Twelve net performance metrics with explicit undefined-value reasons."""

from datetime import date

import numpy as np
import pandas as pd

from .validation import DataContractError, aligned_series, validate_ledger

METRIC_NAMES = (
    "cumulative_return",
    "cagr",
    "annualized_volatility",
    "var_95",
    "cvar_95",
    "mdd",
    "sharpe",
    "sortino",
    "calmar",
    "alpha",
    "beta",
    "information_ratio",
)


def compute_metrics(
    ledger: pd.DataFrame,
    benchmark: pd.Series,
    risk_free: pd.Series,
    *,
    period_start: date | pd.Timestamp,
    annualization: int = 252,
) -> dict:
    validate_ledger(ledger)
    b = aligned_series(benchmark, ledger.index, "benchmark")
    rf = aligned_series(risk_free, ledger.index, "risk_free")
    if (b <= -1).any() or (rf <= -1).any():
        raise DataContractError(
            "benchmark and RF must be simple returns greater than -1"
        )
    start = pd.Timestamp(period_start)
    end = ledger.index[-1]
    if start.tzinfo != end.tzinfo:
        raise DataContractError(
            "period_start and ledger dates must use the same timezone"
        )
    years = (end - start).total_seconds() / (365.25 * 86400)
    if years <= 0 or annualization <= 0:
        raise DataContractError(
            "positive evaluation duration and annualization required"
        )
    r = ledger.net_return_simple.to_numpy(dtype=float)
    excess = r - rf
    active = r - b
    equity = np.r_[ledger.equity_start.iloc[0], ledger.equity_end.to_numpy()]
    cumulative = equity[-1] / equity[0] - 1
    cagr = np.expm1(np.log(equity[-1] / equity[0]) / years)
    mdd = float(np.max(1 - equity / np.maximum.accumulate(equity)))
    losses = -r
    var = float(np.quantile(losses, 0.95, method="linear"))
    results: dict[str, dict] = {}

    def put(name, value=None, reason=None):
        if value is not None and not np.isfinite(value):
            value, reason = None, "nonfinite_result"
        results[name] = {
            "value": None if value is None else float(value),
            "status": "ok" if value is not None else "undefined",
            "reason": reason,
        }

    def ratio(name, numerator, denominator, scale=1):
        if denominator <= 1e-12:
            put(name, reason="zero_denominator")
        else:
            put(name, numerator / denominator * scale)

    put("cumulative_return", cumulative)
    put("cagr", cagr)
    put("var_95", var)
    put("cvar_95", losses[losses >= var].mean())
    put("mdd", mdd)
    ratio("calmar", cagr, mdd)
    if len(r) < 2:
        for name in (
            "annualized_volatility",
            "sharpe",
            "sortino",
            "alpha",
            "beta",
            "information_ratio",
        ):
            put(name, reason="insufficient_sample")
    else:
        scale = np.sqrt(annualization)
        put("annualized_volatility", np.std(r, ddof=1) * scale)
        ratio("sharpe", excess.mean(), np.std(excess, ddof=1), scale)
        downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
        ratio("sortino", excess.mean(), downside, scale)
        ratio("information_ratio", active.mean(), np.std(active, ddof=1), scale)
        market_excess = b - rf
        if np.std(market_excess, ddof=1) <= 1e-12:
            put("alpha", reason="constant_market_excess_return")
            put("beta", reason="constant_market_excess_return")
        else:
            design = np.column_stack([np.ones(len(r)), market_excess])
            alpha, beta = np.linalg.lstsq(design, excess, rcond=None)[0]
            put("alpha", alpha * annualization)
            put("beta", beta)
    return {
        "metric_spec_version": "1.0",
        "n_observations": len(r),
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "return_type": "net_simple",
        "annualization_days": annualization,
        "metrics": {name: results[name] for name in METRIC_NAMES},
    }
