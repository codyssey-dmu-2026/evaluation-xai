"""Reject silent date alignment, missing values and inconsistent account records."""

import numpy as np
import pandas as pd


class DataContractError(ValueError):
    pass


def validate_index(index: pd.Index, name: str = "data") -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise DataContractError(f"{name}: DatetimeIndex required")
    if index.empty or index.hasnans or not index.is_unique:
        raise DataContractError(f"{name}: dates must be nonempty, unique and present")
    if not index.is_monotonic_increasing:
        raise DataContractError(f"{name}: dates must be sorted ascending")
    if not (index == index.normalize()).all():
        raise DataContractError(f"{name}: daily observations must use normalized dates")


def validate_prices(prices: pd.DataFrame, min_assets: int = 1) -> None:
    validate_index(prices.index, "prices")
    if len(prices.columns) < min_assets or not prices.columns.is_unique:
        raise DataContractError("invalid asset count or duplicate assets")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise DataContractError("prices must be finite and positive")


def aligned_series(series: pd.Series, index: pd.Index, name: str) -> np.ndarray:
    validate_index(series.index, name)
    if not series.index.equals(index):
        raise DataContractError(f"{name}: dates must match exactly; no implicit join")
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataContractError(f"{name}: all values must be finite")
    return values


def validate_ledger(ledger: pd.DataFrame) -> None:
    validate_index(ledger.index, "ledger")
    required = {
        "equity_start",
        "equity_end",
        "net_return_simple",
        "net_return_log",
        "commission",
        "slippage",
    }
    if missing := required - set(ledger.columns):
        raise DataContractError(f"ledger missing columns: {sorted(missing)}")
    values = ledger[list(required)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataContractError("ledger contains nonfinite values")
    start = ledger.equity_start.to_numpy()
    end = ledger.equity_end.to_numpy()
    if (start <= 0).any() or (end <= 0).any():
        raise DataContractError(
            "nonpositive equity requires a separate bankruptcy model"
        )
    if not np.allclose(start[1:], end[:-1], rtol=1e-8, atol=1e-8):
        raise DataContractError(
            "ledger is discontinuous or contains external cash flows"
        )
    returns = end / start - 1
    if not np.allclose(returns, ledger.net_return_simple, rtol=0, atol=1e-8):
        raise DataContractError("net return does not match equity")
    if not np.allclose(np.log1p(returns), ledger.net_return_log, rtol=0, atol=1e-8):
        raise DataContractError("log return does not match equity")
    if (ledger[["commission", "slippage"]].to_numpy() < 0).any():
        raise DataContractError("costs must be nonnegative")


def validate_weights(weights: np.ndarray, size: int, max_weight: float) -> None:
    if weights.shape != (size,) or not np.isfinite(weights).all():
        raise DataContractError("invalid weight shape or nonfinite weights")
    if weights.min() < -1e-10 or weights.max() > max_weight + 1e-6:
        raise DataContractError("weights violate long-only or maximum target weight")
    if not np.isclose(weights.sum(), 1, atol=1e-6, rtol=0):
        raise DataContractError("normal portfolio targets must sum to one")
