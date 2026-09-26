"""Deterministic comparison strategies using only the supplied price history."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .validation import DataContractError, validate_weights


@dataclass
class Allocation:
    weights: np.ndarray
    status: str = "ok"
    reason: str | None = None


class Policy(Protocol):
    rebalance: str

    def target(
        self, history: pd.DataFrame, current_weights: np.ndarray
    ) -> Allocation: ...


class EqualWeight:
    def __init__(self, rebalance: str = "monthly"):
        if rebalance not in {"daily", "monthly"}:
            raise ValueError("rebalance must be daily or monthly")
        self.rebalance = rebalance

    def target(self, history, current_weights):
        return Allocation(np.full(history.shape[1], 1 / history.shape[1]))


class MinimumVariance:
    rebalance = "monthly"

    def __init__(self, window: int = 252, max_weight: float = 0.4):
        if window < 2 or not 0 < max_weight <= 1:
            raise ValueError("invalid MVO window or maximum weight")
        self.window, self.max_weight = window, max_weight

    def target(self, history, current_weights):
        n = history.shape[1]
        if n * self.max_weight < 1 - 1e-10:
            raise DataContractError(
                "MVO constraints are infeasible for this asset count"
            )
        if len(history) < self.window + 1:
            raise DataContractError("MVO needs 252 returns and their preceding price")
        returns = history.iloc[-self.window - 1 :].pct_change(fill_method=None).iloc[1:]
        covariance = returns.cov().to_numpy()
        if not np.isfinite(covariance).all():
            raise DataContractError("MVO covariance contains missing values")
        initial = np.full(n, 1 / n)
        result = minimize(
            lambda w: float(w @ covariance @ w),
            initial,
            jac=lambda w: 2 * covariance @ w,
            method="SLSQP",
            bounds=[(0.0, self.max_weight)] * n,
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda w: w.sum() - 1,
                    "jac": lambda w: np.ones(n),
                }
            ],
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        try:
            validate_weights(result.x, n, self.max_weight)
            valid = result.success
        except DataContractError:
            valid = False
        if not valid:
            return Allocation(initial, "fallback", str(result.message))
        return Allocation(np.asarray(result.x))
