"""The caller supplies a fresh trainer/factory; test data never enters training."""

from collections.abc import Callable, Sequence

import pandas as pd

from .backtest import BacktestResult, run_backtest
from .baselines import Policy
from .config import Fold, SimulationConfig
from .validation import DataContractError, validate_prices


def run_walk_forward(
    prices: pd.DataFrame,
    folds: Sequence[Fold],
    train_factory: Callable[[pd.DataFrame, Fold], Policy],
    config: SimulationConfig | None = None,
) -> dict[str, BacktestResult]:
    validate_prices(prices)
    if not folds or len({f.id for f in folds}) != len(folds):
        raise DataContractError("folds must be nonempty with unique IDs")
    previous_end = None
    results = {}
    seen_policies = []
    for fold in folds:
        if previous_end is not None and fold.test_start <= previous_end:
            raise DataContractError("test folds must be sorted and non-overlapping")
        train = prices.loc[str(fold.train_start) : str(fold.train_end)].copy()
        if train.empty:
            raise DataContractError(f"{fold.id}: empty training data")
        policy = train_factory(train, fold)
        if any(policy is old for old in seen_policies):
            raise DataContractError("factory reused a policy object across folds")
        seen_policies.append(policy)
        results[fold.id] = run_backtest(
            prices.loc[: str(fold.test_end)],
            policy,
            test_start=fold.test_start,
            test_end=fold.test_end,
            config=config,
        )
        previous_end = fold.test_end
    return results
