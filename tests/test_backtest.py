import numpy as np
import pandas as pd
import pytest

from evaluation_xai.backtest import rebalance_with_costs, run_backtest
from evaluation_xai.baselines import Allocation, EqualWeight, MinimumVariance
from evaluation_xai.config import SimulationConfig, annual_folds
from evaluation_xai.validation import DataContractError
from evaluation_xai.walk_forward import run_walk_forward


def test_initial_trade_cost():
    holdings, cash, traded, fee, slip = rebalance_with_costs(
        100, np.zeros(3), np.ones(3) / 3, 0.00015, 0.0005
    )
    assert holdings.sum() == pytest.approx(100 / 1.00065)
    assert fee + slip == pytest.approx(traded * 0.00065)
    assert cash == pytest.approx(0)


def test_liquidation_cost():
    holdings, cash, traded, fee, slip = rebalance_with_costs(
        100, np.array([30, 30, 40]), np.zeros(3), 0.00015, 0.0005
    )
    assert traded == 100
    assert cash == pytest.approx(99.935)
    assert holdings.sum() == 0


@pytest.mark.parametrize("fee,slip", [(-0.01, 0.1), (0.1, -0.01), (0.5, 0.5)])
def test_invalid_costs(fee, slip):
    with pytest.raises(DataContractError):
        rebalance_with_costs(100, np.zeros(3), np.ones(3) / 3, fee, slip)


def test_leverage_rejected():
    with pytest.raises(DataContractError, match="borrowed cash"):
        rebalance_with_costs(100, np.array([100, 50]), np.array([0.5, 0.5]), 0, 0)


def test_backtest_conservation(prices):
    result = run_backtest(
        prices, EqualWeight(), test_start="2024-01-01", test_end="2024-12-31"
    )
    assert np.allclose(result.weights.sum(axis=1), 1)
    assert result.weights.CASH.iloc[-1] == pytest.approx(1)
    assert len(result.ledger) == len(prices.loc["2024"])


def test_policy_cannot_see_execution_day(prices):
    class Spy(EqualWeight):
        def __init__(self):
            super().__init__("daily")
            self.last_seen = []

        def target(self, history, weights):
            self.last_seen.append(history.index[-1])
            return super().target(history, weights)

    policy = Spy()
    result = run_backtest(
        prices,
        policy,
        test_start="2024-01-01",
        test_end="2024-01-10",
        config=SimulationConfig(apply_guard=False),
    )
    for seen, execution in zip(policy.last_seen, result.ledger.index):
        assert seen < execution


def test_guard_keeps_full_horizon_and_next_close_liquidation():
    prices = pd.DataFrame(
        np.repeat(np.array([100, 100, 80, 70, 75])[:, None], 3, axis=1),
        index=pd.bdate_range("2024-01-01", periods=5),
        columns=["a", "b", "c"],
    )
    result = run_backtest(
        prices,
        EqualWeight(),
        test_start="2024-01-02",
        test_end="2024-01-05",
        config=SimulationConfig(
            commission_rate=0, slippage_rate=0, liquidate_at_end=False
        ),
    )
    assert len(result.ledger) == 4
    assert result.ledger.safeguard_triggered.tolist() == [False, True, False, False]
    assert result.weights.CASH.iloc[1] == pytest.approx(0)
    assert result.weights.CASH.iloc[2] == 1
    assert result.ledger.equity_end.iloc[-1] == pytest.approx(70_000)


def test_mvo_constraints(prices):
    result = MinimumVariance().target(prices.iloc[:300], np.zeros(3))
    assert result.status == "ok"
    assert result.weights.sum() == pytest.approx(1)
    assert np.max(result.weights) <= 0.4 + 1e-6


def test_mvo_needs_history(prices):
    with pytest.raises(DataContractError):
        MinimumVariance().target(prices.iloc[:200], np.zeros(3))


def test_mvo_fallback_is_recorded(prices, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "evaluation_xai.baselines.minimize",
        lambda *a, **k: SimpleNamespace(
            x=np.array([1.0, 0.0, 0.0]), success=False, message="test failure"
        ),
    )
    result = MinimumVariance().target(prices, np.zeros(3))
    assert result.status == "fallback"
    assert result.reason == "test failure"


def test_invalid_policy_weights(prices):
    class Bad(EqualWeight):
        def target(self, history, current_weights):
            return Allocation(np.array([1, 0, 0]))

    with pytest.raises(DataContractError):
        run_backtest(prices, Bad(), test_start="2024-01-01", test_end="2024-01-05")


def test_fold_factory_only_gets_train_data():
    from evaluation_xai.demo import synthetic_prices

    prices = synthetic_prices(assets=3)
    seen = []

    def factory(train, fold):
        assert train.index.min().date() >= fold.train_start
        assert train.index.max().date() <= fold.train_end
        seen.append(fold.id)
        return EqualWeight()

    result = run_walk_forward(prices, annual_folds(2022, 2023), factory)
    assert list(result) == seen == ["WF2022", "WF2023"]


def test_fold_reuse_rejected():
    from evaluation_xai.demo import synthetic_prices

    policy = EqualWeight()
    with pytest.raises(DataContractError, match="reused"):
        run_walk_forward(
            synthetic_prices(assets=3), annual_folds(2022, 2023), lambda *args: policy
        )


def test_future_prices_cannot_change_past_orders(prices):
    changed = prices.copy()
    changed.loc["2024-06-01":] *= 2
    args = dict(test_start="2024-01-01", test_end="2024-12-31")
    a = run_backtest(prices, MinimumVariance(), **args)
    b = run_backtest(changed, MinimumVariance(), **args)
    pd.testing.assert_frame_equal(
        a.ledger.loc[:"2024-05-31"], b.ledger.loc[:"2024-05-31"]
    )
