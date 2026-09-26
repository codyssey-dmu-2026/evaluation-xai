import numpy as np
import pandas as pd
import pytest

from evaluation_xai.statistics import (
    aggregate_seeds,
    anova,
    market_regimes,
    monthly_returns,
    paired_block_bootstrap,
)
from evaluation_xai.validation import DataContractError


def test_monthly_compounding():
    r = pd.Series([0.1, -0.1], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert monthly_returns(r).iloc[0] == pytest.approx(-0.01)


def test_anova_and_tukey():
    rng = np.random.default_rng(11)
    data = pd.DataFrame(
        {
            "strategy": np.repeat(["a", "b", "c"], 30),
            "value": np.concatenate([rng.normal(i, 0.1, 30) for i in range(3)]),
        }
    )
    result = anova(data)
    assert result["effects"][0]["p_value"] < 0.05
    assert 0 < result["effects"][0]["eta_squared"] < 1
    assert result["tukey_status"] == "performed"
    assert len(result["tukey"]) == 3


def test_two_way_complete_cells():
    rng = np.random.default_rng(11)
    rows = [
        {"strategy": s, "regime": r, "value": rng.normal() + i}
        for i, s in enumerate(["a", "b", "c"])
        for r in ["up", "down", "sideways"]
        for _ in range(8)
    ]
    result = anova(pd.DataFrame(rows), two_way=True)
    assert result["status"] == "ok"
    assert len(result["effects"]) == 3


def test_sparse_cells_not_fabricated():
    data = pd.DataFrame(
        {"strategy": ["a", "b"], "regime": ["up", "down"], "value": [0.1, 0.2]}
    )
    assert anova(data, two_way=True)["status"] == "insufficient_data"


def test_regime_does_not_use_current_month():
    index = pd.bdate_range("2023-01-01", "2024-12-31")
    prices = pd.Series(np.linspace(100, 200, len(index)), index=index)
    before = market_regimes(prices, ["2024-01"])
    prices.loc["2024-01-01":] *= 0.01
    assert market_regimes(prices, ["2024-01"]) == before


def test_seed_aggregation_and_baseline():
    data = pd.DataFrame(
        {
            "month": ["2024-01"] * 4,
            "strategy": ["ppo"] * 3 + ["mvo"],
            "seed": [11, 22, 33, None],
            "return": [0.01, 0.02, 0.03, 0.01],
        }
    )
    result = aggregate_seeds(data)
    assert len(result) == 2
    assert result.loc[result.strategy == "ppo", "value"].iloc[0] == pytest.approx(0.02)
    with pytest.raises(DataContractError):
        aggregate_seeds(data.iloc[1:])


def test_baseline_replication_rejected():
    data = pd.DataFrame(
        {
            "month": ["2024-01"] * 2,
            "strategy": ["mvo"] * 2,
            "seed": [None, None],
            "return": [0.01, 0.01],
        }
    )
    with pytest.raises(DataContractError):
        aggregate_seeds(data)


def test_paired_bootstrap():
    x = pd.Series(
        np.linspace(-0.01, 0.01, 24),
        index=pd.period_range("2023-01", periods=24, freq="M"),
    )
    result = paired_block_bootstrap(x + 0.01, x, repetitions=200)
    assert result["ci_95"] == pytest.approx([0.01, 0.01])


def test_bootstrap_missing_month_rejected():
    x = pd.Series(0.1, index=pd.period_range("2023-01", periods=24, freq="M")).drop(
        pd.Period("2023-06")
    )
    with pytest.raises(DataContractError, match="missing months"):
        paired_block_bootstrap(x, x)


def test_mission_anova_holm_family():
    from evaluation_xai.statistics import mission_anova

    rng = np.random.default_rng(44)
    rows = [
        {"strategy": s, "regime": r, "value": rng.normal() + i}
        for i, s in enumerate(["a", "b", "c"])
        for r in ["up", "down", "sideways"]
        for _ in range(12)
    ]
    data = pd.DataFrame(rows)
    result = mission_anova(data, data, data)
    assert result["status"] == "ok"
    assert result["holm_family_size"] == 5
    for test in result["tests"].values():
        for effect in test["effects"]:
            assert effect["p_holm_family"] >= effect["p_value"]
    assert mission_anova(data.iloc[:3], data, data)["status"] == "incomplete"
