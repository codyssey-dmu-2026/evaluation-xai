"""Regression cases reproduced during the November code review."""

import json

import numpy as np
import pandas as pd
import pytest

from evaluation_xai.backtest import run_backtest
from evaluation_xai.baselines import Allocation, EqualWeight
from evaluation_xai.contracts import RiskTag
from evaluation_xai.mission import analyze_monthly_samples, run_mission_file
from evaluation_xai.statistics import aggregate_seeds
from evaluation_xai.validation import DataContractError


@pytest.fixture
def monthly_samples():
    rng = np.random.default_rng(2026)
    rows = []
    for i, month in enumerate(pd.period_range("2022-01", periods=36, freq="M")):
        for strategy in [
            "PPO-simple",
            "PPO-sharpe",
            "PPO-mdd_l1.0",
            "MVO",
            "EqualWeight",
        ]:
            for seed in ([42, 43, 44] if strategy.startswith("PPO") else [None]):
                rows.append(
                    {
                        "month": str(month),
                        "strategy": strategy,
                        "seed": seed,
                        "return": rng.normal(0.01, 0.04),
                        "regime": ["up", "down", "sideways"][i % 3],
                    }
                )
    return pd.DataFrame(rows)


def test_ppo_single_seed_without_label_is_not_a_baseline(monthly_samples):
    data = monthly_samples.loc[
        ~((monthly_samples.strategy == "PPO-simple") & (monthly_samples.seed != 42))
    ].copy()
    data.loc[data.strategy == "PPO-simple", "seed"] = np.nan
    with pytest.raises(DataContractError, match="every expected seed"):
        analyze_monthly_samples(data)


def test_seeded_baselines_cannot_be_averaged_as_ppo(monthly_samples):
    data = monthly_samples.copy()
    data.loc[data.strategy == "MVO", "seed"] = 42
    with pytest.raises(DataContractError, match="baseline"):
        analyze_monthly_samples(data)


def test_daily_dates_cannot_count_as_months(monthly_samples):
    data = monthly_samples.copy()
    data["month"] = data.month.map(
        dict(
            zip(
                data.month.unique(),
                pd.bdate_range("2022-01-03", periods=36).astype(str),
            )
        )
    )
    with pytest.raises(DataContractError, match="month must"):
        analyze_monthly_samples(data)


@pytest.mark.parametrize(
    "month",
    [
        None,
        "2022-1",
        "2022-13",
        "0000-01",
        pd.Timestamp("2022-01-01"),
        pd.Period("2022-01-01", freq="D"),
    ],
)
def test_invalid_month_labels_are_rejected(monthly_samples, month):
    data = monthly_samples.copy()
    data.loc[0, "month"] = month
    with pytest.raises(DataContractError, match="month must"):
        analyze_monthly_samples(data)


def test_duplicate_month_cannot_hide_behind_period_type(monthly_samples):
    duplicate = monthly_samples.iloc[:1].copy()
    duplicate["month"] = pd.Period("2022-01", freq="M")
    with pytest.raises(DataContractError, match="duplicate month"):
        analyze_monthly_samples(pd.concat([monthly_samples, duplicate]))


def test_monthly_periods_keep_regime_join_and_seed_mean(monthly_samples):
    data = monthly_samples.copy()
    data["month"] = pd.PeriodIndex(data.month, freq="M")
    result = analyze_monthly_samples(data)
    assert result["status"] == "ok"
    assert result["holm_family_size"] == 5
    averaged = aggregate_seeds(data, expected_seeds=(42, 43, 44))
    actual = averaged.loc[
        (averaged.month == "2022-01") & (averaged.strategy == "PPO-simple"), "value"
    ].iloc[0]
    expected = monthly_samples.loc[
        (monthly_samples.month == "2022-01")
        & (monthly_samples.strategy == "PPO-simple"),
        "return",
    ].mean()
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("regime", [None, "invalid", "down"])
def test_bad_regime_preserves_two_valid_tests(monthly_samples, regime):
    monthly_samples.loc[0, "regime"] = regime
    result = analyze_monthly_samples(monthly_samples)
    assert result["status"] == "incomplete"
    assert result["tests"]["reward_variants"]["status"] == "ok"
    assert result["tests"]["strategies"]["status"] == "ok"
    assert result["tests"]["strategy_by_regime"]["status"] == "data_invalid"
    assert result["tests"]["strategy_by_regime"]["reason"]
    assert result["holm_status"] == "incomplete_family"
    assert all(
        "p_holm_family" not in e
        for t in result["tests"].values()
        for e in t.get("effects", [])
    )


@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_partial_mission_is_saved_to_disk(monthly_samples, tmp_path, extension):
    monthly_samples.loc[0, "regime"] = None
    source, destination = tmp_path / f"monthly.{extension}", tmp_path / "result.json"
    if extension == "csv":
        monthly_samples.to_csv(source, index=False)
    else:
        monthly_samples.to_parquet(source, index=False)
    run_mission_file(source, destination)
    result = json.loads(destination.read_text())
    assert result["tests"]["reward_variants"]["effects"][0]["p_value"] is not None
    assert result["tests"]["strategies"]["status"] == "ok"
    assert result["tests"]["strategy_by_regime"]["status"] == "data_invalid"


@pytest.fixture
def scheduled_prices():
    prices = pd.DataFrame(
        [[100, 100, 100], [101, 99, 102], [102, 98, 101], [99, 97, 100]],
        index=pd.bdate_range("2024-01-02", periods=4),
        columns=["A", "B", "C"],
    )
    schedule = pd.DataFrame(
        {
            "decision_at": [f"{d.date()}T22:00:00Z" for d in prices.index[:-1]],
            "execution_at": [f"{d.date()}T21:00:00Z" for d in prices.index[1:]],
        },
        index=prices.index[1:],
    )
    return prices, schedule


def test_future_schedule_cannot_authorize_future_tag(scheduled_prices):
    prices, schedule = scheduled_prices
    for field in ("decision_at", "execution_at"):
        schedule[field] = pd.to_datetime(schedule[field]) + pd.DateOffset(years=1)
    tag = RiskTag(
        tag_id="future",
        asset_id="A",
        risk_type="shock",
        severity=0.5,
        confidence=0.9,
        published_at="2024-12-01T00:00:00Z",
        generated_at="2024-12-01T01:00:00Z",
        available_at="2024-12-01T01:00:00Z",
        expires_at="2026-01-01T00:00:00Z",
        source_url="https://example.org",
        tagger_version="test",
    )

    class Tagged(EqualWeight):
        def target(self, history, current_weights):
            pytest.fail("invalid schedule must be rejected before the policy runs")
            return Allocation(np.ones(3) / 3, risk_tag_ids=("future",))

    with pytest.raises(DataContractError, match="price sessions"):
        run_backtest(
            prices,
            Tagged(),
            test_start=prices.index[1],
            test_end=prices.index[-1],
            execution_schedule=schedule,
            risk_tags=[tag],
        )


@pytest.mark.parametrize(
    "error",
    ["naive", "missing", "execution_day", "decision_day", "before_previous_fill"],
)
def test_schedule_errors_rejected_before_any_policy_call(scheduled_prices, error):
    prices, schedule = scheduled_prices
    second = schedule.index[1]
    if error == "naive":
        schedule.loc[second, "decision_at"] = "2024-01-03T22:00:00"
    elif error == "missing":
        schedule.loc[second, "decision_at"] = None
    elif error == "execution_day":
        schedule.loc[second, "execution_at"] = "2024-01-05T21:00:00Z"
    elif error == "decision_day":
        schedule.loc[second, "decision_at"] = "2024-01-04T01:00:00Z"
    else:
        schedule.loc[second, "decision_at"] = "2024-01-03T20:00:00Z"

    class MustNotRun(EqualWeight):
        def target(self, history, current_weights):
            pytest.fail("validate the full schedule before the first policy call")

    with pytest.raises(DataContractError):
        run_backtest(
            prices,
            MustNotRun(),
            test_start=prices.index[1],
            test_end=prices.index[-1],
            execution_schedule=schedule,
        )


def test_session_timezone_handles_utc_date_boundary(scheduled_prices):
    prices, schedule = scheduled_prices
    schedule["decision_at"] = [
        pd.Timestamp(f"{d.date()} 21:00", tz="Asia/Seoul").tz_convert("UTC")
        for d in prices.index[:-1]
    ]
    schedule["execution_at"] = [
        pd.Timestamp(f"{d.date()} 01:00", tz="Asia/Seoul").tz_convert("UTC")
        for d in prices.index[1:]
    ]
    result = run_backtest(
        prices,
        EqualWeight(),
        test_start=prices.index[1],
        test_end=prices.index[-1],
        execution_schedule=schedule,
        session_timezone="Asia/Seoul",
    )
    assert result.decisions.iloc[0].execution_at == "2024-01-02T16:00:00+00:00"
    with pytest.raises(DataContractError, match="price sessions"):
        run_backtest(
            prices,
            EqualWeight(),
            test_start=prices.index[1],
            test_end=prices.index[-1],
            execution_schedule=schedule,
        )
