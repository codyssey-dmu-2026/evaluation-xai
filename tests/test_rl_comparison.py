"""Regression tests independent of optional torch/gym or teammate checkouts."""

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from evaluation_xai.baselines import Allocation, EqualWeight, MinimumVariance
from evaluation_xai.data_loader import ProcessedWindow
from evaluation_xai.real_xai import compare_daily_replay, daily_replay_report
from evaluation_xai import rl_comparison as rc
from evaluation_xai.statistics import regimes_from_log_returns
from evaluation_xai.validation import DataContractError


def action_to_weights(action, max_weight):
    weights = np.exp(action - np.max(action))
    return weights / weights.sum()


class PortfolioEnv:
    """Small independent contract fixture; no gym dependency or copied team source."""

    def __init__(self, features, returns, **settings):
        self.dates = features.index
        self.returns = np.expm1(returns.to_numpy())
        self.n_assets = returns.shape[1]
        for key, value in settings.items():
            setattr(self, key, value)

    def reset(self, seed):
        self.current_step = self.lookback - 1
        self.weights = np.zeros(self.n_assets, dtype=np.float32)
        self.portfolio_value = self.initial_cash
        self.peak = self.initial_cash
        return self.weights.copy(), {}

    def step(self, action):
        target = action_to_weights(action, self.max_weight)
        turnover = float(np.abs(target - self.weights).sum())
        self.current_step += 1
        returns = self.returns[self.current_step]
        gross = float(target @ returns)
        net = gross - turnover * (self.fee_rate + self.slippage)
        self.portfolio_value *= 1 + net
        self.peak = max(self.peak, self.portfolio_value)
        self.weights = (target * (1 + returns) / (1 + gross)).astype(np.float32)
        return (
            self.weights.copy(),
            net,
            False,
            self.current_step == len(self.dates) - 1,
            {
                "date": self.dates[self.current_step],
                "target_weights": target,
                "weights": self.weights.copy(),
                "turnover": turnover,
                "portfolio_value": self.portfolio_value,
                "net_return": net,
                "drawdown": 1 - self.portfolio_value / self.peak,
            },
        )


class ZeroModel:
    def predict(self, observation, deterministic):
        return np.zeros(len(observation)), None


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "rl"
    folder.mkdir()
    (folder / "environment.py").write_text(
        "import numpy as np\n"
        + inspect.getsource(action_to_weights)
        + "\n"
        + inspect.getsource(PortfolioEnv)
    )
    return folder


@pytest.fixture
def data():
    dates = pd.bdate_range("2020-12-01", "2022-12-30")
    rng = np.random.default_rng(7)
    assets = ["A", "B", "SPY"]
    returns = pd.DataFrame(
        rng.normal(0, 0.015, (len(dates), 3)),
        index=dates,
        columns=[f"{a}_return" for a in assets],
    )
    meta = {
        "train_start": "2020-12-01",
        "train_end": "2022-01-01",
        "test_end": "2023-01-01",
        "window": 1,
    }
    return ProcessedWindow(
        None, returns, returns.copy(), pd.DataFrame(), meta, assets, {}
    )


def test_mvo_return_path_matches_price_path(prices):
    strategy = MinimumVariance()
    history = prices.iloc[:300]
    via_prices = strategy.target(history, np.zeros(3))
    log_returns = np.log(history / history.shift()).iloc[1:]
    via_returns = strategy.target_returns(np.expm1(log_returns))
    np.testing.assert_allclose(via_prices.weights, via_returns.weights, atol=1e-10)
    assert via_returns.diagnostics["observations"] == 252
    assert via_returns.diagnostics["success"]
    assert via_returns.diagnostics["constraints_valid"]


@pytest.mark.parametrize(
    "bad", ["short", "nan", "inf", "loss", "duplicate", "order", "infeasible"]
)
def test_mvo_return_inputs_fail_closed(data, bad):
    frame = np.expm1(data.returns.iloc[:252]).copy()
    if bad == "short":
        frame = frame.iloc[1:]
    elif bad in {"nan", "inf", "loss"}:
        frame.iloc[-1, 0] = {"nan": np.nan, "inf": np.inf, "loss": -1}[bad]
    elif bad == "duplicate":
        frame.columns = ["a", "a", "b"]
    elif bad == "order":
        frame = frame.iloc[::-1]
    else:
        frame = frame.iloc[:, :2]
    with pytest.raises(DataContractError):
        MinimumVariance().target_returns(frame)


def test_solver_failure_reports_constraints_separately(data, monkeypatch):
    monkeypatch.setattr(
        "evaluation_xai.baselines.minimize",
        lambda *a, **k: SimpleNamespace(
            x=np.full(3, 1 / 3),
            success=False,
            message="iteration limit",
            nit=1000,
            fun=0.1,
        ),
    )
    result = MinimumVariance().target_returns(np.expm1(data.returns))
    assert result.status == "fallback"
    assert result.diagnostics["constraints_valid"]
    assert not result.diagnostics["success"]
    assert result.reason == "iteration limit"


def test_private_decoder_does_not_change_ppo_module(source):
    original = rc.load_environment(source)
    direct = rc.load_environment(source, direct_weights=True)
    np.testing.assert_array_equal(
        direct.action_to_weights([0, 0.4, 0.6], 0.4), [0, 0.4, 0.6]
    )
    np.testing.assert_allclose(
        original.action_to_weights([0, 0, 0], 0.4), np.full(3, 1 / 3)
    )


def test_monthly_holds_keep_drift_and_zero_cost(source, data):
    # Push held A above 40% immediately; it must not be capped or sold on hold days.
    data.returns.loc["2022-01-04", "A_return"] = np.log(2)
    direct = rc.load_environment(source, direct_weights=True)
    result = rc.replay_run(
        rc.build_environment(direct, data), data, policy=EqualWeight()
    )
    hold = ~result.daily.rebalanced
    assert (
        (result.daily.loc[hold, ["turnover", "commission", "slippage"]] == 0)
        .all()
        .all()
    )
    assert result.daily.iloc[0].net_return == pytest.approx(
        np.expm1(data.returns.loc["2022-01-04"]).mean() - 0.00065
    )
    assert result.targets.loc["2022-01-05", "A"] > 0.4
    np.testing.assert_array_equal(
        result.targets.loc["2022-01-05"], result.holdings.loc["2022-01-04"]
    )
    dates = [d["decision_date"] for d in result.decisions if d["rebalanced"]]
    assert len(dates) == 12
    assert dates[:2] == ["2022-01-03", "2022-02-01"]
    assert result.daily.index[0] == pd.Timestamp("2022-01-04")
    assert result.daily.index[-1] == pd.Timestamp("2022-12-30")


def test_direct_daily_baseline_matches_original_action_kernel(source, data):
    model = rc.replay_run(
        rc.build_environment(rc.load_environment(source), data), data, model=ZeroModel()
    )
    baseline = rc.replay_run(
        rc.build_environment(rc.load_environment(source, direct_weights=True), data),
        data,
        policy=EqualWeight("daily"),
    )
    pd.testing.assert_frame_equal(model.daily, baseline.daily)
    pd.testing.assert_frame_equal(model.targets, baseline.targets)


def test_mvo_history_never_contains_outcome_or_future(source, data):
    direct = rc.load_environment(source, direct_weights=True)
    first = rc.replay_run(
        rc.build_environment(direct, data), data, policy=MinimumVariance()
    )
    for record in first.optimization:
        assert record["observations"] == 252
        assert record["history_end"] <= record["decision_date"] < record["outcome_date"]
    data.returns.loc["2022-06-02":] += 0.3
    second = rc.replay_run(
        rc.build_environment(direct, data), data, policy=MinimumVariance()
    )
    pd.testing.assert_frame_equal(
        first.targets.loc[:"2022-06-02"], second.targets.loc[:"2022-06-02"]
    )


def test_identity_adapter_is_required(source, data):
    class Unequal(EqualWeight):
        def target(self, history, current_weights):
            return Allocation(np.array([0.4, 0.3, 0.3]))

    with pytest.raises(DataContractError, match="decoder changed"):
        rc.replay_run(
            rc.build_environment(rc.load_environment(source), data),
            data,
            policy=Unequal(),
        )


def test_regime_uses_exactly_60_pre_month_log_returns():
    dates = pd.bdate_range("2023-09-01", "2024-02-29")
    benchmark = pd.Series(0.001, index=dates)
    result = regimes_from_log_returns(benchmark, ["2024-01"])
    assert result.iloc[0].trailing_return == pytest.approx(np.expm1(0.06))
    assert result.iloc[0].regime == "up"
    assert result.iloc[0].observations == 60
    assert result.iloc[0].history_end == "2023-12-29"
    benchmark.loc["2024-01-01":] = -5
    pd.testing.assert_frame_equal(
        result, regimes_from_log_returns(benchmark, ["2024-01"])
    )
    benchmark.loc[:"2023-12-31"] = -0.001
    assert regimes_from_log_returns(benchmark, ["2024-01"]).iloc[0].regime == "down"
    benchmark[:] = 0
    assert regimes_from_log_returns(benchmark, ["2024-01"]).iloc[0].regime == "sideways"


@pytest.mark.parametrize("bad", ["short", "nan", "duplicate", "threshold"])
def test_bad_regime_history_is_not_filled(bad):
    benchmark = pd.Series(0.01, index=pd.bdate_range("2023-09-01", periods=80))
    kwargs = {}
    if bad == "short":
        benchmark = benchmark.iloc[:59]
    elif bad == "nan":
        benchmark.iloc[0] = np.nan
    elif bad == "duplicate":
        benchmark = pd.concat([benchmark, benchmark.tail(1)])
    else:
        kwargs["threshold"] = np.nan
    with pytest.raises(DataContractError):
        regimes_from_log_returns(benchmark, ["2024-01"], **kwargs)


def test_alignment_rejects_different_conditions(source, data):
    replay = rc.replay_run(
        rc.build_environment(rc.load_environment(source), data), data, model=ZeroModel()
    )
    with pytest.raises(DataContractError, match="mismatch"):
        rc.assert_aligned(replay.daily.iloc[1:], replay.daily)
    changed = replay.daily.copy()
    changed.iloc[0, changed.columns.get_loc("benchmark_return")] += 0.01
    with pytest.raises(DataContractError, match="mismatch"):
        rc.assert_aligned(changed, replay.daily)


def test_numeric_mismatch_is_audited_and_strict_xai_still_rejects(
    source, data, tmp_path
):
    replay = rc.replay_run(
        rc.build_environment(rc.load_environment(source), data), data, model=ZeroModel()
    )
    reference = replay.daily.assign(window=1, strategy="PPO-simple", seed=42)
    reference.iloc[0, reference.columns.get_loc("drawdown")] += 2e-8
    path = tmp_path / "daily.csv"
    reference.to_csv(path)
    meta = {"window": 1, "variant": "simple", "seed": 42}
    report = daily_replay_report(replay.daily, path, meta)
    assert report["status"] == "mismatch"
    assert report["mismatched_fields"]["drawdown"]["rows"] == 1
    with pytest.raises(DataContractError, match="drawdown"):
        compare_daily_replay(replay.daily, path, meta)


@pytest.fixture
def comparison_setup(source, data, tmp_path, monkeypatch):
    # Exercise orchestration/files/partial results with a small local contract fixture.
    monkeypatch.setattr(rc, "WINDOWS", (1,))
    monkeypatch.setattr(rc, "load_processed_window", lambda *a, **k: data)
    monkeypatch.setattr(
        rc, "load_model_metadata", lambda *a: ({}, {}, data.asset_order)
    )
    replay = rc.replay_run(
        rc.build_environment(rc.load_environment(source), data), data, model=ZeroModel()
    )
    reference = pd.concat(
        [
            replay.daily.assign(window=1, strategy="EqualWeight", seed="-"),
            *[
                replay.daily.assign(window=1, strategy=f"PPO-{v}", seed=str(s))
                for v in rc.REWARD_VARIANTS
                for s in rc.SEEDS
            ],
        ]
    ).reset_index()
    models = tmp_path / "models"
    models.mkdir()
    reference.to_csv(models / "eval_test_daily.csv", index=False)
    monkeypatch.setattr(
        rc,
        "load_daily",
        lambda path: pd.read_csv(path, dtype={"seed": str}, parse_dates=["date"]),
    )
    for variant in rc.REWARD_VARIANTS:
        for seed in rc.SEEDS:
            path = models / f"ppo_test_w1_{variant}_s{seed}.zip"
            path.write_bytes(b"synthetic test stub, not a checkpoint")
            path.with_suffix(".json").write_text(
                json.dumps({"variant": variant, "seed": seed, "window": 1})
            )
    monkeypatch.setattr(
        rc,
        "load_checkpoint",
        lambda path, *a: (
            ZeroModel(),
            json.loads(path.with_suffix(".json").read_text()),
        ),
    )
    processed = tmp_path / "processed"
    processed.mkdir()
    data.returns.to_parquet(processed / "returns.parquet")
    return models, processed, source


def test_comparison_files_and_monthly_only_statistics(comparison_setup, tmp_path):
    output = tmp_path / "result"
    result = rc.run_comparison(*comparison_setup, output)
    assert len(result["runs"]) == 12
    assert result["reference_reconciliation"]["matched_run_count"] == 10
    samples = pd.read_parquet(output / "stat_samples.parquet")
    assert len(samples) == 132  # 9 PPO + 2 baseline runs, 12 months each
    assert not samples.strategy.str.contains("Control").any()
    assert set(samples.loc[samples.strategy == "EqualWeight", "rebalance"]) == {
        "monthly"
    }
    assert samples.loc[samples.strategy == "EqualWeight", "seed"].isna().all()
    assert (output / "MVO_monthly_w1/optimization.json").exists()
    assert (
        len(json.loads((output / "MVO_monthly_w1/optimization.json").read_text())) == 12
    )
    assert result["acceptance"]["status"] == "incomplete"
    original = (output / "evaluation.json").read_bytes()
    with pytest.raises(DataContractError, match="empty"):
        rc.run_comparison(*comparison_setup, output)
    assert (output / "evaluation.json").read_bytes() == original


def test_partial_failure_preserves_completed_runs(
    comparison_setup, tmp_path, monkeypatch
):
    def fail(*args):
        raise DataContractError("checkpoint broken")

    monkeypatch.setattr(rc, "load_checkpoint", fail)
    output = tmp_path / "failed"
    with pytest.raises(DataContractError, match="broken"):
        rc.run_comparison(*comparison_setup, output)
    assert not (output / "evaluation.json").exists()
    report = json.loads((output / "failure.json").read_text())
    assert len(report["completed_runs"]) == 3
    assert (output / "MVO_monthly_w1/ledger.parquet").exists()


def test_regime_failure_keeps_one_way_results(comparison_setup, tmp_path, monkeypatch):
    def fail(*args):
        raise DataContractError("insufficient pre-month history")

    monkeypatch.setattr(rc, "regimes_from_log_returns", fail)
    result = rc.run_comparison(*comparison_setup, tmp_path / "sparse")
    stats = result["mission_anova"]
    assert stats["tests"]["reward_variants"]["status"] == "ok"
    assert stats["tests"]["strategies"]["status"] == "ok"
    assert stats["tests"]["strategy_by_regime"]["status"] == "data_invalid"
    assert stats["status"] == "incomplete"
    assert stats["holm_status"] == "incomplete_family"


def test_reference_mismatch_never_replaces_replayed_samples(comparison_setup, tmp_path):
    models, *_ = comparison_setup
    path = models / "eval_test_daily.csv"
    frame = pd.read_csv(path, dtype={"seed": str})
    selected = (frame.strategy == "PPO-simple") & (frame.seed == "42")
    frame.loc[selected, "net_return"] += 0.02
    frame.to_csv(path, index=False)
    result = rc.run_comparison(*comparison_setup, tmp_path / "mismatch")
    assert result["status"] == "provisional_reference_mismatch"
    assert result["reference_reconciliation"]["mismatched_runs"] == [
        "ppo_test_w1_simple_s42"
    ]
    samples = pd.read_parquet(tmp_path / "mismatch/stat_samples.parquet")
    simple = samples.loc[
        (samples.strategy == "PPO-simple") & (samples.seed == 42), "return"
    ].to_numpy()
    sharpe = samples.loc[
        (samples.strategy == "PPO-sharpe") & (samples.seed == 42), "return"
    ].to_numpy()
    np.testing.assert_array_equal(simple, sharpe)
