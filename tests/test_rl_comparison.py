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
                for v in rc.VARIANTS
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
        lambda path, **kwargs: pd.read_csv(
            path,
            dtype={"seed": str},
            parse_dates=["date"],
            float_precision="round_trip",
        ),
    )
    for variant in rc.VARIANTS:
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
    assert stats["reward_variants"]["status"] == "complete"
    assert stats["strategies"]["status"] == "complete"
    assert stats["strategy_by_regime"]["status"] == "data_invalid"
    assert stats["overall_status"] == "incomplete"
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


# --- 2026-11-15 review regressions: resume, provenance, flags, SHAP linking ---


def _failing_after(monkeypatch, calls_before_failure, error=DataContractError):
    stub, calls = rc.load_checkpoint, []

    def loader(path, *args):
        calls.append(path.name)
        if len(calls) > calls_before_failure:
            raise error("checkpoint broken")
        return stub(path, *args)

    monkeypatch.setattr(rc, "load_checkpoint", loader)
    return stub, calls


def test_resume_reuses_only_completed_runs(comparison_setup, tmp_path, monkeypatch):
    output = tmp_path / "resumable"
    stub, _ = _failing_after(monkeypatch, 1)
    with pytest.raises(DataContractError, match="broken"):
        rc.run_comparison(*comparison_setup, output)
    first = json.loads((output / "failure.json").read_text())
    assert first["completed_runs"][-1] == "ppo_test_w1_simple_s42"
    assert not (output / "evaluation.json").exists()
    assert json.loads((output / "run_manifest.json").read_text())["status"] == "failed"
    calls = []
    monkeypatch.setattr(
        rc, "load_checkpoint", lambda path, *a: calls.append(path.name) or stub(path)
    )
    result = rc.run_comparison(*comparison_setup, output, resume=True)
    assert result["coverage"]["reused_completed_runs"] == first["completed_runs"]
    assert len(calls) == 8  # only the 8 unfinished PPO checkpoints were loaded
    assert len(result["attempts"]) == 2
    assert not (output / "failure.json").exists()
    assert (output / "attempts/failure-1.json").exists()
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "published"
    fresh = rc.run_comparison(*comparison_setup, tmp_path / "fresh")
    for name in ("reward_variants", "strategies"):
        assert (
            result["mission_anova"][name]["effects"]
            == fresh["mission_anova"][name]["effects"]
        )
    with pytest.raises(DataContractError, match="already published"):
        rc.run_comparison(*comparison_setup, output, resume=True)


def test_resume_refuses_changed_inputs_or_tampered_runs(
    comparison_setup, tmp_path, monkeypatch
):
    models, *_ = comparison_setup
    output = tmp_path / "guarded"
    stub, _ = _failing_after(monkeypatch, 1)
    with pytest.raises(DataContractError):
        rc.run_comparison(*comparison_setup, output)
    monkeypatch.setattr(rc, "load_checkpoint", lambda path, *a: stub(path))
    ledger = output / "MVO_monthly_w1/ledger.parquet"
    original = ledger.read_bytes()
    ledger.write_bytes(original + b"tampered")
    with pytest.raises(DataContractError, match="changed: MVO_monthly_w1"):
        rc.run_comparison(*comparison_setup, output, resume=True)
    ledger.write_bytes(original)
    reference = models / "eval_test_daily.csv"
    reference.write_text(reference.read_text() + "\n")
    with pytest.raises(DataContractError, match="reference_sha256"):
        rc.run_comparison(*comparison_setup, output, resume=True)
    with pytest.raises(DataContractError, match="run_manifest"):
        rc.run_comparison(*comparison_setup, tmp_path / "never-started", resume=True)


def test_resume_preserves_interrupted_partial_folder(
    comparison_setup, tmp_path, monkeypatch
):
    output = tmp_path / "interrupted"
    stub, _ = _failing_after(monkeypatch, 0, KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        rc.run_comparison(*comparison_setup, output)
    failure = json.loads((output / "failure.json").read_text())
    assert failure["error_type"] == "KeyboardInterrupt"
    assert not (output / "evaluation.json").exists()
    partial = output / ".partial-ppo_test_w1_simple_s42"
    partial.mkdir()
    (partial / "ledger.parquet").write_bytes(b"half written")
    monkeypatch.setattr(rc, "load_checkpoint", lambda path, *a: stub(path))
    result = rc.run_comparison(*comparison_setup, output, resume=True)
    kept = output / ".abandoned/ppo_test_w1_simple_s42-attempt1/ledger.parquet"
    assert kept.read_bytes() == b"half written"
    assert "ppo_test_w1_simple_s42" in result["runs"]
    assert not list(output.glob(".partial-*"))


def test_completed_run_files_are_hash_recorded(comparison_setup, tmp_path):
    output = tmp_path / "hashed"
    result = rc.run_comparison(*comparison_setup, output)
    run = json.loads((output / "ppo_test_w1_sharpe_s43/run.json").read_text())
    assert set(run["artifact_sha256"]) == set(rc.RUN_FILES)
    assert (
        run["artifact_sha256"]
        == result["runs"]["ppo_test_w1_sharpe_s43"]["artifact_sha256"]
    )
    runtime = result["source"]["runtime"]
    assert runtime["python"] and runtime["numpy"] == np.__version__
    ppo = json.loads((output / "ppo_test_w1_sharpe_s43/decisions.json").read_text())
    mvo = json.loads((output / "MVO_monthly_w1/decisions.json").read_text())
    assert all(len(d["observation_sha256"]) == 64 for d in ppo)
    assert all(d["observation_sha256"] is None for d in mvo)
    samples = pd.read_parquet(output / "stat_samples.parquet")
    first = samples.iloc[0]
    assert first.first_decision_id == "d20220103"
    assert first.first_outcome_date == "2022-01-04"


def test_supplementary_lambdas_never_enter_anova(comparison_setup, tmp_path):
    base = rc.run_comparison(*comparison_setup, tmp_path / "base")
    full = rc.run_comparison(*comparison_setup, tmp_path / "full", variants=rc.VARIANTS)
    assert full["coverage"]["ppo_replayed"] == 15
    assert full["coverage"]["not_replayed_variants"] == []
    assert base["coverage"]["not_replayed_variants"] == ["mdd_l0.5", "mdd_l5.0"]
    assert not full["runs"]["ppo_test_w1_mdd_l0p5_s42"]["include_in_anova"]
    samples = pd.read_parquet(tmp_path / "full/stat_samples.parquet")
    extra = pd.read_parquet(tmp_path / "full/supplementary_samples.parquet")
    assert len(samples) == 132
    assert set(extra.strategy) == {"PPO-mdd_l0.5", "PPO-mdd_l5.0"}
    assert len(extra) == 72
    for name in rc.TEST_STRATEGIES:
        assert base["mission_anova"][name].get("effects") == full["mission_anova"][
            name
        ].get("effects")
    with pytest.raises(DataContractError, match="predeclared"):
        rc.run_comparison(
            *comparison_setup, tmp_path / "bad", variants=("simple", "sharpe")
        )


def test_missing_benchmark_column_only_invalidates_regime_test(
    comparison_setup, tmp_path
):
    _, processed, _ = comparison_setup
    frame = pd.read_parquet(processed / "returns.parquet")
    frame.rename(columns={"SPY_return": "OTHER_return"}).to_parquet(
        processed / "returns.parquet"
    )
    result = rc.run_comparison(*comparison_setup, tmp_path / "nospy")
    tests = result["mission_anova"]
    assert tests["reward_variants"]["status"] == "complete"
    assert tests["strategies"]["status"] == "complete"
    assert tests["strategy_by_regime"]["status"] == "data_invalid"
    assert "KeyError" in tests["strategy_by_regime"]["reason"]
    assert result["mission_anova"]["holm_status"] == "incomplete_family"


def test_mismatch_is_flagged_per_test_and_diagnosed(comparison_setup, tmp_path):
    models, *_ = comparison_setup
    path = models / "eval_test_daily.csv"
    frame = pd.read_csv(path, dtype={"seed": str}, float_precision="round_trip")
    selected = (frame.strategy == "PPO-simple") & (frame.seed == "42")
    frame.loc[selected, "drawdown"] += 2e-8
    frame.to_csv(path, index=False)
    result = rc.run_comparison(*comparison_setup, tmp_path / "flags")
    tests = result["mission_anova"]
    flag = tests["reward_variants"]["input_reconciliation"]
    assert flag["status"] == "includes_reference_mismatch"
    assert flag["mismatched_runs"] == ["ppo_test_w1_simple_s42"]
    for name in ("strategies", "strategy_by_regime"):
        assert tests[name]["input_reconciliation"]["status"] == "no_reference_mismatch"
        assert tests[name]["input_reconciliation"][
            "new_baselines_without_reference"
        ] == ["EqualWeight_monthly_w1", "MVO_monthly_w1"]
    diagnosis = result["reference_reconciliation"]["diagnosis"]
    assert diagnosis["classification"] == "unresolved_numeric_difference"
    assert diagnosis["tolerance_changed"] is False
    assert diagnosis["kernel_controls"] == {
        "runs": 1,
        "matched": 1,
        "net_return_max_abs_error": [0.0, 0.0],
    }
    run = diagnosis["mismatched_runs"]["ppo_test_w1_simple_s42"]
    assert run["max_tolerance_ratio"]["drawdown"] > 1
    assert run["monthly_return_max_abs_error"] == 0.0
    mission = json.loads((tmp_path / "flags/mission-anova.json").read_text())
    assert mission["reward_variants"]["input_reconciliation"] == flag


def test_report_separates_exact_and_tolerated_rows(source, data, tmp_path):
    replay = rc.replay_run(
        rc.build_environment(rc.load_environment(source), data), data, model=ZeroModel()
    )
    reference = replay.daily.assign(window=1, strategy="PPO-simple", seed=42)
    path = tmp_path / "daily.csv"
    reference.to_csv(path)
    meta = {"window": 1, "variant": "simple", "seed": 42}
    exact = daily_replay_report(replay.daily, path, meta)
    assert exact["bitwise_identical"]
    reference.iloc[0, reference.columns.get_loc("turnover")] += 1e-9
    reference.to_csv(path)
    tolerated = daily_replay_report(replay.daily, path, meta)
    assert tolerated["status"] == "matched"
    assert not tolerated["bitwise_identical"]
    assert tolerated["exact_equal_rows"]["turnover"] == len(replay.daily) - 1
    assert 0 < tolerated["max_tolerance_ratio"]["turnover"] < 1


def test_failed_publication_writes_no_statistics(
    comparison_setup, tmp_path, monkeypatch
):
    def broken(*args, **kwargs):
        raise ValueError("Out of range float values are not JSON compliant")

    monkeypatch.setattr(rc, "json_bytes", broken)
    output = tmp_path / "unpublished"
    with pytest.raises(ValueError):
        rc.run_comparison(*comparison_setup, output)
    assert json.loads((output / "failure.json").read_text())["active_run"] == (
        "bundle_serialization"
    )
    for name in ("evaluation.json", "mission-anova.json", "stat_samples.parquet"):
        assert not (output / name).exists()


def _explanation_bundle(folder, models, source, observation, **override):
    folder.mkdir()
    run_id, decision_id = "ppo_test_w1_mdd_l1p0_s42", "d20220103"
    model = models / "ppo_test_w1_mdd_l1.0_s42.zip"
    meta = {
        "model_sha256": rc.file_hash(model),
        "model_metadata_sha256": rc.file_hash(model.with_suffix(".json")),
        "environment_sha256": rc.file_hash(source / "environment.py"),
        "features_sha256": "f" * 64,
        "returns_sha256": "r" * 64,
        "manifest_sha256": "m" * 64,
        **override,
    }
    (folder / "plots").mkdir()
    (folder / "plots/shap_summary.png").write_bytes(b"\x89PNG fixture")
    (folder / "plots/shap_force.html").write_text("<html></html>")
    np.savez_compressed(
        folder / "explanation_inputs.npz",
        observation=np.asarray(observation, dtype=np.float32),
        background=np.zeros((2, 3), dtype=np.float32),
    )
    key = f"{run_id}/{decision_id}"
    bundle = {
        "schema_version": "0.2",
        "data_mode": "real",
        "status": "explanation_only_not_final_evaluation",
        "source": meta,
        "reconciliation": {"status": "matched"},
        "runs": {run_id: {"decision_ids": [decision_id]}},
        "explanations": {
            key: {
                "asset_names": ["A", "B", "SPY"],
                "predicted_weights": [1 / 3] * 3,
                "decision_date": "2022-01-03",
                "stage": "real_ppo",
                "method": "permutation",
                "group_names": ["g"],
                "base_values": [0.3, 0.3, 0.4],
                "shap_values": [[0.033, 0.033, -0.067]],
                "interpretation": "fixture",
            }
        },
        "plot_artifacts": {
            f"{key}/SPY": {
                "summary": "plots/shap_summary.png",
                "force": "plots/shap_force.html",
            }
        },
    }
    (folder / "evaluation.json").write_text(json.dumps(bundle))
    return folder


def test_existing_shap_links_only_on_identical_observation(
    comparison_setup, data, tmp_path
):
    from fastapi.testclient import TestClient

    from evaluation_xai.api import create_app

    models, _, source = comparison_setup
    data.verification = {
        "file_sha256": {
            "features_window1.parquet": "f" * 64,
            "returns.parquet": "r" * 64,
            "walk_forward_manifest.json": "m" * 64,
        }
    }
    good = _explanation_bundle(tmp_path / "good", models, source, np.zeros(3))
    (good / "plots/shap_force.html").unlink()  # missing derived plot is not fatal
    drifted = _explanation_bundle(tmp_path / "drifted", models, source, [0, 0, 1e-7])
    other = _explanation_bundle(
        tmp_path / "other", models, source, np.zeros(3), model_sha256="0" * 64
    )
    output = tmp_path / "linked"
    result = rc.run_comparison(
        *comparison_setup, output, explanation_dirs=[good, drifted, other]
    )
    status = {link["source"]: link for link in result["explanation_links"]}
    assert status["good"]["status"] == "linked"
    assert status["drifted"]["status"] == "rejected"
    assert not status["drifted"]["checks"]["observation_sha256"]
    assert status["other"]["status"] == "rejected"
    assert not status["other"]["checks"]["model_sha256"]
    key = "ppo_test_w1_mdd_l1p0_s42/d20220103"
    assert list(result["explanations"]) == [key]
    assert result["explanations"][key]["link"]["shap_recomputed"] is False
    run = result["runs"]["ppo_test_w1_mdd_l1p0_s42"]
    assert run["shap_status"] == "linked_verified_existing_shap"
    assert run["explained_decision_ids"] == ["d20220103"]
    assert result["runs"]["ppo_test_w1_mdd_l1p0_s43"]["shap_status"] == (
        "not_available_in_this_bundle"
    )
    client = TestClient(create_app(output))
    request = {"run_id": "ppo_test_w1_mdd_l1p0_s42", "target_asset": "SPY"}
    assert (
        client.post(
            "/explain", json={**request, "decision_id": "d20220103"}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/explain", json={**request, "decision_id": "d20220104"}
        ).status_code
        == 409
    )
    plot = client.get(f"/explanations/{key}/SPY/plots/summary")
    assert plot.status_code == 200 and plot.content == b"\x89PNG fixture"
    assert status["good"]["plots_not_copied"] == ["SPY/force"]
    assert client.get(f"/explanations/{key}/SPY/plots/force").status_code == 404


def test_bundle_uses_the_shared_mission_and_weights_contract(
    comparison_setup, tmp_path
):
    # Same shape import-rl publishes and the platform dashboard/timeseries read.
    output = tmp_path / "contract"
    result = rc.run_comparison(*comparison_setup, output)
    mission = result["mission_anova"]
    for name in ("reward_variants", "strategies"):
        assert mission[name]["status"] == "complete" and mission[name]["effects"]
    # 12 synthetic months leave sparse regime cells: reported, never "complete".
    assert mission["strategy_by_regime"]["status"] == "insufficient_data"
    assert mission["overall_status"] == "incomplete"
    assert mission["holm_status"] == "incomplete_family"
    assert mission["reward_variants"]["representative_mdd_lambda"] == 1.0
    assert "tests" not in mission
    saved = json.loads((output / "mission-anova.json").read_text())
    assert saved == mission
    for run_id in ("MVO_monthly_w1", "ppo_test_w1_simple_s42"):
        weights = pd.read_parquet(output / run_id / "weights.parquet")
        targets = pd.read_parquet(output / run_id / "target_weights.parquet")
        pd.testing.assert_frame_equal(weights, targets)
        ledger = pd.read_parquet(output / run_id / "ledger.parquet")
        assert {"equity_start", "equity_end", "drawdown", "net_return_simple"} <= set(
            ledger.columns
        )
        assert {"safeguard_triggered", "guard_active"} <= set(ledger.columns)
