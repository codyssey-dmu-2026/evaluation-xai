import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from evaluation_xai.adapters import rl_observation_groups
from evaluation_xai.api import create_app
from evaluation_xai.metrics import compute_metrics
from evaluation_xai.team_import import (
    SEEDS,
    VARIANTS,
    imported_ledger,
    import_rl_results,
    load_daily,
    load_model_metadata,
)
from evaluation_xai.validation import DataContractError


@pytest.fixture
def team_files(tmp_path):
    """Synthetic contract fixtures; never downloaded news or bundled real models."""
    models = tmp_path / "models"
    models.mkdir()
    rows = []
    rng = np.random.default_rng(57)
    for window in range(1, 5):
        year = 2021 + window
        dates = pd.bdate_range(f"{year}-01-03", f"{year}-12-30")
        benchmark = rng.normal(0.0002, 0.008, len(dates))
        configs = [("EqualWeight", "-", None)]
        configs += [(f"PPO-{v}", str(s), v) for v in VARIANTS for s in SEEDS]
        for strategy, seed, variant in configs:
            returns = benchmark * 0.7 + rng.normal(0.0001, 0.002, len(dates))
            equity = 1_000_000 * np.cumprod(1 + returns)
            dd = 1 - equity / np.maximum.accumulate(np.r_[1_000_000, equity])[1:]
            for i, date in enumerate(dates):
                rows.append(
                    {
                        "date": date,
                        "window": window,
                        "strategy": strategy,
                        "seed": seed,
                        "net_return": returns[i],
                        "benchmark_return": benchmark[i],
                        "portfolio_value": equity[i],
                        "drawdown": dd[i],
                        "turnover": 0.1,
                        "max_target_weight": 0.2,
                    }
                )
            if variant:
                meta = {
                    "stage": "test",
                    "window": window,
                    "seed": int(seed),
                    "variant": variant,
                    "is_dummy": False,
                    "tickers": [f"TEST{i}" for i in range(12)],
                    "risk_columns": [],
                    "timesteps": 100_000,
                    "lookback": 20,
                    "obs_dim": 288,
                    "max_weight": 0.4,
                    "train_start": "2018-01-01",
                    "train_end": f"{year}-01-01",
                    "test_end": f"{year + 1}-01-01",
                }
                (models / f"ppo_test_w{window}_{variant}_s{seed}.json").write_text(
                    json.dumps(meta)
                )
    pd.DataFrame(rows).to_csv(models / "eval_test_daily.csv", index=False)
    return models


def test_import_bundle_and_api(team_files, tmp_path):
    output = tmp_path / "result"
    result = import_rl_results(team_files, output)
    assert len(result["runs"]) == 64
    assert result["data_mode"] == "imported_unverified"
    assert result["acceptance"]["status"] == "incomplete"
    assert not result["real_model_loaded"]
    assert not result["explanations"]
    assert result["mission_anova"]["reward_variants"]["status"] == "complete"
    assert result["mission_anova"]["strategies"]["status"] == "incomplete"
    samples = pd.read_parquet(output / "stat_samples.parquet")
    assert len(samples[samples.strategy == "EqualWeight"]) == 48
    assert len(samples[samples.strategy == "PPO-simple"]) == 48 * 3
    run_id = "PPO-simple_w1_s42"
    assert result["runs"][run_id]["metrics"]["sharpe"]["value"] is None
    ledger = pd.read_parquet(output / run_id / "ledger.parquet")
    assert not ledger.guard_active.any()
    assert not (output / run_id / "weights.parquet").exists()
    client = TestClient(create_app(output))
    assert client.get("/backtest").json()["data_mode"] == "imported_unverified"
    assert client.get("/backtest", params={"run_id": run_id}).status_code == 200
    assert (
        client.post(
            "/explain",
            json={
                "run_id": run_id,
                "decision_id": "not-exported",
                "target_asset": "TEST0",
            },
        ).status_code
        == 404
    )
    with pytest.raises(DataContractError, match="empty"):
        import_rl_results(team_files, output)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_seed",
        "missing_day",
        "duplicate",
        "benchmark",
        "nonfinite",
        "cap",
        "month",
        "wrong_year",
        "negative_return",
    ],
)
def test_bad_daily_input_rejected(team_files, corruption):
    path = team_files / "eval_test_daily.csv"
    frame = pd.read_csv(path, dtype={"seed": str})
    if corruption == "missing_seed":
        frame = frame[~((frame.strategy == "PPO-simple") & (frame.seed == "43"))]
    elif corruption == "missing_day":
        frame = frame.drop(frame[frame.strategy == "PPO-simple"].index[10])
    elif corruption == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif corruption == "benchmark":
        frame.loc[
            frame[frame.strategy == "PPO-simple"].index[0], "benchmark_return"
        ] = 0.3
    elif corruption == "nonfinite":
        frame.loc[0, "net_return"] = np.inf
    elif corruption == "cap":
        frame.loc[0, "max_target_weight"] = 0.8
    elif corruption == "month":
        frame = frame[~frame.date.str.startswith("2022-02")]
    elif corruption == "wrong_year":
        frame.loc[0, "date"] = "2020-01-01"
    else:
        frame.loc[0, "net_return"] = -1
    frame.to_csv(path, index=False)
    with pytest.raises(DataContractError):
        load_daily(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("is_dummy", True),
        ("obs_dim", 300),
        ("stage", "val"),
        ("seed", 11),
        ("train_start", "2019-01-01"),
        ("timesteps", 100),
    ],
)
def test_bad_metadata_rejected(team_files, field, value):
    path = team_files / "ppo_test_w1_simple_s42.json"
    meta = json.loads(path.read_text())
    meta[field] = value
    path.write_text(json.dumps(meta))
    frame = load_daily(team_files / "eval_test_daily.csv")
    with pytest.raises(DataContractError):
        load_model_metadata(team_files, frame)


@pytest.mark.parametrize("field", ["portfolio_value", "drawdown"])
def test_bad_account_rejected(team_files, field):
    frame = load_daily(team_files / "eval_test_daily.csv")
    group = frame[(frame.window == 1) & (frame.strategy == "EqualWeight")].copy()
    group.loc[group.index[2], field] += 100
    with pytest.raises(DataContractError):
        imported_ledger(group)


def test_missing_rf_does_not_assume_zero(ledger_factory):
    ledger = ledger_factory([0.01, -0.02, 0.005])
    result = compute_metrics(
        ledger,
        pd.Series([0.02, -0.01, 0.003], index=ledger.index),
        None,
        period_start=ledger.index[0] - pd.Timedelta(days=1),
    )
    for name in ("sharpe", "sortino", "alpha", "beta"):
        assert result["metrics"][name]["value"] is None
        assert result["metrics"][name]["reason"] == "risk_free_not_provided"
    assert result["metrics"]["cumulative_return"]["value"] is not None
    assert result["metrics"]["information_ratio"]["value"] is not None


@pytest.mark.parametrize("with_risk", [False, True])
def test_observation_layout_matches_rl_flattening(with_risk):
    assets = [f"A{i}" for i in range(12)]
    columns = [f"{a}_{field}" for a in assets for field in ("return", "rsi", "macd")]
    columns += [f"{a}_is_market_open" for a in assets]
    if with_risk:
        columns += [f"{a}_risk" for a in assets]
    groups = rl_observation_groups(assets, columns)
    assert groups["A0:returns"] == list(range(0, 240, 12))
    assert groups["A1:returns"] == list(range(1, 240, 12))
    assert groups["static:A0_rsi"] == [240]
    dim = 300 if with_risk else 288
    assert groups["portfolio_weights"] == list(range(dim - 12, dim))
    assert sorted(i for group in groups.values() for i in group) == list(range(dim))


def test_bad_observation_schema():
    with pytest.raises(DataContractError):
        rl_observation_groups(["A", "B"], ["A_return"])
