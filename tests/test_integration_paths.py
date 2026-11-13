import json
import hashlib

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from evaluation_xai.api import create_app
from evaluation_xai.artifacts import write_json
from evaluation_xai.backtest import run_backtest
from evaluation_xai.baselines import Allocation, EqualWeight
from evaluation_xai.contracts import RiskTag
from evaluation_xai.data_loader import load_processed_window
from evaluation_xai.mission import analyze_monthly_samples
from evaluation_xai.real_xai import compare_daily_replay, explain_checkpoint
from evaluation_xai.statistics import mission_anova
from evaluation_xai.validation import DataContractError


@pytest.fixture
def processed(tmp_path):
    dates = pd.bdate_range("2017-12-29", "2020-12-31")
    assets = [f"A{i}" for i in range(12)]
    rng = np.random.default_rng(9)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(dates), 12)), axis=0)),
        index=dates,
        columns=assets,
    )
    returns = np.log(prices / prices.shift()).iloc[1:]
    returns.columns = [f"{a}_return" for a in assets]
    features = pd.DataFrame(index=returns.index)
    for a in assets:
        features[f"{a}_return"] = returns[f"{a}_return"]
        features[f"{a}_rsi"] = 0.1
        features[f"{a}_macd"] = 0.2
    columns = list(features.columns)
    flags = [f"{a}_is_market_open" for a in assets]
    for flag in flags:
        prices[flag] = 1
        features[flag] = 1
    metadata = {
        "window": 1,
        "path": "features_window1.parquet",
        "train_start": "2018-01-01",
        "train_end": "2020-01-01",
        "test_end": "2021-01-01",
        "columns": columns,
        "passthrough_columns": flags,
        "mean": dict.fromkeys(columns, 0.0),
        "std": dict.fromkeys(columns, 1.0),
        "lower": dict.fromkeys(columns, -5.0),
        "upper": dict.fromkeys(columns, 5.0),
    }
    prices.to_parquet(tmp_path / "prices.parquet")
    returns.to_parquet(tmp_path / "returns.parquet")
    features.to_parquet(tmp_path / "features_window1.parquet")
    write_json(tmp_path / "walk_forward_manifest.json", {"windows": [metadata]})
    return tmp_path


def test_processed_loader_preserves_schema_and_split(processed):
    data = load_processed_window(processed)
    assert data.features.shape[1] == 48
    assert len(data.asset_order) == 12
    assert data.features.loc[data.train_mask].index.max() < pd.Timestamp("2020-01-01")
    assert data.features.loc[~data.train_mask].index.min() >= pd.Timestamp("2020-01-01")


def test_missing_training_input_never_creates_real_explanation(tmp_path):
    model = tmp_path / "ppo.zip"
    write_json(
        model.with_suffix(".json"),
        {"is_dummy": False, "stage": "test", "risk_columns": [], "window": 1},
    )
    output = tmp_path / "explanation"
    # Stop before deserializing a model or executing external environment code.
    with pytest.raises(FileNotFoundError, match="walk_forward_manifest"):
        explain_checkpoint(
            model,
            tmp_path / "missing-input",
            tmp_path / "missing-code",
            output,
            target_asset="SPY",
        )
    assert not output.exists()


def test_replay_snapshot_explicitly_allows_missing_prices(processed):
    (processed / "prices.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="prices.parquet"):
        load_processed_window(processed)
    data = load_processed_window(processed, require_prices=False)
    assert data.prices is None
    assert data.verification["price_return_consistency"] == "not_checked_prices_absent"
    assert data.verification["return_feature_consistency"] == "verified"


def test_snapshot_checksum_rejects_changed_bytes(processed):
    files = [
        "features_window1.parquet",
        "returns.parquet",
        "walk_forward_manifest.json",
    ]
    hashes = {
        n: hashlib.sha256((processed / n).read_bytes()).hexdigest()[:16] for n in files
    }
    write_json(processed / "SHA256.json", hashes)
    load_processed_window(processed, require_prices=False)
    path = processed / "walk_forward_manifest.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(DataContractError, match="checksum"):
        load_processed_window(processed, require_prices=False)


def test_missing_prices_still_rejects_wrong_return_features(processed):
    (processed / "prices.parquet").unlink()
    path = processed / "features_window1.parquet"
    data = pd.read_parquet(path)
    data.iloc[0, 0] += 0.1
    data.to_parquet(path)
    with pytest.raises(DataContractError, match="features disagree"):
        load_processed_window(processed, require_prices=False)


@pytest.mark.parametrize("mismatch", [None, "value", "date", "run", "nan"])
def test_replay_reconciliation_rejects_mismatched_run(tmp_path, mismatch):
    replayed = pd.DataFrame(
        {
            "net_return": [0.01, -0.02],
            "portfolio_value": [101, 98.98],
            "drawdown": [0, 0.02],
            "turnover": [1, 0.1],
            "max_target_weight": [0.3, 0.3],
            "benchmark_return": [0.01, 0.02],
        },
        index=pd.DatetimeIndex(["2022-01-04", "2022-01-05"], name="date"),
    )
    reference = replayed.assign(window=1, strategy="PPO-simple", seed=42).reset_index()
    if mismatch == "value":
        reference.loc[0, "net_return"] = 0.1
    elif mismatch == "date":
        reference = reference.iloc[1:]
    elif mismatch == "run":
        reference["seed"] = 43
    elif mismatch == "nan":
        reference.loc[0, "net_return"] = np.nan
    path = tmp_path / "reference.csv"
    reference.to_csv(path, index=False)
    meta = {"window": 1, "variant": "simple", "seed": 42}
    if mismatch:
        with pytest.raises(DataContractError):
            compare_daily_replay(replayed, path, meta)
    else:
        assert compare_daily_replay(replayed, path, meta)["status"] == "matched"


@pytest.mark.parametrize(
    "error", ["normalized", "order", "path", "flag", "dummy", "missing_date"]
)
def test_bad_processed_inputs(processed, error):
    path = processed / "walk_forward_manifest.json"
    manifest = json.loads(path.read_text())
    if error == "normalized":
        r = pd.read_parquet(processed / "returns.parquet")
        ((r - r.mean()) / r.std()).to_parquet(processed / "returns.parquet")
    elif error == "order":
        manifest["windows"][0]["columns"].reverse()
    elif error == "path":
        manifest["windows"][0]["path"] = "../escape.parquet"
    elif error == "flag":
        f = pd.read_parquet(processed / "features_window1.parquet")
        f.iloc[0, -1] = 2
        f.to_parquet(processed / "features_window1.parquet")
    elif error == "dummy":
        manifest["is_dummy"] = True
    else:
        f = pd.read_parquet(processed / "features_window1.parquet")
        f.iloc[1:].to_parquet(processed / "features_window1.parquet")
    write_json(path, manifest)
    with pytest.raises(DataContractError):
        load_processed_window(processed)


def test_one_sparse_test_preserves_other_results():
    rng = np.random.default_rng(21)
    data = pd.DataFrame(
        [
            {"strategy": s, "regime": r, "value": rng.normal()}
            for s in ["a", "b", "c"]
            for r in ["up", "down", "sideways"]
            for _ in range(12)
        ]
    )
    result = mission_anova(data, data, data.iloc[:2])
    assert result["status"] == "incomplete"
    assert result["tests"]["reward_variants"]["status"] == "ok"
    assert result["tests"]["strategies"]["status"] == "ok"
    assert result["tests"]["strategy_by_regime"]["status"] == "incomplete"
    assert result["holm_status"] == "incomplete_family"


def test_three_mission_tests_from_samples():
    rng = np.random.default_rng(20)
    rows = []
    months = pd.period_range("2022-01", periods=48, freq="M")
    for i, month in enumerate(months):
        for name in ["PPO-simple", "PPO-sharpe", "PPO-mdd_l1.0", "MVO", "EqualWeight"]:
            for seed in ([42, 43, 44] if name.startswith("PPO") else [None]):
                rows.append(
                    {
                        "month": str(month),
                        "strategy": name,
                        "seed": seed,
                        "return": rng.normal(0.01, 0.04),
                        "regime": ["up", "down", "sideways"][i % 3],
                    }
                )
    result = analyze_monthly_samples(pd.DataFrame(rows))
    assert result["status"] == "ok"
    assert len(result["tests"]) == 3
    assert result["holm_family_size"] == 5


def test_plot_http_and_path_escape(tmp_path):
    plots = tmp_path / "plots"
    plots.mkdir()
    (plots / "summary.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (plots / "force.html").write_text("<html>SHAP</html>")
    data = {
        "data_mode": "synthetic",
        "plot_artifacts": {
            "r/d/A": {"summary": "plots/summary.png", "force": "plots/force.html"}
        },
    }
    write_json(tmp_path / "evaluation.json", data)
    client = TestClient(create_app(tmp_path))
    route = "/explanations/r/d/A/plots/"
    assert client.get(route + "summary").status_code == 409
    response = client.get(route + "summary?allow_synthetic=true")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    response = client.get(route + "force?allow_synthetic=true")
    assert "sandbox" in response.headers["content-security-policy"]
    data["plot_artifacts"]["r/d/A"]["force"] = "../outside.html"
    write_json(tmp_path / "evaluation.json", data)
    assert client.get(route + "force?allow_synthetic=true").status_code == 503


def test_decisions_have_explicit_timestamp_status(prices):
    result = run_backtest(
        prices, EqualWeight(), test_start="2024-01-01", test_end="2024-01-10"
    )
    for row in result.decisions.itertuples():
        assert pd.Timestamp(row.decision_at).tzinfo is not None
        assert pd.Timestamp(row.execution_at) > pd.Timestamp(row.decision_at)
        assert row.risk_tag_ids == []
        assert row.time_status == "assumed_daily_order_not_exchange_clock"


def test_risk_attribution_requires_valid_tag_and_clock(prices):
    class Tagged(EqualWeight):
        def target(self, history, current_weights):
            return Allocation(np.full(3, 1 / 3), risk_tag_ids=("r1",))

    index = prices.loc["2024-01-01":"2024-01-10"].index
    schedule = pd.DataFrame(
        {
            "decision_at": [
                prices.index[prices.index.get_loc(day) - 1].tz_localize("UTC")
                + pd.Timedelta(hours=22)
                for day in index
            ],
            "execution_at": index.tz_localize("UTC") + pd.Timedelta(hours=21),
        },
        index=index,
    )
    tag = RiskTag(
        tag_id="r1",
        asset_id=prices.columns[0],
        risk_type="test",
        severity=0.2,
        confidence=0.8,
        published_at="2023-12-20T00:00:00Z",
        generated_at="2023-12-20T01:00:00Z",
        available_at="2023-12-20T01:00:00Z",
        expires_at="2024-02-01T00:00:00Z",
        source_url="https://example.org/source",
        tagger_version="test",
    )
    kwargs = dict(test_start=index[0], test_end=index[-1], risk_tags=[tag])
    with pytest.raises(DataContractError, match="explicit timestamps"):
        run_backtest(prices, Tagged(), **kwargs)
    result = run_backtest(prices, Tagged(), execution_schedule=schedule, **kwargs)
    assert result.decisions.iloc[0].risk_tag_ids == ["r1"]
    future_tag = tag.model_copy(
        update={"available_at": pd.Timestamp("2024-01-20T00:00:00Z")}
    )
    with pytest.raises(DataContractError, match="future or expired"):
        run_backtest(
            prices,
            Tagged(),
            execution_schedule=schedule,
            **{**kwargs, "risk_tags": [future_tag]},
        )
