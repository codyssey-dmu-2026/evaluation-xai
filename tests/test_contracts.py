from datetime import datetime, timezone

import numpy as np
import pytest
from pydantic import ValidationError

from evaluation_xai.adapters import sb3_weight_predictor
from evaluation_xai.artifacts import validate_market_input
from evaluation_xai.contracts import MarketDataManifest, RiskTag
from evaluation_xai.demo import run_demo
from evaluation_xai.validation import DataContractError


def tag_values():
    return dict(
        tag_id="r1",
        asset_id="SPY",
        risk_type="macro",
        severity=0.8,
        confidence=0.9,
        published_at="2024-01-01T09:00:00Z",
        available_at="2024-01-01T10:00:00Z",
        generated_at="2024-01-01T09:30:00Z",
        expires_at="2024-01-02T10:00:00Z",
        source_url="https://example.com/source",
        tagger_version="v1",
    )


def test_risk_tag_availability():
    tag = RiskTag(**tag_values())
    assert not tag.usable_at(datetime(2024, 1, 1, 9, tzinfo=timezone.utc))
    assert tag.usable_at(datetime(2024, 1, 1, 10, tzinfo=timezone.utc))
    assert not tag.usable_at(datetime(2024, 1, 2, 10, tzinfo=timezone.utc))


def test_risk_tag_no_future_generation_in_live_mode():
    with pytest.raises(ValidationError):
        RiskTag(**{**tag_values(), "generated_at": "2025-01-01T00:00:00Z"})


def test_real_manifest_rejects_five_assets():
    with pytest.raises(ValidationError):
        MarketDataManifest(
            dataset_id="x",
            data_mode="real",
            asset_order=list("ABCDE"),
            source="test",
            price_type="adjusted_close",
            normalized=False,
            base_currency="USD",
            converted_to_base_currency=True,
            calendar_policy="NYSE",
            availability_policy="prior close",
            missing_value_policy="ffill then drop",
        )


def test_market_manifest_checks_column_order(prices):
    manifest = MarketDataManifest(
        dataset_id="x",
        data_mode="synthetic",
        asset_order=list(reversed(prices.columns)),
        source="test",
        price_type="adjusted_close",
        normalized=False,
        base_currency="USD",
        converted_to_base_currency=True,
        calendar_policy="business days",
        availability_policy="prior close",
        missing_value_policy="none",
    )
    with pytest.raises(DataContractError, match="asset order"):
        validate_market_input(prices, manifest)


def test_sb3_adapter_requires_explicit_conversion():
    class Model:
        def predict(self, observation, deterministic):
            assert deterministic is True
            assert observation.shape[1:] == (2, 2)
            return np.zeros((len(observation), 3)), None

    predictor = sb3_weight_predictor(
        Model(),
        observation_shape=(2, 2),
        action_to_weights=lambda actions: np.ones_like(actions) / 3,
    )
    assert predictor(np.zeros((5, 4))).shape == (5, 3)
    with pytest.raises(DataContractError):
        predictor(np.zeros((5, 3)))


def test_demo_does_not_overwrite(tmp_path):
    (tmp_path / "existing.txt").write_text("user data")
    with pytest.raises(ValueError, match="empty"):
        run_demo(tmp_path)


def test_end_to_end_demo(tmp_path):
    from evaluation_xai.artifacts import read_json

    bundle = run_demo(tmp_path, first_year=2022, last_year=2022)
    assert bundle["data_mode"] == "synthetic"
    assert len(bundle["runs"]) == 2
    assert bundle["mission_anova"]["strategies"]["status"] == "incomplete"
    assert read_json(tmp_path / "evaluation.json")["real_model_loaded"] is False
