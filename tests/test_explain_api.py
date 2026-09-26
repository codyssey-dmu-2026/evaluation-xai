import numpy as np
import pytest
from fastapi.testclient import TestClient

from evaluation_xai.api import create_app
from evaluation_xai.artifacts import write_json
from evaluation_xai.explain import explain_weights, render_explanations
from evaluation_xai.validation import DataContractError


@pytest.fixture
def explanation():
    def policy(batch):
        weight = 0.5 + 0.1 * batch[:, 0] + 0.05 * batch[:, 1]
        return np.column_stack([weight, 1 - weight])

    return explain_weights(
        policy,
        [1.0, 1.0],
        np.zeros((4, 2)),
        {"return": [0], "risk": [1]},
        asset_names=["A", "B"],
        permutations=2,
    )


def test_shap_known_linear_policy(explanation):
    assert explanation["base_values"] == pytest.approx([0.5, 0.5])
    assert np.array(explanation["shap_values"])[:, 0] == pytest.approx([0.1, 0.05])
    assert explanation["additivity_error"] < 1e-8


def test_shap_plots_created(explanation, tmp_path):
    render_explanations([explanation], tmp_path, "A")
    assert (tmp_path / "shap_summary.png").stat().st_size > 1000
    assert (tmp_path / "shap_force.html").stat().st_size > 1000


def test_shap_groups_preserve_joint_portfolio():
    seen = []

    def policy(batch):
        seen.extend(batch[:, :2].sum(axis=1))
        return batch[:, :2]

    explain_weights(
        policy,
        [0.3, 0.7, 0.5],
        [[0.5, 0.5, 0.0], [0.4, 0.6, 1.0]],
        {"all_weights": [0, 1], "risk": [2]},
        asset_names=["A", "B"],
        permutations=2,
    )
    assert np.allclose(seen, 1)


def test_shap_invalid_partition():
    with pytest.raises(DataContractError, match="partition"):
        explain_weights(
            lambda x: x,
            [0.5, 0.5],
            [[0.5, 0.5]],
            {"a": [0], "b": [0]},
            asset_names=["A", "B"],
        )


def test_health_without_model_or_artifacts(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/health").json()["real_model_loaded"] is False
    assert client.get("/backtest").status_code == 503


def test_api_explanation_and_incomplete(tmp_path, explanation):
    write_json(
        tmp_path / "evaluation.json",
        {
            "data_mode": "synthetic",
            "runs": {"test": {"decision_ids": ["d1", "d2"]}},
            "explanations": {"test/d1": explanation},
        },
    )
    client = TestClient(create_app(tmp_path))
    body = {"run_id": "test", "decision_id": "d1", "target_asset": "A"}
    result = client.post("/explain", json=body)
    assert result.status_code == 200
    assert result.json()["predicted_weight"] == pytest.approx(0.65)
    assert (
        client.post("/explain", json={**body, "decision_id": "d2"}).status_code == 409
    )
    assert (
        client.post("/explain", json={**body, "run_id": "missing"}).status_code == 404
    )
    assert (
        client.post("/explain", json={**body, "run_id": "../private"}).status_code
        == 422
    )
    assert client.get("/backtest?run_id=missing").status_code == 404


def test_nan_json_not_published(tmp_path):
    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"value": float("nan")})
    assert not (tmp_path / "bad.json").exists()
