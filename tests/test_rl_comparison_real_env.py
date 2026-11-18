"""Optional checks against the REAL rl-engine environment and processed snapshot.

The small fixture in test_rl_comparison.py checks orchestration contracts only.
These tests exercise the producer's actual PortfolioEnv (cap projection, float32
drift, cost kernel) when the team checkout is present; they skip in CI.
No torch/checkpoint is needed. Paths: CODYSSEY_RL_SOURCE, CODYSSEY_PROCESSED,
CODYSSEY_MODELS (defaults: sibling folders of this repository).
"""

import os
from pathlib import Path
import sys

import numpy as np
import pytest

from evaluation_xai import rl_comparison as rc
from evaluation_xai.baselines import EqualWeight, MinimumVariance
from evaluation_xai.real_xai import daily_replay_report

ROOT = Path(__file__).resolve().parents[2]
RL_SOURCE = Path(os.environ.get("CODYSSEY_RL_SOURCE", ROOT / "rl-engine/src"))
PROCESSED = Path(os.environ.get("CODYSSEY_PROCESSED", ROOT / "data/processed"))
MODELS = Path(os.environ.get("CODYSSEY_MODELS", ROOT / "models"))

pytest.importorskip("gymnasium")
pytestmark = pytest.mark.skipif(
    not (
        (RL_SOURCE / "environment.py").is_file()
        and (PROCESSED / "walk_forward_manifest.json").is_file()
        and (MODELS / "eval_test_daily.csv").is_file()
    ),
    reason="team rl-engine checkout / processed snapshot not available",
)


@pytest.fixture(scope="module")
def window4():
    from evaluation_xai.data_loader import load_processed_window

    return load_processed_window(PROCESSED, 4, require_prices=False)


def test_real_decoder_swap_is_private_to_its_instance():
    before = set(sys.modules)
    original = rc.load_environment(RL_SOURCE)
    direct = rc.load_environment(RL_SOURCE, direct_weights=True)
    action = np.array([10.0] + [0.0] * 11)
    capped = original.action_to_weights(action, 0.4)
    assert capped.max() == pytest.approx(0.4) and capped.sum() == pytest.approx(1)
    raw = np.full(12, 1 / 12)
    np.testing.assert_array_equal(direct.action_to_weights(raw, 0.4), raw)
    # Loading the baseline instance must not alter the PPO instance or sys.modules.
    np.testing.assert_array_equal(original.action_to_weights(action, 0.4), capped)
    assert "environment" not in set(sys.modules) - before
    assert "evaluation_rl_comparison" not in sys.modules


def test_real_daily_equal_weight_reproduces_producer_kernel(window4):
    env = rc.build_environment(
        rc.load_environment(RL_SOURCE, direct_weights=True), window4
    )
    replay = rc.replay_run(env, window4, policy=EqualWeight("daily"))
    report = daily_replay_report(
        replay.daily,
        MODELS / "eval_test_daily.csv",
        {"window": 4, "seed": "-"},
        strategy="EqualWeight",
    )
    assert report["status"] == "matched"
    # Kernel/data differences would be >>1e-15; ulp-level libm noise is allowed.
    assert report["max_absolute_errors"]["net_return"] < 1e-15
    assert report["exact_equal_rows"]["turnover"] == report["rows"]


def test_real_monthly_mvo_holds_are_cost_free_and_point_in_time(window4):
    env = rc.build_environment(
        rc.load_environment(RL_SOURCE, direct_weights=True), window4
    )
    replay = rc.replay_run(env, window4, policy=MinimumVariance())
    hold = ~replay.daily.rebalanced
    assert hold.sum() > 200
    assert (
        (replay.daily.loc[hold, ["turnover", "commission", "slippage"]] == 0)
        .all()
        .all()
    )
    assert replay.targets.loc[~hold].to_numpy().max() <= 0.4 + 1e-9
    assert len(replay.optimization) == 12
    for record in replay.optimization:
        assert record["success"] and record["constraints_valid"]
        assert record["observations"] == 252
        assert record["history_end"] == record["decision_date"]
        assert record["outcome_date"] > record["decision_date"]
    sums = replay.targets.sum(axis=1).to_numpy()
    assert np.abs(sums - 1).max() < 1e-6
