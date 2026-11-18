"""Replay an explicitly supplied trusted RL checkpoint with its original inputs."""

import hashlib
import importlib.util
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from . import DISCLAIMER
from .adapters import rl_observation_groups, sb3_weight_predictor
from .artifacts import read_json, write_json
from .data_loader import load_processed_window
from .explain import explain_weights, render_explanations
from .provenance import runtime_info
from .team_import import file_hash
from .validation import DataContractError

# Strict producer-reproduction tolerance. Do not widen to make a run pass.
ATOL, RTOL = 1e-8, 1e-6


def observation_sha256(observation):
    """Exact identity of a policy input; any bit difference is a different input."""
    array = np.ascontiguousarray(np.asarray(observation, dtype=np.float32))
    return hashlib.sha256(array.tobytes()).hexdigest()


def compare_daily_replay(replayed, reference_daily, meta, *, strategy=None):
    """Fail closed on incomplete dates or a different recorded policy trajectory."""
    report = daily_replay_report(replayed, reference_daily, meta, strategy=strategy)
    if report["status"] != "matched":
        fields = ", ".join(report["mismatched_fields"])
        raise DataContractError(f"replay differs from producer CSV: {fields}")
    return report


def daily_replay_report(replayed, reference_daily, meta, *, strategy=None):
    """Audit numeric disagreement without silently changing the strict tolerances.

    A mismatch report is not a successful reproduction. Structural or nonfinite
    inputs still raise; newly replayed statistics must never substitute CSV rows.
    """
    # round_trip parsing: the default fast parser can be off by one ulp, which
    # would hide whether a replay is bitwise identical to the producer's CSV.
    reference = pd.read_csv(
        reference_daily, dtype={"seed": str}, float_precision="round_trip"
    )
    reference = reference.loc[
        (reference.window == meta["window"])
        & (reference.strategy == (strategy or f"PPO-{meta['variant']}"))
        & (reference.seed == str(meta["seed"]))
    ].copy()
    reference["date"] = pd.to_datetime(reference.date)
    reference = reference.set_index("date")
    if not replayed.index.equals(reference.index):
        raise DataContractError(
            "replay/reference dates differ or selected run is absent"
        )
    errors, mismatches, exact, ratios = {}, {}, {}, {}
    for field in (
        "net_return",
        "portfolio_value",
        "drawdown",
        "turnover",
        "max_target_weight",
        "benchmark_return",
    ):
        actual = replayed[field].to_numpy(dtype=float)
        expected = reference[field].to_numpy(dtype=float)
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise DataContractError(f"nonfinite replay/reference field: {field}")
        different = ~np.isclose(actual, expected, atol=ATOL, rtol=RTOL)
        if different.any():
            mismatches[field] = {
                "rows": int(different.sum()),
                "dates": [str(d.date()) for d in replayed.index[different]],
            }
        absolute = np.abs(actual - expected)
        errors[field] = float(np.max(absolute))
        exact[field] = int((actual == expected).sum())
        # >1 means outside the unchanged tolerance; diagnostic only, never a pass.
        ratios[field] = float(np.max(absolute / (ATOL + RTOL * np.abs(expected))))
    return {
        "status": "matched" if not mismatches else "mismatch",
        "mismatched_fields": mismatches,
        "rows": len(replayed),
        "atol": ATOL,
        "rtol": RTOL,
        "max_absolute_errors": errors,
        "exact_equal_rows": exact,
        "max_tolerance_ratio": ratios,
        "bitwise_identical": all(v == len(replayed) for v in exact.values()),
        "reference_sha256": file_hash(reference_daily),
        "scope": "producer_reproduction_check_not_financial_protocol_acceptance",
    }


def explain_checkpoint(
    model_path,
    processed_dir,
    rl_source_dir,
    output_directory,
    *,
    target_asset,
    decision_date=None,
    reference_daily=None,
):
    """One real decision; no fabricated inputs, fitting, or new training.

    The external environment and checkpoint must be trusted by the caller.
    Loading a PyTorch/SB3 checkpoint can execute serialized Python objects.
    """
    started = perf_counter()
    model_path = Path(model_path)
    output = Path(output_directory)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise DataContractError("output directory must be empty")
    meta = read_json(model_path.with_suffix(".json"))
    if meta.get("is_dummy") is not False or meta.get("stage") != "test":
        raise DataContractError("requires a real test-stage model declaration")
    if meta.get("risk_columns"):
        raise DataContractError(
            "risk-conditioned models require historical risk inputs; not supported yet"
        )
    data = load_processed_window(processed_dir, meta["window"], require_prices=False)
    if meta["tickers"] != data.asset_order or target_asset not in data.asset_order:
        raise DataContractError("model/target asset order mismatch")
    for field in ("train_start", "train_end", "test_end"):
        if meta[field] != data.metadata[field]:
            raise DataContractError("model and data training/test periods differ")
    source = Path(rl_source_dir) / "environment.py"
    spec = importlib.util.spec_from_file_location("evaluation_external_rl_env", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path), device="cpu")
    if meta.get("timesteps", 0) < 100_000 or model.num_timesteps < meta["timesteps"]:
        raise DataContractError(
            "checkpoint does not substantiate declared training steps"
        )
    lookback = meta["lookback"]
    train_features = data.features.loc[data.train_mask]
    train_returns = data.returns.loc[data.train_mask]
    env = module.PortfolioEnv(
        train_features,
        train_returns,
        lookback=lookback,
        mdd_limit=10.0,
        max_weight=meta["max_weight"],
    )
    if (
        model.observation_space.shape != env.observation_space.shape
        or meta["obs_dim"] != env.observation_space.shape[0]
    ):
        raise DataContractError("model observation dimension mismatch")
    if model.action_space.shape != (len(data.asset_order),):
        raise DataContractError("model action dimension mismatch")
    obs, _ = env.reset(seed=meta["seed"])
    history, history_dates = [], []
    while True:
        history.append(obs.copy())
        history_dates.append(str(env.dates[env.current_step].date()))
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    if len(history) < 32:
        raise DataContractError(
            "insufficient real training rollout for SHAP background"
        )
    selection = np.linspace(0, len(history) - 1, 32, dtype=int)
    background = np.stack(history)[selection]
    # Match the producer's 19-row prefix and its first scored next-day step.
    features = pd.concat(
        [train_features.tail(lookback - 1), data.features.loc[~data.train_mask]]
    )
    returns = pd.concat(
        [train_returns.tail(lookback - 1), data.returns.loc[~data.train_mask]]
    )
    test_env = module.PortfolioEnv(
        features,
        returns,
        lookback=lookback,
        mdd_limit=10.0,
        max_weight=meta["max_weight"],
    )
    obs, _ = test_env.reset(seed=meta["seed"])
    target = (
        pd.Timestamp(decision_date)
        if decision_date
        else test_env.dates[test_env.current_step]
    )
    if (
        target not in data.features.index[~data.train_mask]
        or target >= features.index[-1]
    ):
        raise DataContractError(
            "decision must be a test observation with a next outcome day"
        )
    records, target_observation = [], None
    while True:
        if test_env.dates[test_env.current_step] == target:
            target_observation = obs.copy()
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, info = test_env.step(action)
        records.append(
            {
                "date": info["date"],
                **{
                    k: info[k]
                    for k in ("net_return", "portfolio_value", "drawdown", "turnover")
                },
                "max_target_weight": float(info["target_weights"].max()),
                "benchmark_return": float(
                    np.expm1(returns.loc[info["date"], "SPY_return"])
                ),
            }
        )
        if done or truncated:
            break
    if target_observation is None:
        raise DataContractError("replay did not contain requested decision")
    obs = target_observation
    replayed = pd.DataFrame(records).set_index("date")
    reconciliation = (
        compare_daily_replay(replayed, reference_daily, meta)
        if reference_daily
        else {"status": "not_checked_reference_not_provided"}
    )
    predictor = sb3_weight_predictor(
        model,
        observation_shape=model.observation_space.shape,
        action_to_weights=lambda a: module.action_to_weights(a, meta["max_weight"]),
    )
    actual_action, _ = model.predict(obs, deterministic=True)
    actual_weights = module.action_to_weights(actual_action, meta["max_weight"])
    if not np.allclose(predictor(obs[None])[0], actual_weights, atol=1e-6, rtol=0):
        raise DataContractError("explanation wrapper differs from actual RL policy")
    groups = rl_observation_groups(
        data.asset_order, data.features.columns, lookback=lookback
    )
    shap_started = perf_counter()
    explanation = explain_weights(
        predictor, obs, background, groups, asset_names=data.asset_order, seed=20260926
    )
    shap_seconds = perf_counter() - shap_started
    explanation["model_kind"] = "real_ppo_checkpoint"
    explanation["background_dates"] = [history_dates[i] for i in selection]
    explanation["decision_date"] = str(target.date())
    explanation["observation_sha256"] = observation_sha256(obs)
    explanation["time_status"] = "date_only_upstream_clock_not_verified"
    run_id = model_path.stem.replace(".", "p")
    decision_id = f"d{target:%Y%m%d}"
    output.mkdir(parents=True, exist_ok=True)
    render_explanations([explanation], output / "plots", target_asset)
    replayed.to_parquet(output / "replayed_daily.parquet")
    np.savez_compressed(
        output / "explanation_inputs.npz", observation=obs, background=background
    )
    bundle = {
        "schema_version": "0.2",
        "data_mode": "real",
        "status": "explanation_only_not_final_evaluation",
        "real_model_loaded": True,
        "input_verification": data.verification,
        "reconciliation": reconciliation,
        "timing_seconds": {
            "shap_compute": shap_seconds,
            "total_offline": perf_counter() - started,
        },
        "disclaimer": DISCLAIMER,
        "limitations": [
            "Single decision replay, not all-decision explanation coverage.",
            "Raw price provenance, mixed-market FX/timing and "
            "full scaler fit remain unverified.",
            "No RAG/Safe-Guard explanation or 5-second performance claim.",
        ],
        "source": {
            "model_sha256": file_hash(model_path),
            "model_metadata_sha256": file_hash(model_path.with_suffix(".json")),
            "environment_sha256": file_hash(source),
            "features_sha256": file_hash(Path(processed_dir) / data.metadata["path"]),
            "returns_sha256": file_hash(Path(processed_dir) / "returns.parquet"),
            "manifest_sha256": file_hash(
                Path(processed_dir) / "walk_forward_manifest.json"
            ),
            "runtime": runtime_info(),
        },
        "runs": {
            run_id: {
                "strategy": "real_ppo",
                "status": "explanation_only",
                "decision_ids": [decision_id],
                "risk_tag_ids": [],
            }
        },
        "explanations": {f"{run_id}/{decision_id}": explanation},
        "plot_artifacts": {
            f"{run_id}/{decision_id}/{target_asset}": {
                "summary": "plots/shap_summary.png",
                "force": "plots/shap_force.html",
            }
        },
    }
    write_json(output / "evaluation.json", bundle)
    return bundle
