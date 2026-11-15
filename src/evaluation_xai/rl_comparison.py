"""Return-based baselines and PPO replays under the producer's execution kernel.

This is a separate, provisional profile, not evaluation v1's next-close backtest.
Only trusted local environment source and checkpoints may be passed to this module.
"""

from dataclasses import dataclass
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from . import DISCLAIMER
from .artifacts import read_json, write_json
from .baselines import EqualWeight, MinimumVariance
from .data_loader import load_processed_window
from .metrics import compute_metrics
from .mission import analyze_monthly_samples
from .real_xai import compare_daily_replay, daily_replay_report
from .statistics import (
    aggregate_seeds,
    monthly_returns,
    paired_block_bootstrap,
    regimes_from_log_returns,
)
from .team_import import SEEDS, WINDOWS, file_hash, imported_ledger, load_daily
from .team_import import load_model_metadata
from .validation import DataContractError, validate_weights

PROFILE = "rl_return_comparison_monthly_v1"
REWARD_VARIANTS = ("simple", "sharpe", "mdd_l1.0")
ENV_SETTINGS = {
    "lookback": 20,
    "fee_rate": 0.00015,
    "slippage": 0.0005,
    "mdd_limit": 10.0,
    "max_weight": 0.4,
    "initial_cash": 1_000_000.0,
    "random_start": False,
}
LIMITATIONS = [
    "Provisional reproduction of RL execution, not evaluation v1 acceptance.",
    "Raw prices, common-currency FX and exchange clocks remain unverified.",
    "SPY ETF proxy on union-calendar observations, not the S&P 500 price index.",
    "Guard disabled; no final liquidation; expanding rather than rolling training.",
    "First test observation is a decision only; scoring starts the next session.",
    "RF absent: Sharpe, Sortino, Alpha and Beta remain null.",
    "Monthly baselines and daily PPO differ in policy frequency, not cost rules.",
    "The 40% cap applies to new targets; held weights may drift above it.",
    "No new SHAP, platform integration, or all-60-checkpoint verification.",
]


def load_environment(source_dir, *, direct_weights=False):
    """Load a PRIVATE module instance; never patch a shared producer import.

    The producer step kernel is unchanged. In the baseline-only instance its
    action decoder is replaced with an identity copy, avoiding lossy softmax
    inverses and forced trades on hold days. Targets are checked by replay_run.
    Separate PPO instances keep the producer's original action_to_weights.
    """
    source = Path(source_dir) / "environment.py"
    spec = importlib.util.spec_from_file_location("evaluation_rl_comparison", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if direct_weights:
        module.action_to_weights = lambda weights, max_weight: np.asarray(
            weights, dtype=np.float64
        ).copy()
    return module


def build_environment(module, data):
    prefix = ENV_SETTINGS["lookback"] - 1
    features = pd.concat(
        [
            data.features.loc[data.train_mask].tail(prefix),
            data.features.loc[~data.train_mask],
        ]
    )
    returns = data.returns.loc[features.index]
    if len(data.features.loc[data.train_mask]) < prefix:
        raise DataContractError("insufficient observation warmup")
    return module.PortfolioEnv(features, returns, **ENV_SETTINGS)


@dataclass
class Replay:
    daily: pd.DataFrame
    targets: pd.DataFrame
    holdings: pd.DataFrame
    decisions: list[dict]
    optimization: list[dict]


def replay_run(env, data, *, model=None, policy=None, seed=42):
    """Model actions use the original module; policies use a direct-weight instance.

    Rebalance at the first available DECISION in each calendar month (and initial
    entry). A hold passes the exact float32 drifted weights back to the kernel:
    turnover and cost are exactly zero, even after drift above the target cap.
    """
    if (model is None) == (policy is None):
        raise DataContractError("provide exactly one model or baseline policy")
    obs, _ = env.reset(seed=seed)
    daily, targets, holdings, decisions, optimization = [], [], [], [], []
    previous_month = None
    while True:
        decision_date = pd.Timestamp(env.dates[env.current_step])
        month = decision_date.to_period("M")
        rebalance = (
            model is not None
            or policy.rebalance == "daily"
            or (month != previous_month)
        )
        allocation = None
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        elif rebalance:
            history = data.returns.loc[data.returns.index <= decision_date]
            if isinstance(policy, MinimumVariance):
                allocation = policy.target_returns(np.expm1(history), env.weights)
            else:
                allocation = policy.target(history, env.weights)
            action = allocation.weights
            validate_weights(action, len(data.asset_order), ENV_SETTINGS["max_weight"])
        else:
            # Do NOT cap, normalize, or round held weights: that would create trades.
            action = env.weights.copy()
            validate_weights(action, len(data.asset_order), 1.0)
        equity_start = float(env.portfolio_value)
        obs, _, done, truncated, info = env.step(action)
        if not rebalance and float(info["turnover"]) != 0.0:
            raise DataContractError("hold day incurred turnover")
        outcome = pd.Timestamp(info["date"])
        if outcome <= decision_date:
            raise DataContractError("RL outcome must follow decision")
        target = np.asarray(info["target_weights"], dtype=float)
        if policy is not None and not np.array_equal(target, np.asarray(action)):
            raise DataContractError(
                "direct-weight decoder changed the requested weights"
            )
        validate_weights(
            target,
            len(data.asset_order),
            ENV_SETTINGS["max_weight"] if rebalance else 1.0,
        )
        cost = float(info["turnover"]) * (env.fee_rate + env.slippage)
        gross = float(target @ np.expm1(data.returns.loc[outcome].to_numpy()))
        if not np.isclose(info["net_return"], gross - cost, atol=1e-12, rtol=0):
            raise DataContractError("RL cost/return kernel differs from profile")
        daily.append(
            {
                "date": outcome,
                **{
                    k: float(info[k])
                    for k in ("net_return", "portfolio_value", "drawdown", "turnover")
                },
                "max_target_weight": float(target.max()),
                "benchmark_return": float(
                    np.expm1(data.returns.loc[outcome, "SPY_return"])
                ),
                "gross_return": gross,
                "commission": equity_start * float(info["turnover"]) * env.fee_rate,
                "slippage": equity_start * float(info["turnover"]) * env.slippage,
                "rebalanced": bool(rebalance),
            }
        )
        targets.append(target)
        holdings.append(np.asarray(info["weights"], dtype=float))
        decision = {
            "decision_id": f"d{decision_date:%Y%m%d}",
            "decision_date": str(decision_date.date()),
            "outcome_date": str(outcome.date()),
            "rebalanced": bool(rebalance),
            "status": (
                allocation.status if allocation else ("ok" if rebalance else "hold")
            ),
            "reason": allocation.reason if allocation else None,
            "time_status": "date_only_upstream_exchange_clock_not_verified",
        }
        decisions.append(decision)
        if allocation is not None and allocation.diagnostics is not None:
            optimization.append({**decision, **allocation.diagnostics})
        previous_month = month
        if done or truncated:
            break
    frame = pd.DataFrame(daily).set_index("date")
    expected_dates = data.features.index[~data.train_mask][1:]
    if not frame.index.equals(expected_dates):
        raise DataContractError(
            "replay does not cover the full producer scoring interval"
        )
    return Replay(
        frame,
        pd.DataFrame(targets, index=frame.index, columns=data.asset_order),
        pd.DataFrame(holdings, index=frame.index, columns=data.asset_order),
        decisions,
        optimization,
    )


def load_checkpoint(path, data, env):
    meta = read_json(path.with_suffix(".json"))
    if (
        meta.get("is_dummy") is not False
        or meta.get("stage") != "test"
        or meta.get("risk_columns") != []
        or meta.get("tickers") != data.asset_order
        or meta.get("lookback") != ENV_SETTINGS["lookback"]
        or meta.get("max_weight") != ENV_SETTINGS["max_weight"]
        or any(
            meta.get(k) != data.metadata[k]
            for k in ("window", "train_start", "train_end", "test_end")
        )
    ):
        raise DataContractError("checkpoint metadata differs from replay profile")
    from stable_baselines3 import PPO

    model = PPO.load(str(path), device="cpu")
    if (
        model.observation_space.shape != env.observation_space.shape
        or meta.get("obs_dim") != env.observation_space.shape[0]
        or model.action_space.shape != (len(data.asset_order),)
        or meta.get("timesteps", 0) < 100_000
        or model.num_timesteps < meta["timesteps"]
    ):
        raise DataContractError("checkpoint dimensions or training steps mismatch")
    return model, meta


def assert_aligned(daily, reference):
    """All policies must use exactly the same scored dates and benchmark."""
    if not daily.index.equals(reference.index) or not np.allclose(
        daily.benchmark_return, reference.benchmark_return, atol=1e-12, rtol=0
    ):
        raise DataContractError("comparison date/benchmark mismatch")


def save_run(
    output,
    run_id,
    replay,
    *,
    strategy,
    window,
    seed,
    reconciliation,
    source,
    include_in_anova=True,
):
    ledger = imported_ledger(replay.daily.reset_index())
    metrics = compute_metrics(
        ledger,
        replay.daily.benchmark_return,
        None,
        period_start=f"{2021 + window}-01-01",
    )
    folder = output / run_id
    folder.mkdir()
    ledger.to_parquet(folder / "ledger.parquet")
    replay.daily.to_parquet(folder / "daily.parquet")
    replay.targets.to_parquet(folder / "target_weights.parquet")
    replay.holdings.to_parquet(folder / "held_weights.parquet")
    write_json(folder / "decisions.json", replay.decisions)
    write_json(folder / "optimization.json", replay.optimization)
    fallback_count = sum(x["status"] == "fallback" for x in replay.optimization)
    run = {
        "strategy": strategy if seed is None else f"{strategy}_s{seed}",
        "strategy_family": strategy,
        "seed": seed,
        "fold_id": f"WF{2021 + window}",
        "execution_profile": PROFILE,
        "status": (
            "provisional_reference_mismatch"
            if reconciliation["status"] == "mismatch"
            else "provisional"
        ),
        **metrics,
        "source": source,
        "reconciliation": reconciliation,
        "decision_ids": [d["decision_id"] for d in replay.decisions],
        "shap_status": "not_available_in_this_bundle",
        "include_in_anova": include_in_anova,
        "rebalance": "monthly" if strategy in {"MVO", "EqualWeight"} else "daily",
        "optimization": {
            "count": len(replay.optimization),
            "fallback_count": fallback_count,
            "fallback_fraction": (
                fallback_count / len(replay.optimization)
                if replay.optimization
                else 0.0
            ),
        },
        "ledger_path": f"{run_id}/ledger.parquet",
    }
    write_json(folder / "run.json", run)
    samples = (
        [
            {
                "month": str(month),
                "strategy": strategy,
                "seed": seed,
                "return": value,
                "run_id": run_id,
                "execution_profile": PROFILE,
                "rebalance": run["rebalance"],
            }
            for month, value in monthly_returns(ledger.net_return_simple).items()
        ]
        if include_in_anova
        else []
    )
    return run, samples


def run_comparison(models_dir, processed_dir, rl_source_dir, output_directory):
    """36 preselected PPO runs + 8 monthly baselines + 4 daily EW controls.

    Each completed run is checkpointed. On failure, earlier runs and failure.json
    survive, but no successful evaluation.json is published. Never overwrites.
    """
    output, models = Path(output_directory), Path(models_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise DataContractError("output directory must be empty")
    reference_path = models / "eval_test_daily.csv"
    reference = load_daily(reference_path)
    _, metadata_hashes, assets = load_model_metadata(models, reference)
    source = {
        "reference_sha256": file_hash(reference_path),
        "environment_sha256": file_hash(Path(rl_source_dir) / "environment.py"),
        "evaluation_code_sha256": {
            p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "model_metadata_sha256": metadata_hashes,
    }
    output.mkdir(parents=True, exist_ok=True)
    runs, samples, verification = {}, [], {}
    active_run = "input_validation"
    try:
        for window in WINDOWS:
            active_run = f"window{window}_inputs"
            data = load_processed_window(processed_dir, window, require_prices=False)
            if data.asset_order != assets:
                raise DataContractError("producer and processed asset orders differ")
            verification[str(window)] = data.verification
            original = load_environment(rl_source_dir)
            direct = load_environment(rl_source_dir, direct_weights=True)
            window_reference = reference.loc[
                (reference.window == window) & (reference.strategy == "EqualWeight")
            ].set_index("date")
            for strategy, policy in (
                ("EqualWeightDailyControl", EqualWeight("daily")),
                ("MVO", MinimumVariance()),
                ("EqualWeight", EqualWeight()),
            ):
                active_run = (
                    f"{strategy}_monthly_w{window}"
                    if policy.rebalance == "monthly"
                    else f"{strategy}_w{window}"
                )
                env = build_environment(direct, data)
                replay = replay_run(env, data, policy=policy)
                assert_aligned(replay.daily, window_reference)
                reconciliation = (
                    compare_daily_replay(
                        replay.daily,
                        reference_path,
                        {"window": window, "seed": "-"},
                        strategy="EqualWeight",
                    )
                    if policy.rebalance == "daily"
                    else {"status": "new_baseline_common_dates_and_kernel_verified"}
                )
                run, rows = save_run(
                    output,
                    active_run,
                    replay,
                    strategy=strategy,
                    window=window,
                    seed=None,
                    reconciliation=reconciliation,
                    source=data.verification,
                    include_in_anova=policy.rebalance == "monthly",
                )
                runs[active_run], samples = run, samples + rows
            for variant in REWARD_VARIANTS:
                for seed in SEEDS:
                    path = models / f"ppo_test_w{window}_{variant}_s{seed}.zip"
                    active_run = path.stem.replace(".", "p")
                    env = build_environment(original, data)
                    model, meta = load_checkpoint(path, data, env)
                    if meta["variant"] != variant or meta["seed"] != seed:
                        raise DataContractError("checkpoint variant/seed mismatch")
                    replay = replay_run(env, data, model=model, seed=seed)
                    assert_aligned(replay.daily, window_reference)
                    reconciliation = daily_replay_report(
                        replay.daily, reference_path, meta
                    )
                    run, rows = save_run(
                        output,
                        active_run,
                        replay,
                        strategy=f"PPO-{variant}",
                        window=window,
                        seed=seed,
                        reconciliation=reconciliation,
                        source={
                            "model_sha256": file_hash(path),
                            "metadata_sha256": file_hash(path.with_suffix(".json")),
                            "inputs": data.verification,
                        },
                    )
                    runs[active_run], samples = run, samples + rows
        active_run = "monthly_statistics"
        frame = pd.DataFrame(samples)
        benchmark = pd.read_parquet(Path(processed_dir) / "returns.parquet")[
            "SPY_return"
        ]
        regime_error = None
        try:
            regimes = regimes_from_log_returns(benchmark, sorted(frame.month.unique()))
            frame = frame.merge(
                regimes[["month", "regime"]], on="month", validate="many_to_one"
            )
        except DataContractError as exc:
            regime_error = str(exc)
            regimes = pd.DataFrame(columns=["month", "regime"])
        averaged = aggregate_seeds(frame, expected_seeds=SEEDS)
        mission = analyze_monthly_samples(frame, expected_seeds=SEEDS)
        if regime_error:
            mission["tests"]["strategy_by_regime"] = {
                "status": "data_invalid",
                "reason": regime_error,
            }
        mission["profile"] = PROFILE
        mission["selection_reason"] = (
            "preexisting_mdd_lambda_1_default_not_best_test_result"
        )
        mission["regime_month_counts"] = {
            k: int(v) for k, v in regimes.regime.value_counts().items()
        }
        mismatched = [
            key
            for key, run in runs.items()
            if run["reconciliation"]["status"] == "mismatch"
        ]
        mission["input_provenance"] = (
            "all_samples_from_local_replays_no_imported_returns"
        )
        mission["reference_mismatched_runs"] = mismatched
        pivot = averaged.pivot(index="month", columns="strategy", values="value")
        pivot.index = pd.PeriodIndex(pivot.index, freq="M")
        bootstrap = [
            {
                **paired_block_bootstrap(
                    pivot["PPO-mdd_l1.0"], pivot[baseline], block_length=n
                ),
                "comparison": f"PPO-mdd_l1.0_seed_mean_minus_{baseline}_monthly",
            }
            for baseline in ("MVO", "EqualWeight")
            for n in (1, 3, 6)
        ]
        frame.to_parquet(output / "stat_samples.parquet", index=False)
        averaged.to_parquet(output / "stat_samples_seed_mean.parquet", index=False)
        regimes.to_csv(output / "regimes.csv", index=False)
        write_json(output / "mission-anova.json", mission)
        bundle = {
            "schema_version": "0.2",
            "data_mode": "real",
            "status": "provisional_reference_mismatch" if mismatched else "provisional",
            "real_model_loaded": True,
            "execution_profile": PROFILE,
            "execution_settings": {
                **ENV_SETTINGS,
                "baseline_rebalance": "first_decision_of_month",
                "target_cap_scope": "rebalance_only_drift_allowed",
                "weight_adapter": "private_module_identity_decoder",
            },
            "asset_order": assets,
            "source": source,
            "input_verification": verification,
            "runs": runs,
            "reference_reconciliation": {
                "status": "mismatch" if mismatched else "matched",
                "mismatched_runs": mismatched,
                "matched_run_count": sum(
                    run["reconciliation"]["status"] == "matched"
                    for run in runs.values()
                ),
                "scope": "PPO_and_daily_EW_only_monthly_baselines_are_new",
            },
            "mission_anova": mission,
            "bootstrap": bootstrap,
            "explanations": {},
            "disclaimer": DISCLAIMER,
            "limitations": LIMITATIONS,
            "coverage": {
                "ppo_replayed": sum(r["seed"] is not None for r in runs.values()),
                "monthly_baselines": sum(
                    r["rebalance"] == "monthly" for r in runs.values()
                ),
                "daily_ew_controls": sum(
                    not r["include_in_anova"] for r in runs.values()
                ),
                "not_replayed_variants": ["mdd_l0.5", "mdd_l5.0"],
            },
            "acceptance": {
                "status": "incomplete",
                "sharpe_target": None,
                "return_target": None,
                "reason": "FX/prices/RF/timing and final protocol remain unverified",
            },
        }
        write_json(output / "evaluation.json", bundle)
        return bundle
    except Exception as exc:
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "active_run": active_run,
                "error_type": type(exc).__name__,
                "reason": str(exc),
                "completed_runs": list(runs),
                "source": source,
            },
        )
        raise
