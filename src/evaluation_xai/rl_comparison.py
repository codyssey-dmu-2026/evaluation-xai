"""Return-based baselines and PPO replays under the producer's execution kernel.

This is a separate, provisional profile, not evaluation v1's next-close backtest.
Only trusted local environment source and checkpoints may be passed to this module.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import DISCLAIMER
from .artifacts import read_json
from .baselines import EqualWeight, MinimumVariance
from .data_loader import load_processed_window
from .metrics import compute_metrics
from .mission import analyze_monthly_samples
from .provenance import atomic_write_bytes, atomic_write_json, json_bytes
from .provenance import runtime_info
from .real_xai import compare_daily_replay, daily_replay_report, observation_sha256
from .statistics import (
    aggregate_seeds,
    monthly_returns,
    paired_block_bootstrap,
    regimes_from_log_returns,
)
from .team_import import SEEDS, VARIANTS, WINDOWS, file_hash, imported_ledger
from .team_import import load_daily, load_model_metadata
from .validation import DataContractError, validate_weights

PROFILE = "rl_return_comparison_monthly_v1"
SCHEMA_VERSION = "0.3"
# Predeclared statistical set; fixed before results, never chosen by performance.
ANOVA_VARIANTS = ("simple", "sharpe", "mdd_l1.0")
REWARD_VARIANTS = ANOVA_VARIANTS
SUPPLEMENTARY_VARIANTS = tuple(v for v in VARIANTS if v not in ANOVA_VARIANTS)
MONTHLY_BASELINES = ("MVO", "EqualWeight")
TEST_STRATEGIES = {
    "reward_variants": ("PPO-simple", "PPO-sharpe", "PPO-mdd_l1.0"),
    "strategies": ("PPO-mdd_l1.0", "MVO", "EqualWeight"),
    "strategy_by_regime": ("PPO-mdd_l1.0", "MVO", "EqualWeight"),
}
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
    "MVO uses the last 252 union-calendar (KR+US) observations, including "
    "zero returns on closed sessions; not 252 exchange trading days.",
    "PPO float32 outputs differ by platform/BLAS; reconciliation status is "
    "runtime-dependent and recorded with source.runtime.",
    "Supplementary lambda 0.5/5.0 runs are descriptive only and excluded from ANOVA.",
    "No new SHAP; existing SHAP is attached only after exact input verification.",
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
        observation_hash = None
        if model is not None:
            # Hash the exact policy input BEFORE acting: the SHAP link key.
            observation_hash = observation_sha256(obs)
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
            "observation_sha256": observation_hash,
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


def monthly_reference_error(daily, reference, window, strategy, seed):
    """Audit-only month-level effect of a replay/CSV difference; never a sample."""
    selected = reference.loc[
        (reference.window == window)
        & (reference.strategy == strategy)
        & (reference.seed == seed)
    ].set_index("date")
    left = monthly_returns(daily.net_return)
    right = monthly_returns(selected.net_return)
    if not left.index.equals(right.index):
        raise DataContractError("replay/reference months differ")
    return float(np.max(np.abs(left.to_numpy() - right.to_numpy())))


RUN_FILES = (
    "ledger.parquet",
    "daily.parquet",
    "weights.parquet",
    "target_weights.parquet",
    "held_weights.parquet",
    "decisions.json",
    "optimization.json",
)


def build_run(replay, *, strategy, window, seed, reconciliation, source, anova):
    ledger = imported_ledger(replay.daily.reset_index())
    metrics = compute_metrics(
        ledger,
        replay.daily.benchmark_return,
        None,
        period_start=f"{2021 + window}-01-01",
    )
    fallback_count = sum(x["status"] == "fallback" for x in replay.optimization)
    run = {
        "strategy": strategy if seed is None else f"{strategy}_s{seed}",
        "strategy_family": strategy,
        "seed": seed,
        "window": window,
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
        "include_in_anova": anova,
        "rebalance": "monthly" if strategy in MONTHLY_BASELINES else "daily",
        "optimization": {
            "count": len(replay.optimization),
            "fallback_count": fallback_count,
            "fallback_fraction": (
                fallback_count / len(replay.optimization)
                if replay.optimization
                else 0.0
            ),
        },
    }
    return run, ledger


def persist_run(output, run_id, replay, run, ledger):
    """Write into a hidden partial folder, hash every file, then rename atomically.

    A folder named run_id therefore always holds a complete, hash-checked run.
    """
    partial = output / f".partial-{run_id}"
    partial.mkdir()
    ledger.to_parquet(partial / "ledger.parquet")
    replay.daily.to_parquet(partial / "daily.parquet")
    # weights.parquet: the per-outcome-day weights file name every evaluation-xai
    # backtest folder uses (platform timeseries reads it); same as target weights.
    replay.targets.to_parquet(partial / "weights.parquet")
    replay.targets.to_parquet(partial / "target_weights.parquet")
    replay.holdings.to_parquet(partial / "held_weights.parquet")
    atomic_write_json(partial / "decisions.json", replay.decisions)
    atomic_write_json(partial / "optimization.json", replay.optimization)
    run = {
        **run,
        "ledger_path": f"{run_id}/ledger.parquet",
        "artifact_sha256": {name: file_hash(partial / name) for name in RUN_FILES},
    }
    atomic_write_json(partial / "run.json", run)
    os.replace(partial, output / run_id)
    return run


def load_completed_run(output, run_id):
    """Reuse a finished run only if every recorded file hash still matches."""
    folder = output / run_id
    run = read_json(folder / "run.json")
    hashes = run.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(RUN_FILES):
        raise DataContractError(f"completed run lacks file hashes: {run_id}")
    for name, expected in hashes.items():
        if file_hash(folder / name) != expected:
            raise DataContractError(f"completed run file changed: {run_id}/{name}")
    ledger = pd.read_parquet(folder / "ledger.parquet")
    decisions = read_json(folder / "decisions.json")
    return run, ledger, decisions


def monthly_samples(run_id, run, ledger, decisions):
    """Monthly compounding per run; seed averaging happens later, never here."""
    series = ledger.net_return_simple
    by_outcome = {d["outcome_date"]: d["decision_id"] for d in decisions}
    rows = []
    for month, value in monthly_returns(series).items():
        dates = series.index[series.index.to_period("M") == month]
        rows.append(
            {
                "month": str(month),
                "strategy": run["strategy_family"],
                "seed": run["seed"],
                "return": value,
                "run_id": run_id,
                "execution_profile": PROFILE,
                "rebalance": run["rebalance"],
                "first_outcome_date": str(dates[0].date()),
                "last_outcome_date": str(dates[-1].date()),
                "first_decision_id": by_outcome[str(dates[0].date())],
                "last_decision_id": by_outcome[str(dates[-1].date())],
            }
        )
    return rows


def planned_runs(window, variants):
    """Deterministic run order: kernel control, monthly baselines, then PPO."""
    plan = [
        ("EqualWeightDailyControl", f"EqualWeightDailyControl_w{window}", None),
        ("MVO", f"MVO_monthly_w{window}", None),
        ("EqualWeight", f"EqualWeight_monthly_w{window}", None),
    ]
    for variant in (v for v in VARIANTS if v in variants):
        for seed in SEEDS:
            stem = f"ppo_test_w{window}_{variant}_s{seed}"
            plan.append((variant, stem.replace(".", "p"), seed))
    return plan


def input_fingerprint(models, processed_dir, rl_source_dir, variants):
    processed = Path(processed_dir)
    model_files = {}
    for window in WINDOWS:
        for variant in (v for v in VARIANTS if v in variants):
            for seed in SEEDS:
                path = models / f"ppo_test_w{window}_{variant}_s{seed}.zip"
                model_files[path.name] = file_hash(path)
    return {
        "profile": PROFILE,
        "schema_version": SCHEMA_VERSION,
        "env_settings": ENV_SETTINGS,
        "variants": [v for v in VARIANTS if v in variants],
        "anova_variants": list(ANOVA_VARIANTS),
        "reference_sha256": file_hash(models / "eval_test_daily.csv"),
        "environment_sha256": file_hash(Path(rl_source_dir) / "environment.py"),
        "evaluation_code_sha256": {
            p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "model_sha256": model_files,
        "processed_sha256": {
            p.name: file_hash(p)
            for p in sorted(processed.iterdir())
            if p.is_file() and p.suffix in {".parquet", ".json"}
        },
        "runtime": runtime_info(),
    }


def prepare_output(output, fingerprint, resume):
    """Fresh runs need an empty folder; resume needs an identical fingerprint."""
    manifest_path = output / "run_manifest.json"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not resume:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise DataContractError("output directory must be empty")
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"fingerprint": fingerprint, "attempts": []}
    else:
        if not manifest_path.is_file():
            raise DataContractError("resume requires an existing run_manifest.json")
        if (output / "evaluation.json").exists():
            raise DataContractError("resume refused: bundle already published")
        manifest = read_json(manifest_path)
        previous = manifest.get("fingerprint", {})
        changed = sorted(
            key
            for key in set(previous) | set(fingerprint)
            if previous.get(key) != fingerprint.get(key)
        )
        if changed:
            raise DataContractError(
                "resume refused: inputs/code/runtime changed: " + ", ".join(changed)
            )
        attempts_dir = output / "attempts"
        failure = output / "failure.json"
        if failure.exists():
            attempts_dir.mkdir(exist_ok=True)
            os.replace(
                failure, attempts_dir / f"failure-{len(manifest['attempts'])}.json"
            )
        for partial in sorted(output.glob(".partial-*")):
            # Preserve, never delete, interrupted output for later inspection.
            abandoned = output / ".abandoned"
            abandoned.mkdir(exist_ok=True)
            target = (
                abandoned / f"{partial.name[9:]}-attempt{len(manifest['attempts'])}"
            )
            os.replace(partial, target)
    manifest["attempts"].append(
        {"attempt": len(manifest["attempts"]) + 1, "started": now}
    )
    manifest["status"] = "running"
    atomic_write_json(manifest_path, manifest)
    return manifest


def reconciliation_diagnosis(runs, mismatched, runtime):
    """Facts first; the cause stays a labelled hypothesis until reproduced."""
    checked = {
        k: r for k, r in runs.items() if "max_absolute_errors" in r["reconciliation"]
    }
    kernel = {
        k: r["reconciliation"]
        for k, r in checked.items()
        if r["strategy_family"] == "EqualWeightDailyControl"
    }
    ppo = {k: r["reconciliation"] for k, r in checked.items() if r["seed"] is not None}

    def span(group, key, field):
        values = [entry[key][field] for entry in group.values() if key in entry]
        return [min(values), max(values)] if values else None

    return {
        "classification": (
            "unresolved_numeric_difference" if mismatched else "within_strict_tolerance"
        ),
        "runtime_dependent": True,
        "runtime": runtime,
        "kernel_controls": {
            "runs": len(kernel),
            "matched": sum(v["status"] == "matched" for v in kernel.values()),
            "net_return_max_abs_error": span(
                kernel, "max_absolute_errors", "net_return"
            ),
        },
        "ppo": {
            "runs": len(ppo),
            "matched": sum(v["status"] == "matched" for v in ppo.values()),
            "bitwise_identical": sum(
                bool(v.get("bitwise_identical")) for v in ppo.values()
            ),
            "max_target_weight_abs_error_range": span(
                ppo, "max_absolute_errors", "max_target_weight"
            ),
            "net_return_abs_error_range": span(
                ppo, "max_absolute_errors", "net_return"
            ),
            "drawdown_tolerance_ratio_range": span(
                ppo, "max_tolerance_ratio", "drawdown"
            ),
            "monthly_return_abs_error_range": (
                [
                    min(v["monthly_return_max_abs_error"] for v in ppo.values()),
                    max(v["monthly_return_max_abs_error"] for v in ppo.values()),
                ]
                if ppo
                and all("monthly_return_max_abs_error" in v for v in ppo.values())
                else None
            ),
        },
        "mismatched_runs": {
            k: {
                "fields": runs[k]["reconciliation"]["mismatched_fields"],
                "max_tolerance_ratio": runs[k]["reconciliation"].get(
                    "max_tolerance_ratio"
                ),
                "monthly_return_max_abs_error": runs[k]["reconciliation"].get(
                    "monthly_return_max_abs_error"
                ),
            }
            for k in mismatched
        },
        "hypothesis": (
            "Float32 policy-network arithmetic differs by platform/BLAS path; "
            "weight differences near float32 epsilon compound in portfolio value "
            "and exceed the strict drawdown atol on low-drawdown days."
        ),
        "not_established": [
            "Producer platform, torch build and thread settings are unknown.",
            "No tested platform reproduces the producer CSV bit for bit.",
        ],
        "tolerance_changed": False,
    }


def mission_bundle_view(mission):
    """One mission_anova contract for every evaluation bundle (import-rl too).

    Tests sit at the top level; a computed test is "complete" (import-rl's word),
    which says nothing about significance, assumptions or acceptance.
    """
    view = {}
    for name in TEST_STRATEGIES:
        test = dict(mission["tests"][name])
        if test.get("status") == "ok":
            test["status"] = "complete"
        view[name] = test
    view["reward_variants"]["representative_mdd_lambda"] = 1.0
    view["reward_variants"]["selection_reason"] = mission["selection_reason"]
    view["overall_status"] = (
        "complete" if mission["status"] == "ok" else (mission["status"])
    )
    view.update({k: v for k, v in mission.items() if k not in {"tests", "status"}})
    view["status_vocabulary"] = (
        "complete = computed; not significance, assumption validity or acceptance"
    )
    return view


def annotate_test_inputs(mission, frame, runs, mismatched):
    for name, strategies in TEST_STRATEGIES.items():
        used = sorted(set(frame.loc[frame.strategy.isin(strategies), "run_id"]))
        bad = sorted(set(used) & set(mismatched))
        mission["tests"][name]["input_reconciliation"] = {
            "status": (
                "includes_reference_mismatch" if bad else "no_reference_mismatch"
            ),
            "mismatched_runs": bad,
            "new_baselines_without_reference": sorted(
                k for k in used if runs[k]["rebalance"] == "monthly"
            ),
            "run_count": len(used),
        }


def link_explanations(output, runs, decisions_by_run, directories, environment_sha):
    """Attach pre-existing SHAP only when model, inputs and the exact observation match.

    A SHAP value explains one input vector. A different model, input file, or any
    bit of the observation means it explains a different decision: reject it.
    """
    explanations, plots, report = {}, {}, []
    for directory in map(Path, directories):
        entry = {"source": str(directory.name), "status": "rejected", "checks": {}}
        report.append(entry)
        try:
            bundle_path = directory / "evaluation.json"
            entry["source_evaluation_sha256"] = file_hash(bundle_path)
            bundle = read_json(bundle_path)
            observation = np.load(directory / "explanation_inputs.npz")["observation"]
        except (OSError, ValueError, KeyError) as exc:
            entry["reason"] = f"unreadable_explanation_bundle: {type(exc).__name__}"
            continue
        items = bundle.get("explanations", {})
        if bundle.get("data_mode") != "real" or len(items) != 1:
            entry["reason"] = "expected exactly one real explanation"
            continue
        key, explanation = next(iter(items.items()))
        run_id, _, decision_id = key.partition("/")
        entry["key"] = key
        run = runs.get(run_id)
        source = bundle.get("source", {})
        decision = next(
            (
                d
                for d in decisions_by_run.get(run_id, [])
                if d["decision_id"] == decision_id
            ),
            None,
        )
        inputs = (run or {}).get("source", {}).get("inputs", {}).get("file_sha256", {})
        checks = {
            "ppo_run_in_bundle": run is not None and run.get("seed") is not None,
            "model_sha256": run is not None
            and source.get("model_sha256") == run["source"].get("model_sha256"),
            "model_metadata_sha256": run is not None
            and source.get("model_metadata_sha256")
            == run["source"].get("metadata_sha256"),
            "environment_sha256": source.get("environment_sha256") == environment_sha,
            "returns_sha256": source.get("returns_sha256")
            == inputs.get("returns.parquet"),
            "manifest_sha256": source.get("manifest_sha256")
            == inputs.get("walk_forward_manifest.json"),
            "features_sha256": source.get("features_sha256") in set(inputs.values()),
            "decision_in_run": decision is not None,
            "producer_reconciliation_not_mismatch": bundle.get(
                "reconciliation", {}
            ).get("status")
            != "mismatch",
        }
        observed = observation_sha256(observation)
        checks["observation_sha256"] = (
            decision is not None and decision.get("observation_sha256") == observed
        )
        if "observation_sha256" in explanation:
            checks["explanation_records_same_observation"] = (
                explanation["observation_sha256"] == observed
            )
        if decision is not None:
            checks["decision_date"] = (
                explanation.get("decision_date") == decision["decision_date"]
            )
        if all(checks.values()):
            targets = pd.read_parquet(output / run_id / "target_weights.parquet")
            actual = targets.loc[pd.Timestamp(decision["outcome_date"])]
            checks["asset_order"] = list(targets.columns) == explanation.get(
                "asset_names"
            )
            checks["predicted_weights_match_replay_target"] = bool(
                checks["asset_order"]
                and np.allclose(
                    explanation["predicted_weights"],
                    actual.to_numpy(),
                    atol=1e-6,
                    rtol=0,
                )
            )
        if key in explanations:
            checks["unique_key"] = False
        entry["checks"] = checks
        if not all(checks.values()):
            entry["reason"] = "failed: " + ", ".join(
                k for k, v in checks.items() if not v
            )
            continue
        for plot_key, kinds in bundle.get("plot_artifacts", {}).items():
            if not plot_key.startswith(key + "/"):
                continue
            asset = plot_key.rsplit("/", 1)[1]
            copied = {}
            for kind, relative in kinds.items():
                source_file = (directory / str(relative)).resolve()
                if (
                    not source_file.is_relative_to(directory.resolve())
                    or not source_file.is_file()
                ):
                    # Plots are derived views; values stay linked, plot is not served.
                    entry.setdefault("plots_not_copied", []).append(f"{asset}/{kind}")
                    continue
                destination = Path("explanations") / run_id / decision_id / asset
                (output / destination).mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, output / destination / source_file.name)
                copied[kind] = str(destination / source_file.name)
            if copied:
                plots[plot_key] = copied
        explanations[key] = {
            **explanation,
            "link": {
                "status": "verified_same_model_inputs_and_observation",
                "source_bundle": entry["source"],
                "source_evaluation_sha256": entry["source_evaluation_sha256"],
                "observation_sha256": observed,
                "shap_recomputed": False,
            },
        }
        entry["status"] = "linked"
    return explanations, plots, report


def _publish_frame(frame, path, writer):
    temporary = path.with_name(f".{path.name}.tmp")
    writer(frame, temporary)
    os.replace(temporary, path)


def run_comparison(
    models_dir,
    processed_dir,
    rl_source_dir,
    output_directory,
    *,
    variants=ANOVA_VARIANTS,
    resume=False,
    explanation_dirs=(),
):
    """PPO replays + 8 monthly baselines + 4 daily EW controls, resumable.

    Each finished run is renamed into place atomically and hash-recorded. On
    failure, finished runs and failure.json survive and evaluation.json is not
    published; --resume reuses only runs whose files and fingerprint are intact.
    Statistics always use the predeclared ANOVA_VARIANTS, whatever was replayed.
    """
    variants = tuple(v for v in VARIANTS if v in set(variants))
    if not set(ANOVA_VARIANTS) <= set(variants) or len(set(variants)) != len(variants):
        raise DataContractError("replay must include the predeclared ANOVA variants")
    output, models = Path(output_directory), Path(models_dir)
    reference_path = models / "eval_test_daily.csv"
    fingerprint = input_fingerprint(models, processed_dir, rl_source_dir, variants)
    manifest = prepare_output(output, fingerprint, resume)
    attempt = len(manifest["attempts"])
    source = {
        "reference_sha256": fingerprint["reference_sha256"],
        "environment_sha256": fingerprint["environment_sha256"],
        "evaluation_code_sha256": fingerprint["evaluation_code_sha256"],
        "runtime": fingerprint["runtime"],
    }
    runs, samples, supplementary, verification = {}, [], [], {}
    decisions_by_run, reused = {}, []
    active_run = "input_validation"
    try:
        reference = load_daily(reference_path, exact=True)
        _, metadata_hashes, assets = load_model_metadata(models, reference)
        source["model_metadata_sha256"] = metadata_hashes
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
            for family, run_id, seed in planned_runs(window, variants):
                active_run = run_id
                if (output / run_id).exists():
                    run, ledger, decisions = load_completed_run(output, run_id)
                    reused.append(run_id)
                elif seed is None:
                    policy = (
                        MinimumVariance()
                        if family == "MVO"
                        else EqualWeight(
                            "monthly" if family == "EqualWeight" else "daily"
                        )
                    )
                    replay = replay_run(
                        build_environment(direct, data), data, policy=policy
                    )
                    assert_aligned(replay.daily, window_reference)
                    if policy.rebalance == "daily":
                        reconciliation = compare_daily_replay(
                            replay.daily,
                            reference_path,
                            {"window": window, "seed": "-"},
                            strategy="EqualWeight",
                        )
                        reconciliation["monthly_return_max_abs_error"] = (
                            monthly_reference_error(
                                replay.daily, reference, window, "EqualWeight", "-"
                            )
                        )
                    else:
                        reconciliation = {
                            "status": "new_baseline_common_dates_and_kernel_verified"
                        }
                    run, ledger = build_run(
                        replay,
                        strategy=family,
                        window=window,
                        seed=None,
                        reconciliation=reconciliation,
                        source=data.verification,
                        anova=policy.rebalance == "monthly",
                    )
                    run = persist_run(output, run_id, replay, run, ledger)
                    decisions = replay.decisions
                else:
                    path = models / f"ppo_test_w{window}_{family}_s{seed}.zip"
                    env = build_environment(original, data)
                    model, meta = load_checkpoint(path, data, env)
                    if meta["variant"] != family or meta["seed"] != seed:
                        raise DataContractError("checkpoint variant/seed mismatch")
                    replay = replay_run(env, data, model=model, seed=seed)
                    assert_aligned(replay.daily, window_reference)
                    reconciliation = daily_replay_report(
                        replay.daily, reference_path, meta
                    )
                    reconciliation["monthly_return_max_abs_error"] = (
                        monthly_reference_error(
                            replay.daily, reference, window, f"PPO-{family}", str(seed)
                        )
                    )
                    run, ledger = build_run(
                        replay,
                        strategy=f"PPO-{family}",
                        window=window,
                        seed=seed,
                        reconciliation=reconciliation,
                        source={
                            "model_sha256": file_hash(path),
                            "metadata_sha256": file_hash(path.with_suffix(".json")),
                            "inputs": data.verification,
                        },
                        anova=family in ANOVA_VARIANTS,
                    )
                    run = persist_run(output, run_id, replay, run, ledger)
                    decisions = replay.decisions
                runs[run_id], decisions_by_run[run_id] = run, decisions
                if run["include_in_anova"]:
                    samples += monthly_samples(run_id, run, ledger, decisions)
                elif run["seed"] is not None:
                    supplementary += monthly_samples(run_id, run, ledger, decisions)
        active_run = "monthly_statistics"
        frame = pd.DataFrame(samples)
        regime_error = None
        try:
            benchmark = pd.read_parquet(Path(processed_dir) / "returns.parquet")[
                "SPY_return"
            ]
            regimes = regimes_from_log_returns(benchmark, sorted(frame.month.unique()))
            frame = frame.merge(
                regimes[["month", "regime"]], on="month", validate="many_to_one"
            )
        except (DataContractError, KeyError, OSError, ValueError) as exc:
            # Only test 3 depends on regimes; tests 1-2 stay valid and reported.
            regime_error = f"{type(exc).__name__}: {exc}"
            regimes = pd.DataFrame(columns=["month", "regime"])
        averaged = aggregate_seeds(frame, expected_seeds=SEEDS)
        mission = analyze_monthly_samples(frame, expected_seeds=SEEDS)
        if regime_error:
            mission["tests"]["strategy_by_regime"] = {
                "status": "data_invalid",
                "reason": regime_error,
            }
        mismatched = [
            key
            for key, run in runs.items()
            if run["reconciliation"]["status"] == "mismatch"
        ]
        annotate_test_inputs(mission, frame, runs, mismatched)
        mission["profile"] = PROFILE
        mission["selection_reason"] = (
            "preexisting_mdd_lambda_1_default_not_best_test_result"
        )
        mission["regime_month_counts"] = {
            k: int(v) for k, v in regimes.regime.value_counts().items()
        }
        mission["input_provenance"] = (
            "all_samples_from_local_replays_no_imported_returns"
        )
        mission["reference_mismatched_runs"] = mismatched
        mission["runtime"] = fingerprint["runtime"]
        mission_view = mission_bundle_view(mission)
        pivot = averaged.pivot(index="month", columns="strategy", values="value")
        pivot.index = pd.PeriodIndex(pivot.index, freq="M")
        bootstrap = [
            {
                **paired_block_bootstrap(
                    pivot["PPO-mdd_l1.0"], pivot[baseline], block_length=n
                ),
                "comparison": f"PPO-mdd_l1.0_seed_mean_minus_{baseline}_monthly",
            }
            for baseline in MONTHLY_BASELINES
            for n in (1, 3, 6)
        ]
        active_run = "explanation_links"
        explanations, plot_artifacts, links = link_explanations(
            output,
            runs,
            decisions_by_run,
            explanation_dirs,
            fingerprint["environment_sha256"],
        )
        runs = {key: dict(run) for key, run in runs.items()}
        for key in explanations:
            run_id, _, decision_id = key.partition("/")
            runs[run_id]["shap_status"] = "linked_verified_existing_shap"
            runs[run_id].setdefault("explained_decision_ids", []).append(decision_id)
        replayed = sorted({r["strategy_family"] for r in runs.values() if r["seed"]})
        bundle = {
            "schema_version": SCHEMA_VERSION,
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
                "diagnosis": reconciliation_diagnosis(
                    runs, mismatched, fingerprint["runtime"]
                ),
            },
            "mission_anova": mission_view,
            "bootstrap": bootstrap,
            "explanations": explanations,
            "plot_artifacts": plot_artifacts,
            "explanation_links": links,
            "disclaimer": DISCLAIMER,
            "limitations": LIMITATIONS,
            "coverage": {
                "ppo_replayed": sum(r["seed"] is not None for r in runs.values()),
                "replayed_ppo_strategies": replayed,
                "anova_variants": list(ANOVA_VARIANTS),
                "supplementary_variants_replayed": [
                    v for v in variants if v not in ANOVA_VARIANTS
                ],
                "monthly_baselines": sum(
                    r["rebalance"] == "monthly" for r in runs.values()
                ),
                "daily_ew_controls": sum(
                    r["strategy_family"] == "EqualWeightDailyControl"
                    for r in runs.values()
                ),
                "not_replayed_variants": [v for v in VARIANTS if v not in variants],
                "reused_completed_runs": reused,
                "linked_explanations": len(explanations),
            },
            "attempts": manifest["attempts"],
            "acceptance": {
                "status": "incomplete",
                "sharpe_target": None,
                "return_target": None,
                "reason": "FX/prices/RF/timing and final protocol remain unverified",
            },
        }
        # Serialize everything BEFORE writing: a late error leaves no partial stats.
        active_run = "bundle_serialization"
        bundle_bytes, mission_bytes = json_bytes(bundle), json_bytes(mission_view)
        active_run = "publish"
        _publish_frame(
            frame,
            output / "stat_samples.parquet",
            lambda f, p: f.to_parquet(p, index=False),
        )
        _publish_frame(
            averaged,
            output / "stat_samples_seed_mean.parquet",
            lambda f, p: f.to_parquet(p, index=False),
        )
        _publish_frame(
            pd.DataFrame(supplementary),
            output / "supplementary_samples.parquet",
            lambda f, p: f.to_parquet(p, index=False),
        )
        _publish_frame(
            regimes, output / "regimes.csv", lambda f, p: f.to_csv(p, index=False)
        )
        atomic_write_bytes(output / "mission-anova.json", mission_bytes)
        atomic_write_bytes(output / "evaluation.json", bundle_bytes)
        manifest["status"] = "published"
        manifest["attempts"][-1]["finished"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        atomic_write_json(output / "run_manifest.json", manifest)
        return bundle
    except BaseException as exc:
        atomic_write_json(
            output / "failure.json",
            {
                "status": "failed",
                "attempt": attempt,
                "active_run": active_run,
                "error_type": type(exc).__name__,
                "reason": str(exc),
                "completed_runs": list(runs),
                "reused_runs": reused,
                "source": source,
                "resume": "rerun with --resume and the same inputs/code/runtime",
            },
        )
        manifest["status"] = "failed"
        atomic_write_json(output / "run_manifest.json", manifest)
        raise
