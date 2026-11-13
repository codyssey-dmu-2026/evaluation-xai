"""Import RL's published daily CSV; never equate imported results with a rerun.

Profile: rl-engine 05579d2, evaluate.py test export, unguarded daily rebalancing.
No checkpoint is loaded, no news is copied, and no missing observation is invented.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import DISCLAIMER
from .artifacts import read_json, write_json
from .metrics import compute_metrics
from .statistics import aggregate_seeds, anova, monthly_returns, paired_block_bootstrap
from .validation import DataContractError, validate_index

PROFILE = "rl_engine_05579d2_test_export_v1"
VARIANTS = ("simple", "sharpe", "mdd_l0.5", "mdd_l1.0", "mdd_l5.0")
SEEDS = (42, 43, 44)
WINDOWS = (1, 2, 3, 4)
REQUIRED = (
    "date",
    "window",
    "strategy",
    "seed",
    "net_return",
    "benchmark_return",
    "portfolio_value",
    "drawdown",
    "turnover",
    "max_target_weight",
)
LIMITATIONS = [
    "Imported producer-declared real results, not independently rerun or certified.",
    "Expanding training windows (2018 start) differ from fixed four-year training.",
    "Evaluation guard is off; MDD breaches are not liquidation events.",
    "RL trades before next-day returns, unlike evaluation's next-close execution.",
    "EqualWeight is daily, not the monthly baseline in evaluation spec v1.0.",
    "Mixed KRW/USD local returns lack a verified common-currency FX conversion.",
    "SPY adjusted ETF returns are a proxy, not the S&P 500 price index.",
    "First test-session coverage and exchange calendars need raw data verification.",
    "Reward uses simple return/current drawdown plus a terminal penalty, not v1.0.",
    "No MVO, historical regime warmup, real SHAP, risk integration or trade weights.",
    "Cost amounts are reconstructed from producer turnover and documented rates.",
    "No final liquidation cost was added; imported net returns are not changed.",
]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_daily(path):
    """Require the complete published 4-fold/3-seed grid and aligned daily rows."""
    frame = pd.read_csv(path, dtype={"strategy": str, "seed": str})
    if frame.empty or not set(REQUIRED).issubset(frame):
        raise DataContractError("RL daily CSV missing required columns or rows")
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    if frame[list(REQUIRED)].isna().any().any():
        raise DataContractError("RL daily CSV contains missing values")
    numeric = list(REQUIRED[4:])
    values = frame[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataContractError("RL daily CSV contains nonfinite numbers")
    if (frame[["net_return", "benchmark_return"]] <= -1).any().any():
        raise DataContractError("simple returns must exceed -1")
    if (frame.portfolio_value <= 0).any() or (frame.turnover < 0).any():
        raise DataContractError("invalid account value or turnover")
    if not frame.drawdown.between(0, 1).all():
        raise DataContractError("drawdown must be a positive loss fraction")
    if not frame.max_target_weight.between(0, 0.4 + 1e-6).all():
        raise DataContractError("target weight exceeds profile cap")
    expected = {(w, "EqualWeight", "-") for w in WINDOWS}
    expected |= {
        (w, f"PPO-{v}", str(s)) for w in WINDOWS for v in VARIANTS for s in SEEDS
    }
    groups = dict(tuple(frame.groupby(["window", "strategy", "seed"], sort=False)))
    if set(groups) != expected:
        raise DataContractError(
            "expected complete 4 windows x (5 variants x 3 seeds + EW)"
        )
    for (window, strategy, seed), group in groups.items():
        index = pd.DatetimeIndex(group.date)
        validate_index(index, f"{window}/{strategy}/{seed}")
        if not (index.year == 2021 + window).all():
            raise DataContractError("window contains dates outside its test year")
        baseline = groups[(window, "EqualWeight", "-")]
        if not index.equals(pd.DatetimeIndex(baseline.date)):
            raise DataContractError(
                "missing or misaligned daily observations between runs"
            )
        if not np.allclose(
            group.benchmark_return, baseline.benchmark_return, atol=1e-12, rtol=0
        ):
            raise DataContractError("benchmark differs between runs")
        if set(index.month) != set(range(1, 13)):
            raise DataContractError("each test window must cover all twelve months")
    return frame


def load_model_metadata(models_dir, frame):
    records, hashes, asset_order = {}, {}, None
    for window, strategy, seed in (
        frame[["window", "strategy", "seed"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        if strategy == "EqualWeight":
            continue
        variant = strategy.removeprefix("PPO-")
        path = Path(models_dir) / f"ppo_test_w{window}_{variant}_s{seed}.json"
        meta = read_json(path)
        expected = {
            "stage": "test",
            "window": window,
            "seed": int(seed),
            "variant": variant,
            "is_dummy": False,
        }
        if any(meta.get(key) != value for key, value in expected.items()):
            raise DataContractError(
                f"incompatible or dummy model metadata: {path.name}"
            )
        tickers = meta.get("tickers", [])
        if len(tickers) < 10 or len(set(tickers)) != len(tickers):
            raise DataContractError("metadata requires at least ten unique assets")
        if asset_order is not None and asset_order != tickers:
            raise DataContractError("asset order differs across models")
        asset_order = tickers
        expected_dates = {
            "train_start": "2018-01-01",
            "train_end": f"{2021 + window}-01-01",
            "test_end": f"{2022 + window}-01-01",
        }
        if any(meta.get(k) != v for k, v in expected_dates.items()):
            raise DataContractError(
                "model dates differ from the declared import profile"
            )
        if meta.get("risk_columns") != [] or meta.get("timesteps", 0) < 100_000:
            raise DataContractError(
                "profile expects no risk inputs and >=100000 training steps"
            )
        if meta.get("lookback") != 20 or meta.get("obs_dim") != len(tickers) * 24:
            raise DataContractError("unexpected observation layout")
        if meta.get("max_weight") != 0.4:
            raise DataContractError("unexpected model weight cap")
        # Do not re-publish producer-local absolute paths or timestamps unnecessarily.
        records[(window, strategy, seed)] = {
            **expected,
            **expected_dates,
            "timesteps": meta["timesteps"],
            "obs_dim": meta["obs_dim"],
        }
        hashes[path.name] = file_hash(path)
    return records, hashes, asset_order


def imported_ledger(group):
    """Reconcile account values before reconstructing documented cost amounts."""
    frame = group.set_index("date")
    end = frame.portfolio_value.to_numpy(dtype=float)
    r = frame.net_return.to_numpy(dtype=float)
    initial = end[0] / (1 + r[0])
    if not np.isclose(initial, 1_000_000, rtol=1e-8, atol=1e-6):
        raise DataContractError("unexpected initial equity for RL import profile")
    start = np.r_[initial, end[:-1]]
    if not np.allclose(end / start - 1, r, atol=1e-8, rtol=0):
        raise DataContractError("RL portfolio values do not reconcile with net returns")
    drawdown = 1 - end / np.maximum.accumulate(np.r_[initial, end])[1:]
    if not np.allclose(drawdown, frame.drawdown, atol=1e-8, rtol=0):
        raise DataContractError("RL drawdown does not reconcile with account values")
    return pd.DataFrame(
        {
            "equity_start": start,
            "equity_end": end,
            "net_return_simple": r,
            "net_return_log": np.log1p(r),
            "commission": start * frame.turnover.to_numpy() * 0.00015,
            "slippage": start * frame.turnover.to_numpy() * 0.0005,
            "drawdown": drawdown,
            "turnover": frame.turnover,
            "safeguard_triggered": False,
            "guard_active": False,
            "breach_15pct": drawdown > 0.15,
        },
        index=frame.index,
    )


def import_rl_results(models_dir, output_directory):
    """Create a dashboard-readable, explicitly provisional evaluation bundle."""
    models = Path(models_dir)
    output = Path(output_directory)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise DataContractError("output directory must be empty")
    source = models / "eval_test_daily.csv"
    daily = load_daily(source)
    metadata, hashes, assets = load_model_metadata(models, daily)
    runs, ledgers, samples = {}, {}, []
    for (window, strategy, seed), group in daily.groupby(
        ["window", "strategy", "seed"], sort=True
    ):
        label = strategy.replace(".", "p")
        run_id = f"{label}_w{window}" + ("" if seed == "-" else f"_s{seed}")
        ledger = imported_ledger(group)
        benchmark = group.set_index("date").benchmark_return
        metrics = compute_metrics(
            ledger, benchmark, None, period_start=f"{2021 + window}-01-01"
        )
        ledgers[run_id] = ledger
        runs[run_id] = {
            # Distinguish seeds in current dashboard, which groups by strategy/fold.
            "strategy": strategy if seed == "-" else f"{strategy}_s{seed}",
            "strategy_family": strategy,
            "seed": None if seed == "-" else int(seed),
            "fold_id": f"WF{2021 + window}",
            **metrics,
            "status": "provisional",
            "decision_ids": [],
            "metadata": metadata.get((window, strategy, seed)),
            "weights_status": "not_available",
            "shap_status": "not_available",
            "cost_provenance": "reconstructed_from_turnover_not_trade_records",
        }
        for month, value in monthly_returns(ledger.net_return_simple).items():
            samples.append(
                {
                    "month": str(month),
                    "strategy": strategy,
                    "seed": None if seed == "-" else int(seed),
                    "return": value,
                }
            )
    sample_frame = pd.DataFrame(samples)
    averaged = aggregate_seeds(sample_frame, expected_seeds=SEEDS)
    reward = anova(
        averaged[averaged.strategy.isin(["PPO-simple", "PPO-sharpe", "PPO-mdd_l1.0"])]
    )
    reward["status"] = "complete" if reward["status"] == "ok" else reward["status"]
    reward["scope"] = "exploratory_imported_returns_not_protocol_acceptance"
    reward["representative_mdd_lambda"] = 1.0
    reward["selection_reason"] = "preexisting_v1_default_not_best_test_result"
    reward["holm_status"] = "incomplete_other_mission_tests_missing"
    pivot = averaged.pivot(index="month", columns="strategy", values="value")
    pivot.index = pd.PeriodIndex(pivot.index, freq="M")
    boot = [
        paired_block_bootstrap(pivot["PPO-mdd_l1.0"], pivot.EqualWeight, block_length=n)
        for n in (1, 3, 6)
    ]
    for result in boot:
        result["comparison"] = "PPO-mdd_l1.0_seed_mean_minus_EqualWeight_daily"
    contains_zero = [b["ci_95"][0] <= 0 <= b["ci_95"][1] for b in boot]
    bundle = {
        "schema_version": "0.2",
        "import_profile": PROFILE,
        "data_mode": "imported_unverified",
        "producer_data_mode": "real",
        "real_model_loaded": False,
        "status": "provisional",
        "disclaimer": DISCLAIMER,
        "asset_order": assets,
        "source": {
            "daily_file": source.name,
            "sha256": file_hash(source),
            "metadata_sha256": hashes,
        },
        "limitations": LIMITATIONS
        + ["RF is missing: Sharpe/Sortino/Alpha/Beta are null, not RF=0 estimates."],
        "runs": runs,
        "explanations": {},
        "mission_anova": {
            "reward_variants": reward,
            "strategies": {
                "status": "incomplete",
                "reason": "MVO results under the same execution profile are missing",
            },
            "strategy_by_regime": {
                "status": "incomplete",
                "reason": "MVO and verified pre-test benchmark history are missing",
            },
        },
        "bootstrap": boot,
        "bootstrap_warning": (
            "unstable_interval" if len(set(contains_zero)) > 1 else None
        ),
        "acceptance": {
            "status": "incomplete",
            "sharpe_target": None,
            "return_target": None,
            "reason": (
                "Imported execution profile differs from evaluation v1.0; "
                "inputs and MVO/SHAP missing"
            ),
        },
    }
    # Finish validation/calculation before writing; never publish a half-ready bundle.
    output.mkdir(parents=True, exist_ok=True)
    for run_id, ledger in ledgers.items():
        folder = output / run_id
        folder.mkdir()
        ledger.to_parquet(folder / "ledger.parquet")
    sample_frame.to_parquet(output / "stat_samples.parquet", index=False)
    averaged.to_parquet(output / "stat_samples_seed_mean.parquet", index=False)
    write_json(output / "evaluation.json", bundle)
    return bundle
