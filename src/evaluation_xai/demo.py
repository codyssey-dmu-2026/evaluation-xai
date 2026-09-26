"""Entirely synthetic smoke test. No external data or trained PPO is used."""

from pathlib import Path

import numpy as np
import pandas as pd

from . import DISCLAIMER
from .artifacts import write_json
from .baselines import EqualWeight, MinimumVariance
from .config import annual_folds
from .metrics import compute_metrics
from .statistics import anova, monthly_returns, paired_block_bootstrap
from .walk_forward import run_walk_forward


def synthetic_prices(seed=20260926, start="2017-01-02", end="2025-12-31", assets=10):
    dates = pd.bdate_range(start, end, name="date")
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0002, 0.006, (len(dates), 1))
    noise = rng.normal(0, np.linspace(0.002, 0.015, assets), (len(dates), assets))
    return pd.DataFrame(
        100 * np.exp(np.cumsum(common + noise, axis=0)),
        index=dates,
        columns=[f"SYN{i:02d}" for i in range(assets)],
    )


def run_demo(output_directory, *, with_shap=False, first_year=2022, last_year=2025):
    output = Path(output_directory)
    # Explicitly refuse clobbering previous work, including user-created directories.
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; choose a new --output path")
    output.mkdir(parents=True, exist_ok=True)
    prices = synthetic_prices()
    benchmark = prices.mean(axis=1).pct_change(fill_method=None)
    folds = annual_folds(first_year, last_year)
    bundle = {
        "schema_version": "0.1",
        "data_mode": "synthetic",
        "real_model_loaded": False,
        "seed": 20260926,
        "disclaimer": DISCLAIMER,
        "limitations": [
            "Not investment performance; synthetic prices and zero risk-free rate.",
            "Business-day calendar is not an exchange calendar.",
            "No trained PPO, news integration or completed three-part mission ANOVA.",
        ],
        "runs": {},
        "explanations": {},
    }
    monthly = {}
    for name, factory in [("equal_weight", EqualWeight), ("mvo", MinimumVariance)]:
        results = run_walk_forward(prices, folds, lambda train, fold: factory())
        series = []
        for fold_id, result in results.items():
            run_id = f"{name}_{fold_id}"
            folder = output / run_id
            folder.mkdir()
            result.ledger.to_parquet(folder / "ledger.parquet")
            result.weights.to_parquet(folder / "weights.parquet")
            result.decisions.to_json(
                folder / "decisions.json", orient="records", date_format="iso", indent=2
            )
            metrics = compute_metrics(
                result.ledger,
                benchmark.loc[result.ledger.index],
                pd.Series(0.0, index=result.ledger.index),
                period_start=result.period_start,
            )
            bundle["runs"][run_id] = {
                "strategy": name,
                "fold_id": fold_id,
                **metrics,
                "decision_ids": result.decisions.decision_id.tolist(),
            }
            series.append(monthly_returns(result.ledger.net_return_simple))
        monthly[name] = pd.concat(series).sort_index()
    rows = [
        {"strategy": name, "value": value, "month": str(month)}
        for name, values in monthly.items()
        for month, value in values.items()
    ]
    bundle["synthetic_baseline_comparison"] = anova(pd.DataFrame(rows))
    bundle["bootstrap"] = [
        paired_block_bootstrap(
            monthly["mvo"], monthly["equal_weight"], block_length=block
        )
        for block in (1, 3, 6)
    ]
    bundle["mission_anova"] = {
        name: {
            "status": "incomplete",
            "reason": "real PPO runs and validated market regime observations required",
        }
        for name in ("reward_variants", "strategies", "strategy_by_regime")
    }
    if with_shap:
        from .explain import explain_weights, render_explanations

        rng = np.random.default_rng(11)
        bg = rng.normal(size=(32, 6))
        coefficients = rng.normal(0, 0.1, (6, 10))

        def toy_policy(batch):
            logits = batch @ coefficients
            exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
            return exponent / exponent.sum(axis=1, keepdims=True)

        explanations = []
        decision_ids = [f"toy{i}" for i in range(3)]
        for decision in decision_ids:
            explanation = explain_weights(
                toy_policy,
                rng.normal(size=6),
                bg,
                {"returns": [0, 1], "indicators": [2, 3], "risk_inputs": [4, 5]},
                asset_names=list(prices.columns),
            )
            explanation["model_kind"] = "synthetic_toy_not_ppo"
            bundle["explanations"][f"toy_policy/{decision}"] = explanation
            explanations.append(explanation)
        bundle["runs"]["toy_policy"] = {
            "strategy": "synthetic_toy_not_ppo",
            "decision_ids": decision_ids,
            "status": "explanation_only_not_backtested",
        }
        render_explanations(explanations, output / "plots", "SYN00")
    write_json(output / "evaluation.json", bundle)
    return bundle
