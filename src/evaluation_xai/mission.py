"""Three independently reported mission tests from explicit monthly run samples."""

from pathlib import Path

import pandas as pd

from .artifacts import write_json
from .statistics import aggregate_seeds, mission_anova, normalize_monthly_samples
from .validation import DataContractError


def analyze_monthly_samples(samples, *, expected_seeds=(42, 43, 44)):
    """Columns: month,strategy,seed,return; optional externally verified regime.

    Names: PPO-simple/PPO-sharpe/PPO-mdd_l1.0/MVO/EqualWeight.
    Regimes: up/down/sideways, assigned BEFORE the evaluated month.
    """
    samples = normalize_monthly_samples(samples)
    averaged = aggregate_seeds(samples, expected_seeds=expected_seeds)
    reward_names = ["PPO-simple", "PPO-sharpe", "PPO-mdd_l1.0"]
    strategy_names = ["PPO-mdd_l1.0", "MVO", "EqualWeight"]

    def comparable(names):
        table = averaged[averaged.strategy.isin(names)]
        months = [set(table.loc[table.strategy == name, "month"]) for name in names]
        return table if all(m == months[0] for m in months) else None

    rewards = comparable(reward_names)
    strategies = comparable(strategy_names)
    regimes = None
    regime_error = None
    if "regime" in samples and strategies is not None:
        lookup = samples[["month", "regime"]].drop_duplicates()
        if (
            lookup.month.duplicated().any()
            or not lookup.regime.isin(["up", "down", "sideways"]).all()
        ):
            regime_error = "each month must have one valid preassigned regime"
        else:
            regimes = strategies.merge(
                lookup, on="month", how="left", validate="many_to_one"
            )
    result = mission_anova(rewards, strategies, regimes)
    if regime_error is not None:
        # mission_anova(None) already withholds the incomplete Holm family.
        result["tests"]["strategy_by_regime"] = {
            "status": "data_invalid",
            "reason": regime_error,
        }
    result["sample_definition"] = "monthly_net_simple_return_seed_mean"
    result["expected_seeds"] = list(expected_seeds)
    result["warning"] = (
        "Exploratory; matching months and supplied regimes "
        "do not prove point-in-time correctness."
    )
    return result


def run_mission_file(input_path, output_path):
    destination = Path(output_path)
    if destination.exists():
        raise DataContractError("refusing to overwrite statistics output")
    path = Path(input_path)
    samples = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    result = analyze_monthly_samples(samples)
    write_json(destination, result)
    return result
