"""Exploratory ANOVA and paired block bootstrap; not IID claims about markets."""

import numpy as np
import pandas as pd
from scipy.stats import levene, shapiro
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

from .validation import DataContractError, validate_index


def mission_anova(rewards, strategies, strategy_regime):
    """Three mission tests plus Holm adjustment across five omnibus effects.

    Inputs are aligned monthly observations, already averaged across seeds.
    Tukey is conditional on raw p<.05 as specified by the mission.
    """
    required_tables = [
        (rewards, "strategy"),
        (strategies, "strategy"),
        (strategy_regime, "strategy"),
        (strategy_regime, "regime"),
    ]
    if any(
        column not in table or table[column].nunique() != 3
        for table, column in required_tables
    ):
        return {
            "status": "incomplete",
            "reason": "three levels per mission factor required",
        }
    results = {
        "reward_variants": anova(rewards),
        "strategies": anova(strategies),
        "strategy_by_regime": anova(strategy_regime, two_way=True),
    }
    if any(result["status"] != "ok" for result in results.values()):
        return {"status": "incomplete", "tests": results}
    effects = [effect for result in results.values() for effect in result["effects"]]
    adjusted = multipletests([e["p_value"] for e in effects], method="holm")[1]
    for effect, p_value in zip(effects, adjusted):
        effect["p_holm_family"] = float(p_value)
    return {"status": "ok", "tests": results, "holm_family_size": len(effects)}


def monthly_returns(daily: pd.Series) -> pd.Series:
    validate_index(daily.index, "daily returns")
    if not np.isfinite(daily).all() or (daily <= -1).any():
        raise DataContractError("daily returns must be finite simple returns > -1")
    return (1 + daily).groupby(daily.index.to_period("M")).prod() - 1


def market_regimes(benchmark_prices: pd.Series, months, lookback=60, threshold=0.05):
    """Classify each month from prices strictly before its first calendar day."""
    validate_index(benchmark_prices.index, "benchmark prices")
    if lookback < 1 or threshold <= 0:
        raise DataContractError("invalid regime settings")
    if not np.isfinite(benchmark_prices).all() or (benchmark_prices <= 0).any():
        raise DataContractError("invalid benchmark prices")
    result = {}
    for month in months:
        period = pd.Period(month, freq="M")
        history = benchmark_prices.loc[benchmark_prices.index < period.start_time]
        if len(history) < lookback + 1:
            raise DataContractError(f"insufficient pre-month history: {period}")
        change = history.iloc[-1] / history.iloc[-lookback - 1] - 1
        result[str(period)] = (
            "up"
            if change > threshold
            else "down" if change < -threshold else "sideways"
        )
    return result


def aggregate_seeds(frame: pd.DataFrame, expected_seeds=(11, 22, 33)):
    """Input: month, strategy, seed, return. Never replicate baseline samples."""
    required = {"month", "strategy", "seed", "return"}
    if not required.issubset(frame.columns) or frame.empty:
        raise DataContractError("seed table missing required fields")
    if not np.isfinite(frame["return"]).all() or (frame["return"] <= -1).any():
        raise DataContractError("invalid monthly returns")
    rows = []
    for (month, strategy), group in frame.groupby(["month", "strategy"]):
        if group.seed.isna().all():
            if len(group) != 1:
                raise DataContractError("baseline must occur once per month")
        elif group.seed.isna().any() or sorted(group.seed.tolist()) != sorted(
            expected_seeds
        ):
            raise DataContractError("every expected seed must occur exactly once")
        rows.append(
            {"month": str(month), "strategy": strategy, "value": group["return"].mean()}
        )
    return pd.DataFrame(rows)


def anova(frame, *, two_way=False, min_cell=6, min_group=12, alpha=0.05):
    """Columns: value, strategy; plus regime for strategy x regime analysis.

    Same-month observations and serial dependence violate simple IID assumptions.
    Results are descriptive/exploratory; pair them with blocked uncertainty estimates.
    """
    factors = ["strategy", "regime"] if two_way else ["strategy"]
    if not {"value", *factors}.issubset(frame.columns) or frame.empty:
        raise DataContractError("ANOVA input missing columns or observations")
    data = frame[["value", *factors]].copy()
    if data.isna().any().any() or not np.isfinite(data.value).all():
        raise DataContractError("ANOVA input contains missing or nonfinite values")
    levels = [sorted(data[f].unique()) for f in factors]
    if any(len(level) < 2 for level in levels):
        return {
            "status": "insufficient_data",
            "reason": "at least two levels per factor required",
        }
    if two_way:
        index = pd.MultiIndex.from_product(levels, names=factors)
        counts = data.groupby(factors).size().reindex(index, fill_value=0)
    else:
        counts = data.groupby("strategy").size()
    minimum = min_cell if two_way else min_group
    if counts.min() < minimum:
        return {
            "status": "insufficient_data",
            "reason": "sparse or empty groups",
            "minimum_count": int(counts.min()),
            "required_count": minimum,
        }
    formula = (
        "value ~ C(strategy, Sum) * C(regime, Sum)"
        if two_way
        else "value ~ C(strategy)"
    )
    model = ols(formula, data=data).fit()
    if np.sum(model.resid**2) < 1e-20:
        return {"status": "undefined", "reason": "zero_residual_variance"}
    table = anova_lm(model, typ=3 if two_way else 1)
    total_ss = float(np.sum((data.value - data.value.mean()) ** 2))
    effects = []
    for term, row in table.iterrows():
        if term in {"Intercept", "Residual"}:
            continue
        f, p = float(row["F"]), float(row["PR(>F)"])
        if not np.isfinite([f, p]).all():
            return {"status": "undefined", "reason": "nonfinite_anova"}
        effects.append(
            {
                "term": term,
                "f": f,
                "p_value": p,
                "eta_squared": float(row.sum_sq / total_ss),
            }
        )
    groups = [g.value.to_numpy() for _, g in data.groupby(factors)]
    lev = levene(*groups, center="median")
    normal = shapiro(model.resid[:5000])
    significant = any(e["p_value"] < alpha for e in effects)
    comparisons = []
    if significant:
        labels = data.strategy.astype(str)
        if two_way:
            labels = labels + " / " + data.regime.astype(str)
        tukey = pairwise_tukeyhsd(data.value.to_numpy(), labels.to_numpy(), alpha=alpha)
        for row in tukey.summary().data[1:]:
            comparisons.append(
                dict(
                    zip(
                        [
                            "group1",
                            "group2",
                            "mean_difference",
                            "p_adjusted",
                            "lower",
                            "upper",
                            "reject",
                        ],
                        [
                            str(row[0]),
                            str(row[1]),
                            float(row[2]),
                            float(row[3]),
                            float(row[4]),
                            float(row[5]),
                            bool(row[6]),
                        ],
                    )
                )
            )

    def finite_or_none(value):
        return float(value) if np.isfinite(value) else None

    return {
        "status": "ok",
        "interpretation": "exploratory_not_independent_market_samples",
        "anova_type": 3 if two_way else 1,
        "effects": effects,
        "eta_squared_definition": "effect_SS / centered_total_SS",
        "diagnostics": {
            "levene_p": finite_or_none(lev.pvalue),
            "shapiro_residual_p": finite_or_none(normal.pvalue),
        },
        "tukey_status": "performed" if significant else "not_required",
        "tukey_family": "all_strategy_regime_cells" if two_way else "all_groups",
        "tukey": comparisons,
        "warning": (
            "Repeated months and serial dependence are not corrected by "
            "ANOVA or Tukey; use paired block bootstrap as a supplementary check."
        ),
    }


def paired_block_bootstrap(
    left: pd.Series,
    right: pd.Series,
    *,
    block_length=3,
    repetitions=5000,
    seed=20260926,
):
    """CI for mean monthly difference; paired circular blocks within each year."""
    if not left.index.equals(right.index) or not isinstance(left.index, pd.PeriodIndex):
        raise DataContractError("bootstrap requires identical monthly PeriodIndex")
    if (
        left.index.freqstr != "M"
        or not left.index.is_unique
        or not left.index.is_monotonic_increasing
    ):
        raise DataContractError("bootstrap months must be unique, sorted and monthly")
    if len(left) < 12 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise DataContractError("bootstrap requires at least 12 finite paired months")
    expected = pd.period_range(left.index[0], left.index[-1], freq="M")
    if not left.index.equals(expected) or block_length < 1 or repetitions < 100:
        raise DataContractError("invalid bootstrap settings or missing months")
    delta = (left - right).to_numpy()
    rng = np.random.default_rng(seed)
    samples = np.zeros(repetitions)
    for year in sorted(set(left.index.year)):
        values = delta[left.index.year == year]
        n = len(values)
        if block_length > n:
            raise DataContractError("block length exceeds a year segment")
        starts = rng.integers(0, n, size=(repetitions, int(np.ceil(n / block_length))))
        positions = (starts[..., None] + np.arange(block_length)) % n
        draws = values[positions.reshape(repetitions, -1)[:, :n]]
        samples += draws.sum(axis=1)
    samples /= len(delta)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "mean_monthly_difference": float(delta.mean()),
        "ci_95": [float(lo), float(hi)],
        "block_length": block_length,
        "repetitions": repetitions,
        "seed": seed,
        "resampling": "paired_circular_blocks_within_year",
    }
