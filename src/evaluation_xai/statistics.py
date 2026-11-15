"""Exploratory ANOVA and paired block bootstrap; not IID claims about markets."""

import re

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
    results = {}
    for name, table, two_way in (
        ("reward_variants", rewards, False),
        ("strategies", strategies, False),
        ("strategy_by_regime", strategy_regime, True),
    ):
        factors = ["strategy", "regime"] if two_way else ["strategy"]
        if table is None or any(
            col not in table or table[col].nunique() != 3 for col in factors
        ):
            results[name] = {
                "status": "incomplete",
                "reason": "three levels per mission factor required",
            }
            continue
        try:
            results[name] = anova(table, two_way=two_way)
        except DataContractError as exc:
            results[name] = {"status": "data_invalid", "reason": str(exc)}
    if any(result["status"] != "ok" for result in results.values()):
        return {
            "status": "incomplete",
            "tests": results,
            "holm_status": "incomplete_family",
            "holm_family_size": 5,
        }
    effects = [effect for result in results.values() for effect in result["effects"]]
    adjusted = multipletests([e["p_value"] for e in effects], method="holm")[1]
    for effect, p_value in zip(effects, adjusted):
        effect["p_holm_family"] = float(p_value)
    return {
        "status": "ok",
        "tests": results,
        "holm_status": "complete",
        "holm_family_size": len(effects),
    }


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


def regimes_from_log_returns(benchmark, months, lookback=60, threshold=0.05):
    """Pre-month regime and audit trail using raw log returns, not invented prices.

    Observations follow the supplied union-market calendar, including zero returns
    on SPY-closed sessions. They are not certified US exchange trading days.
    """
    validate_index(benchmark.index, "benchmark log returns")
    if lookback < 1 or not np.isfinite(threshold) or threshold <= 0:
        raise DataContractError("invalid regime settings")
    if not np.isfinite(benchmark.to_numpy(dtype=float)).all():
        raise DataContractError("nonfinite benchmark log returns")
    rows = []
    for month in months:
        period = pd.Period(month, freq="M")
        history = benchmark.loc[benchmark.index < period.start_time].tail(lookback)
        if len(history) != lookback:
            raise DataContractError(f"insufficient pre-month history: {period}")
        change = float(np.expm1(history.sum()))
        if not np.isfinite(change) or change <= -1:
            raise DataContractError("invalid compounded benchmark return")
        rows.append(
            {
                "month": str(period),
                "regime": (
                    "up"
                    if change > threshold
                    else "down" if change < -threshold else "sideways"
                ),
                "trailing_return": change,
                "history_start": str(history.index[0].date()),
                "history_end": str(history.index[-1].date()),
                "observations": lookback,
            }
        )
    return pd.DataFrame(rows)


def normalize_monthly_samples(frame: pd.DataFrame):
    """Reject daily dates; canonicalize monthly Periods without regrouping returns."""
    required = {"month", "strategy", "seed", "return"}
    if not required.issubset(frame.columns) or frame.empty:
        raise DataContractError("seed table missing required fields")
    frame = frame.copy()
    months = []
    for month in frame["month"]:
        if isinstance(month, pd.Period) and month.freqstr == "M":
            months.append(str(month))
        elif isinstance(month, str) and re.fullmatch(
            r"[0-9]{4}-(0[1-9]|1[0-2])", month
        ):
            if month.startswith("0000"):
                raise DataContractError("month must be YYYY-MM or a monthly Period")
            months.append(month)
        else:
            raise DataContractError("month must be YYYY-MM or a monthly Period")
    frame["month"] = months
    if not frame.strategy.map(lambda s: isinstance(s, str) and bool(s.strip())).all():
        raise DataContractError("strategy must be a nonempty string")
    if not np.isfinite(frame["return"]).all() or (frame["return"] <= -1).any():
        raise DataContractError("invalid monthly returns")
    if frame.duplicated(["month", "strategy", "seed"]).any():
        raise DataContractError("duplicate month/strategy/seed sample")
    return frame


def aggregate_seeds(frame: pd.DataFrame, expected_seeds=(11, 22, 33)):
    """Only named deterministic baselines may omit seeds; PPO never may."""
    frame = normalize_monthly_samples(frame)
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise DataContractError("expected seeds must be nonempty and unique")
    rows = []
    for (month, strategy), group in frame.groupby(["month", "strategy"]):
        if strategy.casefold() in {"mvo", "ew", "equalweight", "equal_weight"}:
            if len(group) != 1 or not group.seed.isna().all():
                raise DataContractError(
                    "baseline must occur once per month without seed"
                )
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
