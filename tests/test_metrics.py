import numpy as np
import pandas as pd
import pytest

from evaluation_xai.metrics import METRIC_NAMES, compute_metrics
from evaluation_xai.validation import DataContractError, validate_ledger


def calculate(ledger, benchmark=None, rf=None):
    return compute_metrics(
        ledger,
        pd.Series(
            (
                benchmark
                if benchmark is not None
                else np.linspace(-0.01, 0.02, len(ledger))
            ),
            index=ledger.index,
        ),
        pd.Series(0.0 if rf is None else rf, index=ledger.index),
        period_start=ledger.index[0] - pd.Timedelta(days=1),
    )["metrics"]


def test_twelve_metrics(ledger_factory):
    values = calculate(ledger_factory([0.1, -0.1, 0.02]))
    assert set(values) == set(METRIC_NAMES)
    assert values["cumulative_return"]["value"] == pytest.approx(1.1 * 0.9 * 1.02 - 1)
    assert values["mdd"]["value"] == pytest.approx(0.1)


def test_mdd_includes_initial_equity(ledger_factory):
    assert calculate(ledger_factory([-0.25, 0.1]))["mdd"]["value"] == 0.25


def test_alpha_beta_known_relation(ledger_factory):
    bench = np.array([-0.01, 0.02, -0.005, 0.015])
    rf = np.full(4, 0.0001)
    r = rf + 0.0002 + 1.5 * (bench - rf)
    values = calculate(ledger_factory(r), bench, rf)
    assert values["alpha"]["value"] == pytest.approx(0.0002 * 252)
    assert values["beta"]["value"] == pytest.approx(1.5)


def test_var_cvar(ledger_factory):
    r = [-0.1, -0.05, 0.0, 0.02]
    values = calculate(ledger_factory(r))
    assert values["var_95"]["value"] == pytest.approx(np.quantile(-np.array(r), 0.95))
    assert values["cvar_95"]["value"] == pytest.approx(0.1)


def test_zero_denominators_are_not_fake_zero(ledger_factory):
    values = calculate(ledger_factory([0, 0, 0]), [0, 0, 0])
    for name in ["sharpe", "sortino", "calmar", "alpha", "beta", "information_ratio"]:
        assert values[name]["value"] is None
        assert values[name]["status"] == "undefined"


def test_sample_volatility(ledger_factory):
    r = np.array([0.01, -0.03, 0.02])
    assert calculate(ledger_factory(r))["annualized_volatility"][
        "value"
    ] == pytest.approx(np.std(r, ddof=1) * np.sqrt(252))


def test_single_sample(ledger_factory):
    assert (
        calculate(ledger_factory([0.01]))["sharpe"]["reason"] == "insufficient_sample"
    )


def test_wrong_dates_rejected(ledger_factory):
    ledger = ledger_factory([0.01, 0.02])
    with pytest.raises(DataContractError, match="dates must match"):
        compute_metrics(
            ledger,
            pd.Series(0.0, index=ledger.index + pd.Timedelta(days=1)),
            pd.Series(0.0, index=ledger.index),
            period_start="2024-01-01",
        )


@pytest.mark.parametrize(
    "column,value",
    [
        ("net_return_simple", 0.5),
        ("net_return_log", 0.5),
        ("equity_start", -1),
        ("commission", -1),
        ("slippage", np.nan),
        ("equity_end", np.inf),
    ],
)
def test_corrupt_ledger_rejected(ledger_factory, column, value):
    ledger = ledger_factory([0.01, 0.02])
    ledger.loc[ledger.index[1], column] = value
    with pytest.raises(DataContractError):
        validate_ledger(ledger)


def test_no_silent_duplicate_dates(ledger_factory):
    ledger = ledger_factory([0.01, 0.02])
    ledger.index = pd.to_datetime(["2024-01-02", "2024-01-02"])
    with pytest.raises(DataContractError):
        validate_ledger(ledger)
