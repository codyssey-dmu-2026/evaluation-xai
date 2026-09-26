import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ledger_factory():
    def make(returns):
        r = np.asarray(returns)
        end = 100 * np.cumprod(1 + r)
        return pd.DataFrame(
            {
                "equity_start": np.r_[100, end[:-1]],
                "equity_end": end,
                "net_return_simple": r,
                "net_return_log": np.log1p(r),
                "commission": 0.0,
                "slippage": 0.0,
            },
            index=pd.date_range("2024-01-02", periods=len(r), name="date"),
        )

    return make


@pytest.fixture
def prices():
    from evaluation_xai.demo import synthetic_prices

    return synthetic_prices(start="2022-01-03", end="2024-12-31", assets=3)
