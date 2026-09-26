"""Strict JSON export. Synthetic and real evaluations must remain distinguishable."""

import json
from pathlib import Path

import pandas as pd

from .contracts import MarketDataManifest
from .validation import DataContractError, validate_prices


def write_json(path, value):
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content + "\n", encoding="utf-8")


def read_json(path):
    def reject(value):
        raise DataContractError(f"nonstandard JSON constant: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def validate_market_input(prices, manifest: MarketDataManifest):
    validate_prices(prices, min_assets=10 if manifest.data_mode == "real" else 1)
    if list(prices.columns) != manifest.asset_order:
        raise DataContractError("price columns must exactly match manifest asset order")
    if manifest.data_mode == "real":
        if prices.index[-1] < prices.index[0] + pd.DateOffset(years=5) - pd.Timedelta(
            days=7
        ):
            raise DataContractError("at least five years of real prices required")
    # Manifest is producer-supplied provenance, not proof of economic correctness.
