"""Read data-pipeline's actual processed format without fitting or downloading.

This verifies file/return consistency, not common-currency valuation or point-in-
time correctness. Those require the upstream FX/calendar/availability contract.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import read_json
from .validation import DataContractError, validate_index, validate_prices


@dataclass
class ProcessedWindow:
    prices: pd.DataFrame | None
    returns: pd.DataFrame
    features: pd.DataFrame
    market_open: pd.DataFrame
    metadata: dict
    asset_order: list[str]
    verification: dict

    @property
    def train_mask(self):
        return self.features.index < pd.Timestamp(self.metadata["train_end"])


def load_processed_window(directory, window=1, *, require_prices=True):
    """Price validation is strict by default; explicit replay mode permits no prices.

    Replay mode still requires raw-return/normalized-feature consistency and,
    when supplied, valid snapshot checksums. It never reconstructs fake prices.
    """
    root = Path(directory).resolve()
    manifest = read_json(root / "walk_forward_manifest.json")
    if manifest.get("is_dummy", False):
        raise DataContractError("real loader refuses is_dummy input")
    matches = [w for w in manifest["windows"] if w["window"] == window]
    if len(matches) != 1:
        raise DataContractError("manifest must contain exactly one requested window")
    meta = matches[0]
    feature_path = (root / meta["path"]).resolve()
    if not feature_path.is_relative_to(root):
        raise DataContractError("manifest feature path escapes processed directory")
    features = pd.read_parquet(feature_path)
    returns = pd.read_parquet(root / "returns.parquet")
    price_path = root / "prices.parquet"
    raw_prices = (
        pd.read_parquet(price_path) if require_prices or price_path.is_file() else None
    )
    frames = {"features": features, "returns": returns}
    if raw_prices is not None:
        frames["prices"] = raw_prices
    files = [
        feature_path,
        root / "returns.parquet",
        root / "walk_forward_manifest.json",
    ]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    checksum_path = root / "SHA256.json"
    checksum_status = "not_provided"
    if checksum_path.is_file():
        declared = read_json(checksum_path)
        for name, actual in hashes.items():
            expected_hash = declared.get(name)
            if (
                not isinstance(expected_hash, str)
                or len(expected_hash) not in (16, 64)
                or not actual.startswith(expected_hash.lower())
            ):
                raise DataContractError(
                    f"snapshot checksum mismatch or missing: {name}"
                )
        checksum_status = "matches_supplied_sha256_full_or_16char_prefix"
    for name, frame in frames.items():
        validate_index(frame.index, name)
        if frame.index.tz is not None:
            raise DataContractError(
                "processed daily dates must use producer-local naive labels"
            )
        if (
            not frame.columns.is_unique
            or not np.isfinite(frame.to_numpy(dtype=float)).all()
        ):
            raise DataContractError(f"{name}: duplicate columns or nonfinite values")
    if not all(isinstance(c, str) and c.endswith("_return") for c in returns.columns):
        raise DataContractError(
            "returns.parquet requires unnormalized *_return columns"
        )
    assets = [c.removesuffix("_return") for c in returns.columns]
    if len(assets) < 10:
        raise DataContractError("at least ten assets required")
    columns = meta["columns"]
    flags = meta.get("passthrough_columns", [])
    if list(features.columns) != columns + flags:
        raise DataContractError("features do not match manifest column order")
    expected_flags = [f"{a}_is_market_open" for a in assets]
    expected_features = {f"{a}_{f}" for a in assets for f in ("return", "rsi", "macd")}
    if set(columns) != expected_features or set(flags) != set(expected_flags):
        raise DataContractError("unexpected feature or market-open schema")
    if not np.isin(features[flags].to_numpy(), [0, 1]).all():
        raise DataContractError("market-open flags must be binary")
    for field in ("mean", "std", "lower", "upper"):
        if (
            set(meta[field]) != set(columns)
            or not np.isfinite(list(meta[field].values())).all()
        ):
            raise DataContractError("scaler metadata missing or nonfinite")
    if any(v <= 0 for v in meta["std"].values()):
        raise DataContractError("scaler standard deviations must be positive")
    if any(meta["lower"][c] > meta["upper"][c] for c in columns):
        raise DataContractError("invalid winsorization bounds")
    start, split, end = (
        pd.Timestamp(meta[k]) for k in ("train_start", "train_end", "test_end")
    )
    if not start < split < end:
        raise DataContractError("invalid train/test boundaries")
    expected = returns.index[(returns.index >= start) & (returns.index < end)]
    if not features.index.equals(expected):
        raise DataContractError("feature dates do not exactly cover declared window")
    if not (features.index < split).any() or not (features.index >= split).any():
        raise DataContractError("empty train or test partition")
    return_columns = list(returns.columns)
    clipped = returns.loc[features.index].clip(
        lower=pd.Series(meta["lower"])[return_columns],
        upper=pd.Series(meta["upper"])[return_columns],
        axis=1,
    )
    normalized = (clipped - pd.Series(meta["mean"])[return_columns]) / pd.Series(
        meta["std"]
    )[return_columns]
    if not np.allclose(
        normalized.to_numpy(),
        features[return_columns].to_numpy(),
        atol=1e-8,
        rtol=1e-6,
    ):
        raise DataContractError("raw returns and normalized return features disagree")
    prices = None
    if raw_prices is not None:
        if not set(assets + flags).issubset(raw_prices.columns):
            raise DataContractError("prices missing assets or market-open flags")
        prices = raw_prices[assets]
        validate_prices(prices, min_assets=10)
        if not np.isin(raw_prices[flags].to_numpy(), [0, 1]).all():
            raise DataContractError("price market-open flags must be binary")
        if len(returns.index.difference(prices.index)):
            raise DataContractError("prices missing return dates")
        computed = np.log(prices / prices.shift(1)).reindex(returns.index)
        if not np.isfinite(computed.to_numpy()).all() or not np.allclose(
            computed.to_numpy(), returns.to_numpy(), atol=1e-8, rtol=1e-6
        ):
            raise DataContractError("returns are not raw price-derived log returns")
        if not np.array_equal(features[flags], raw_prices.loc[features.index, flags]):
            raise DataContractError("feature and price market-open flags differ")
    return ProcessedWindow(
        prices,
        returns.loc[features.index],
        features,
        features[flags],
        meta,
        assets,
        {
            "price_return_consistency": (
                "verified" if prices is not None else "not_checked_prices_absent"
            ),
            "return_feature_consistency": "verified",
            "snapshot_checksums": checksum_status,
            "file_sha256": hashes,
            "scope": "file_consistency_not_market_provenance_or_fx_certification",
        },
    )
