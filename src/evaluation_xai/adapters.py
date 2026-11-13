"""Integration hooks; no fabricated or bundled PPO checkpoint."""

import numpy as np

from .validation import DataContractError


def rl_observation_groups(asset_order, feature_columns, *, lookback=20):
    """Match rl-engine PortfolioEnv: time-major returns, static fields, weights.

    Input columns must be the actual ordered features_window file columns, not
    alphabetically re-sorted names. This constructs a layout, not a fake rollout.
    """
    assets, columns = list(asset_order), list(feature_columns)
    if (
        not assets
        or len(set(assets)) != len(assets)
        or len(set(columns)) != len(columns)
        or lookback < 1
    ):
        raise DataContractError("invalid observation schema")
    returns = [f"{asset}_return" for asset in assets]
    if set(c for c in columns if c.endswith("_return")) != set(returns):
        raise DataContractError("return feature columns must match asset order")
    static = [c for c in columns if not c.endswith("_return")]
    groups = {
        f"{asset}:returns": [day * len(assets) + i for day in range(lookback)]
        for i, asset in enumerate(assets)
    }
    offset = lookback * len(assets)
    for i, column in enumerate(static):
        groups[f"static:{column}"] = [offset + i]
    groups["portfolio_weights"] = list(
        range(offset + len(static), offset + len(static) + len(assets))
    )
    return groups


def sb3_weight_predictor(
    model, *, observation_shape, action_to_weights, preprocess=None
):
    """Use the RL team's fitted preprocessing and action-to-weight conversion.

    Explain deterministic policy weights, not subsequent Safe-Guard overrides.
    The caller loads the trusted checkpoint in its matching SB3/torch environment.
    """
    shape = tuple(observation_shape)
    if not shape or any(d < 1 for d in shape):
        raise DataContractError("positive observation dimensions required")

    def predict(batch):
        array = np.asarray(batch, dtype=float)
        if array.ndim != 2 or array.shape[1] != np.prod(shape):
            raise DataContractError("flattened observation shape mismatch")
        observation = array.reshape((len(array), *shape))
        if preprocess is not None:
            observation = preprocess(observation)
        actions, _state = model.predict(observation, deterministic=True)
        return np.asarray(action_to_weights(actions), dtype=float)

    return predict
