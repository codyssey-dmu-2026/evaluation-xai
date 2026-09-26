"""Integration hooks; no fabricated or bundled PPO checkpoint."""

import numpy as np

from .validation import DataContractError


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
