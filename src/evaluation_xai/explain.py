"""Grouped SHAP for an explicitly supplied deterministic policy-weight function."""

from pathlib import Path

import numpy as np

from .validation import DataContractError


def explain_weights(
    predict_weights,
    observation,
    background,
    groups,
    *,
    asset_names,
    seed=11,
    permutations=10,
):
    """Explain pre-Safe-Guard policy weights, not trades or a causal effect.

    Each missing group is replaced with one complete training-background row.
    This is grouped interventional SHAP, not per-feature conditional SHAP.
    Keep coupled features (e.g. all portfolio weights) in the same group.
    """
    import shap

    x = np.asarray(observation, dtype=float)
    bg = np.asarray(background, dtype=float)
    if x.ndim != 1 or bg.ndim != 2 or bg.shape[1] != x.size or bg.shape[0] < 1:
        raise DataContractError("observation/background shape mismatch")
    if not np.isfinite(x).all() or not np.isfinite(bg).all():
        raise DataContractError("SHAP input must be finite")
    indices = [i for group in groups.values() for i in group]
    if (
        not groups
        or any(not group for group in groups.values())
        or sorted(indices) != list(range(x.size))
    ):
        raise DataContractError(
            "groups must partition every input dimension exactly once"
        )
    if len(set(asset_names)) != len(asset_names) or permutations < 1:
        raise DataContractError("invalid SHAP output labels or permutations")

    def predict(batch):
        out = np.asarray(predict_weights(batch), dtype=float)
        if out.shape != (len(batch), len(asset_names)) or not np.isfinite(out).all():
            raise DataContractError(
                "predict_weights must return finite [batch, asset] weights"
            )
        if (out < -1e-8).any() or not np.allclose(
            out.sum(axis=1), 1, atol=1e-6, rtol=0
        ):
            raise DataContractError("explain the normalized long-only policy weights")
        return out

    actual = predict(x[None])[0]
    if not np.allclose(actual, predict(x[None])[0], atol=1e-6, rtol=0):
        raise DataContractError("policy prediction must be deterministic")

    def coalition(masks):
        results = []
        for mask in masks:
            mixed = bg.copy()
            for enabled, cols in zip(mask, groups.values()):
                if enabled >= 0.5:
                    mixed[:, cols] = x[cols]
            results.append(predict(mixed).mean(axis=0))
        return np.asarray(results)

    names = list(groups)
    explainer = shap.PermutationExplainer(
        coalition, np.zeros((1, len(groups))), feature_names=names, seed=seed
    )
    explanation = explainer(
        np.ones((1, len(groups))), max_evals=permutations * (2 * len(groups) + 1)
    )
    contributions = explanation.values[0]
    base = np.asarray(explanation.base_values[0])
    error = np.max(np.abs(base + contributions.sum(axis=0) - actual))
    if error > 1e-3:
        raise DataContractError("SHAP additivity check failed")
    return {
        "method": "grouped_interventional_permutation_shap",
        "stage": "policy_weights_before_safeguard",
        "asset_names": list(asset_names),
        "group_names": names,
        "base_values": base.tolist(),
        "predicted_weights": actual.tolist(),
        "shap_values": contributions.tolist(),
        "additivity_error": float(error),
        "background_count": len(bg),
        "seed": seed,
        "interpretation": (
            "Model attribution relative to training background; not causality, "
            "policy quality, or an explanation of subsequent rule overrides."
        ),
    }


def render_explanations(explanations, output_directory, target_asset):
    """Summary importance over decisions and a Force Plot of the first decision."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    if not explanations:
        raise DataContractError("at least one explanation required")
    first = explanations[0]
    asset = first["asset_names"].index(target_asset)
    names = first["group_names"]
    if any(
        item["group_names"] != names or item["asset_names"] != first["asset_names"]
        for item in explanations
    ):
        raise DataContractError("inconsistent explanation dimensions")
    values = np.array([item["shap_values"] for item in explanations])[:, :, asset]
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(
        values,
        feature_names=names,
        plot_type="bar",
        show=False,
        rng=np.random.default_rng(11),
    )
    plt.title(f"Grouped SHAP — {target_asset}")
    plt.tight_layout()
    plt.savefig(output / "shap_summary.png", dpi=140)
    plt.close()
    force = shap.force_plot(first["base_values"][asset], values[0], feature_names=names)
    shap.save_html(str(output / "shap_force.html"), force)
