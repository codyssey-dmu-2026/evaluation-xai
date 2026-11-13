"""Evaluation-only API. Reads computed artifacts, never invents model results."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import DISCLAIMER
from .artifacts import read_json
from .contracts import ExplainRequest


class HealthResponse(BaseModel):
    status: str
    real_model_loaded: bool
    artifacts_available: bool
    service: str = "evaluation-xai"


def create_app(artifact_directory=None):
    directory = Path(
        artifact_directory or os.getenv("EVALUATION_ARTIFACT_DIR", "artifacts/demo")
    )
    app = FastAPI(
        title="Evaluation & XAI",
        version="0.1.0",
        description=(
            "Offline artifact API; /optimize and /research " "belong to other services."
        ),
    )

    def bundle():
        path = directory / "evaluation.json"
        if not path.is_file():
            raise HTTPException(
                503, "Run the offline demo or publish evaluation artifacts first."
            )
        try:
            return read_json(path)
        except (ValueError, OSError):
            raise HTTPException(
                503, "Evaluation artifact is invalid or unreadable."
            ) from None

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            status="ok",
            real_model_loaded=False,
            artifacts_available=(directory / "evaluation.json").is_file(),
        )

    @app.get("/backtest")
    def backtest(
        run_id: str | None = Query(default=None, pattern=r"^[A-Za-z0-9_-]{1,80}$")
    ):
        data = bundle()
        if run_id is None:
            return data
        if run_id not in data["runs"]:
            raise HTTPException(404, "Unknown evaluation run.")
        return {
            "data_mode": data["data_mode"],
            "run": data["runs"][run_id],
            "disclaimer": DISCLAIMER,
        }

    @app.post("/explain")
    def explain(request: ExplainRequest):
        data = bundle()
        run = data["runs"].get(request.run_id)
        if run is None:
            raise HTTPException(404, "Unknown evaluation run.")
        if request.decision_id not in run.get("decision_ids", []):
            raise HTTPException(404, "Unknown decision.")
        if data.get("data_mode") == "synthetic" and not request.allow_synthetic:
            raise HTTPException(
                409,
                "Synthetic demonstration, not a trading decision. "
                "Opt in with allow_synthetic=true.",
            )
        key = f"{request.run_id}/{request.decision_id}"
        explanation = data.get("explanations", {}).get(key)
        if explanation is None:
            raise HTTPException(
                409,
                "No computed SHAP artifact for this decision; coverage is incomplete.",
            )
        if request.target_asset not in explanation["asset_names"]:
            raise HTTPException(404, "Unknown target asset.")
        i = explanation["asset_names"].index(request.target_asset)
        return {
            "data_mode": data["data_mode"],
            "explanation_scope": (
                "synthetic_demonstration_not_trading_decision"
                if data["data_mode"] == "synthetic"
                else "policy_attribution"
            ),
            "run_id": request.run_id,
            "decision_id": request.decision_id,
            "target_asset": request.target_asset,
            "stage": explanation["stage"],
            "method": explanation["method"],
            "groups": explanation["group_names"],
            "base_value": explanation["base_values"][i],
            "predicted_weight": explanation["predicted_weights"][i],
            "contributions": [row[i] for row in explanation["shap_values"]],
            "interpretation": explanation["interpretation"],
            "disclaimer": DISCLAIMER,
        }

    @app.get("/explanations/{run_id}/{decision_id}/{target_asset}/plots/{kind}")
    def explanation_plot(
        run_id: str,
        decision_id: str,
        target_asset: str,
        kind: str,
        allow_synthetic: bool = False,
    ):
        data = bundle()
        if data.get("data_mode") == "synthetic" and not allow_synthetic:
            raise HTTPException(409, "Synthetic plot requires allow_synthetic=true.")
        if kind not in {"summary", "force"}:
            raise HTTPException(404, "Unknown plot kind.")
        key = f"{run_id}/{decision_id}/{target_asset}"
        relative = data.get("plot_artifacts", {}).get(key, {}).get(kind)
        if not isinstance(relative, str):
            raise HTTPException(404, "No computed plot for this decision and asset.")
        path = (directory / relative).resolve()
        suffix = ".png" if kind == "summary" else ".html"
        if not path.is_relative_to(directory.resolve()) or path.suffix != suffix:
            raise HTTPException(503, "Invalid plot artifact path.")
        if not path.is_file():
            raise HTTPException(404, "Plot artifact missing.")
        return FileResponse(
            path,
            media_type="image/png" if kind == "summary" else "text/html",
            headers={
                "X-Content-Type-Options": "nosniff",
                "X-Evaluation-Data-Mode": data["data_mode"],
                "Content-Security-Policy": (
                    "sandbox allow-scripts; default-src 'none'; "
                    "script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                    "img-src data:"
                ),
            },
        )

    return app


app = create_app()
