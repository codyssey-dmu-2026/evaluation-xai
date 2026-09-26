"""Evaluation-only API. Reads computed artifacts, never invents model results."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
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

    return app


app = create_app()
