"""Cross-team metadata and risk-tag contracts. No data downloads are performed."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class RiskTag(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tag_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    risk_type: str = Field(min_length=1)
    severity: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    confidence: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    published_at: AwareDatetime
    available_at: AwareDatetime
    generated_at: AwareDatetime
    expires_at: AwareDatetime
    source_url: str = Field(pattern=r"^https?://")
    tagger_version: str
    mode: Literal["live", "historical_replay"] = "live"

    @model_validator(mode="after")
    def chronological(self):
        if self.generated_at < self.published_at:
            raise ValueError("tag generation cannot precede source publication")
        if not self.published_at <= self.available_at < self.expires_at:
            raise ValueError(
                "tag availability must follow publication and precede expiry"
            )
        if self.mode == "live" and self.generated_at > self.available_at:
            raise ValueError("live tag cannot be available before it is generated")
        return self

    def usable_at(self, decision_at: datetime) -> bool:
        if decision_at.tzinfo is None:
            raise ValueError("decision time must include timezone")
        return self.available_at <= decision_at < self.expires_at


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    decision_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    target_asset: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,30}$")


class MarketDataManifest(BaseModel):
    """An attestation to accompany unnormalized, common-currency prices."""

    model_config = ConfigDict(extra="forbid")
    dataset_id: str = Field(min_length=1)
    data_mode: Literal["real", "synthetic"]
    asset_order: list[str] = Field(min_length=1)
    source: str = Field(min_length=1)
    price_type: Literal["adjusted_close"]
    normalized: Literal[False]
    base_currency: str = Field(pattern=r"^[A-Z]{3}$")
    converted_to_base_currency: Literal[True]
    calendar_policy: str = Field(min_length=1)
    availability_policy: str = Field(min_length=1)
    missing_value_policy: str = Field(min_length=1)

    @model_validator(mode="after")
    def assets_unique(self):
        if len(set(self.asset_order)) != len(self.asset_order):
            raise ValueError("asset order contains duplicates")
        if self.data_mode == "real" and len(self.asset_order) < 10:
            raise ValueError("final real-data evaluation requires at least 10 assets")
        return self
