"""Validated settings shared by metrics, simulation and Walk-Forward runners."""

from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class Fold(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    train_start: date
    train_end: date
    test_start: date
    test_end: date

    @model_validator(mode="after")
    def ordered(self):
        if not self.train_start <= self.train_end < self.test_start <= self.test_end:
            raise ValueError("fold dates must be chronological and non-overlapping")
        return self


class SimulationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    initial_equity: PositiveFloat = 100_000
    commission_rate: Annotated[float, Field(ge=0, lt=1)] = 0.00015
    slippage_rate: Annotated[float, Field(ge=0, lt=1)] = 0.0005
    max_asset_weight: Annotated[float, Field(gt=0, le=1)] = 0.4
    drawdown_threshold: Annotated[float, Field(gt=0, lt=1)] = 0.15
    apply_guard: bool = True
    liquidate_at_end: bool = True

    @model_validator(mode="after")
    def cost_below_one(self):
        if self.commission_rate + self.slippage_rate >= 1:
            raise ValueError("combined cost rate must be less than one")
        return self


def annual_folds(first_test_year: int = 2022, last_test_year: int = 2025):
    if last_test_year < first_test_year:
        raise ValueError("last test year precedes first test year")
    return [
        Fold(
            id=f"WF{year}",
            train_start=date(year - 4, 1, 1),
            train_end=date(year - 1, 12, 31),
            test_start=date(year, 1, 1),
            test_end=date(year, 12, 31),
        )
        for year in range(first_test_year, last_test_year + 1)
    ]
