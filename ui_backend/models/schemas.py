from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RunOneDBRequest(BaseModel):
    db_schema: str = Field(default="planwise_fresh_produce")
    scenario_id: int = Field(default=1, ge=1)

    level: Optional[str] = Field(default=None, description="e.g. 111 / 121 / 221")
    period: str = Field(..., description="Daily / Weekly / Monthly")
    horizon: int = Field(..., ge=1)

    history_table: Optional[str] = Field(
        default=None,
        description="Explicit table name (without schema). If omitted, we build history_{level}_{periodlower}.",
    )

    weather_table: str = Field(default="weather_daily")
    promo_table: str = Field(default="promotions")
    tag: Optional[str] = Field(default=None, description="Output tag used in filenames")
    save_to_db: bool = True


class RunAllDBRequest(BaseModel):
    db_schema: str = Field(default="planwise_fresh_produce")
    scenario_id: int = Field(default=1, ge=1)
    weather_table: str = Field(default="weather_daily")
    promo_table: str = Field(default="promotions")


class KeysRequest(BaseModel):
    keys: List[Dict[str, str]]


class CleanseProfileIn(BaseModel):
    name: str
    config: dict


class CleansedIngestRequest(BaseModel):
    period: str
    rows: List[Dict[str, object]]
    scenario_id: Optional[int] = 0


class SavedSearchIn(BaseModel):
    name: str
    query: str


class ClassifyComputeRequest(BaseModel):
    period: str
    scenario_id: Optional[int] = 1
    lookback_buckets: Optional[int] = None
    min_sum: Optional[float] = None


class ClassifySaveRow(BaseModel):
    ProductID: str
    ChannelID: str
    LocationID: str
    ADI: Optional[float] = None
    CV2: Optional[float] = None
    Category: str
    Algorithm: str
    CreatedAt: Optional[str] = None


class ClassifySaveRequest(BaseModel):
    period: str
    scenario_id: Optional[int] = 1
    rows: List[ClassifySaveRow] = []


class ScenarioOut(BaseModel):
    scenario_id: int
    name: str
    parent_scenario_id: Optional[int] = None
    is_base: bool
    created_at: Optional[str] = None
    created_by: Optional[str] = None
    status: str


class ScenarioCopyIn(BaseModel):
    name: str
    created_by: Optional[str] = None


class OverrideUpsertIn(BaseModel):
    table_name: str
    pk: Dict[str, Any]
    row: Optional[Dict[str, Any]] = None
    is_deleted: bool = False
    updated_by: Optional[str] = None


class KpiRequest(BaseModel):
    ProductID: str
    ChannelID: str
    LocationID: str