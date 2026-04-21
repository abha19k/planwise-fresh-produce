from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from core.db import ENGINE, get_engine, _qualified
from core.config import (
    DEFAULT_SCHEMA,
    CLEANSED_HISTORY_TABLE,
    CLASSIFIED_FE_TABLE,
)
from models.schemas import (
    ClassifyComputeRequest,
    ClassifySaveRequest,
)
from services.classify_service import (
    _LAST_CLASSIFY_COMPUTED,
    _ensure_classified_fe_table,
)
from services.data_service import (
    _ensure_cleansed_history_table,
    _period_ui_from_slug,
)

router = APIRouter()


def _ensure_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()


@router.post("/api/classify/compute")
def api_classify_compute(
    body: ClassifyComputeRequest,
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    period = (body.period or "").strip().lower()
    if period not in ("daily", "weekly", "monthly"):
        raise HTTPException(status_code=400, detail="period must be one of: daily, weekly, monthly")

    sid = int(getattr(body, "scenario_id", 1) or 1)

    ts = datetime.now(timezone.utc).isoformat()
    _LAST_CLASSIFY_COMPUTED[f"{db_schema}:{sid}:{period}"] = ts

    return {"ok": True, "period": period, "scenario_id": sid, "computed_at": ts}


@router.get("/api/classify/results")
def api_classify_results(
    period: str = Query(..., description="daily|weekly|monthly"),
    scenario_id: int = Query(1, ge=1),
    include_inactive: bool = Query(True),
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    _ensure_engine()

    p = (period or "").strip().lower()
    if p not in ("daily", "weekly", "monthly"):
        raise HTTPException(status_code=400, detail="period must be one of: daily, weekly, monthly")

    period_ui = _period_ui_from_slug(p)
    sid = int(scenario_id)

    table_qual = _qualified(db_schema, CLEANSED_HISTORY_TABLE)

    sql = text(f"""
      SELECT DISTINCT
        "ProductID"  AS "ProductID",
        "ChannelID"  AS "ChannelID",
        "LocationID" AS "LocationID"
      FROM {table_qual}
      WHERE "scenario_id" = :scenario_id
        AND "Period" = :period_ui
      ORDER BY 1,2,3
      LIMIT 50000;
    """)

    computed_at = _LAST_CLASSIFY_COMPUTED.get(f"{db_schema}:{sid}:{p}") \
                  or datetime.now(timezone.utc).isoformat()

    try:
        with ENGINE.begin() as conn:
            keys = conn.execute(sql, {
                "scenario_id": sid,
                "period_ui": period_ui
            }).mappings().all()

        out = []
        for k in keys:
            out.append({
                "ProductID": k["ProductID"],
                "ChannelID": k["ChannelID"],
                "LocationID": k["LocationID"],
                "Period": p,
                "Label": "Active",
                "Score": 1.0,
                "IsActive": True,
                "ComputedAt": computed_at,
                "scenario_id": sid,
            })
        return out

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/classify/results failed: {e}")


@router.post("/api/classify/save")
def api_classify_save(
    body: ClassifySaveRequest,
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    _ensure_engine()

    _ensure_classified_fe_table(db_schema)

    period = (body.period or "").strip().lower()
    if period not in ("daily", "weekly", "monthly"):
        raise HTTPException(status_code=400, detail="period must be one of: daily, weekly, monthly")

    period_ui = _period_ui_from_slug(period)
    sid = int(getattr(body, "scenario_id", 1) or 1)

    rows = body.rows or []
    if not rows:
        return {"ok": True, "count": 0, "scenario_id": sid}

    qual = _qualified(db_schema, CLASSIFIED_FE_TABLE)

    sql = text(f"""
      INSERT INTO {qual}
        ("scenario_id","ProductID","ChannelID","LocationID","Period","ADI","CV2","Category","Algorithm","CreatedAt","UpdatedAt")
      VALUES
        (:scenario_id,:ProductID,:ChannelID,:LocationID,:Period,:ADI,:CV2,:Category,:Algorithm,
         COALESCE(CAST(:CreatedAt AS timestamptz), now()),
         now())
      ON CONFLICT ("scenario_id","ProductID","ChannelID","LocationID","Period")
      DO UPDATE SET
        "ADI"       = EXCLUDED."ADI",
        "CV2"       = EXCLUDED."CV2",
        "Category"  = EXCLUDED."Category",
        "Algorithm" = EXCLUDED."Algorithm",
        "UpdatedAt" = now();
    """)

    payload: List[Dict[str, Any]] = []
    for r in rows:
        payload.append({
            "scenario_id": sid,
            "ProductID": (r.ProductID or "").strip(),
            "ChannelID": (r.ChannelID or "").strip(),
            "LocationID": (r.LocationID or "").strip(),
            "Period": period_ui,
            "ADI": r.ADI,
            "CV2": r.CV2,
            "Category": (r.Category or "").strip(),
            "Algorithm": (r.Algorithm or "").strip(),
            "CreatedAt": (r.CreatedAt or None),
        })

    deduped = {}
    for p in payload:
        k = (
            p["scenario_id"],
            p["ProductID"],
            p["ChannelID"],
            p["LocationID"],
            p["Period"],
        )
        deduped[k] = p
    payload = list(deduped.values())

    try:
        with ENGINE.begin() as conn:
            chunk = 5000
            for i in range(0, len(payload), chunk):
                conn.execute(sql, payload[i:i+chunk])
        return {"ok": True, "count": len(payload), "scenario_id": sid}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/classify/save failed: {e}")


@router.get("/api/classify/saved")
def api_classify_saved(
    period: str = Query(..., description="daily|weekly|monthly"),
    scenario_id: int = Query(1, ge=1),
    include_inactive: bool = Query(False, description="if false, only rows that are active in scenario cleansed history"),
    limit: int = Query(20000, ge=1, le=50000),
    offset: int = Query(0, ge=0),
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    _ensure_engine()

    _ensure_classified_fe_table(db_schema)
    _ensure_cleansed_history_table(db_schema)

    p = (period or "").strip().lower()
    if p not in ("daily", "weekly", "monthly"):
        raise HTTPException(status_code=400, detail="period must be one of: daily, weekly, monthly")

    period_ui = _period_ui_from_slug(p)
    sid = int(scenario_id)

    cls_qual = _qualified(db_schema, CLASSIFIED_FE_TABLE)
    hist_qual = _qualified(db_schema, CLEANSED_HISTORY_TABLE)

    sql = text(f"""
      WITH active_keys AS (
        SELECT DISTINCT "ProductID","ChannelID","LocationID"
        FROM {hist_qual}
        WHERE "scenario_id" = :scenario_id
          AND "Period" = :period_ui
      )
      SELECT
        c."scenario_id",
        c."ProductID",
        c."ChannelID",
        c."LocationID",
        c."Period",
        c."ADI",
        c."CV2",
        c."Category",
        c."Algorithm",
        c."CreatedAt",
        c."UpdatedAt",
        (ak."ProductID" IS NOT NULL) AS "IsActive"
      FROM {cls_qual} c
      LEFT JOIN active_keys ak
        ON ak."ProductID" = c."ProductID"
       AND ak."ChannelID" = c."ChannelID"
       AND ak."LocationID" = c."LocationID"
      WHERE c."scenario_id" = :scenario_id
        AND c."Period" = :period_ui
        AND (:include_inactive = TRUE OR ak."ProductID" IS NOT NULL)
      ORDER BY c."UpdatedAt" DESC, c."ProductID", c."ChannelID", c."LocationID"
      LIMIT :limit OFFSET :offset;
    """)

    try:
        with ENGINE.begin() as conn:
            rows = conn.execute(sql, {
                "scenario_id": sid,
                "period_ui": period_ui,
                "include_inactive": include_inactive,
                "limit": limit,
                "offset": offset
            }).mappings().all()
        return [dict(r) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/classify/saved failed: {e}")