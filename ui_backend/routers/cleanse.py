from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from core.db import ENGINE, get_engine, _qident, _qualified
from core.config import DEFAULT_SCHEMA, CLEANSED_HISTORY_TABLE
from models.schemas import CleanseProfileIn, CleansedIngestRequest
from services.data_service import (
    _ensure_cleansed_history_table,
    _period_ui_from_slug,
)

router = APIRouter()


def _ensure_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()


@router.get("/api/cleanse/profiles")
def list_cleanse_profiles(db_schema: str = Query(DEFAULT_SCHEMA)):
    _ensure_engine()

    ddl = f"""
    CREATE TABLE IF NOT EXISTS {_qident(db_schema)}.cleanse_profiles (
      id SERIAL PRIMARY KEY,
      name TEXT UNIQUE NOT NULL,
      config JSONB NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """

    sql = text(f"""
      SELECT id, name, config, created_at::text AS created_at
      FROM {_qident(db_schema)}.cleanse_profiles
      ORDER BY created_at DESC, id DESC;
    """)

    try:
        with ENGINE.begin() as conn:
            conn.execute(text(ddl))
            rows = conn.execute(sql).mappings().all()
        return [dict(r) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/cleanse/profiles failed: {e}")


@router.post("/api/cleanse/profiles")
def upsert_cleanse_profile(
    body: CleanseProfileIn,
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    _ensure_engine()

    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Profile name is required.")

    config_json = json.dumps(body.config or {}, ensure_ascii=False)

    ddl = f"""
    CREATE TABLE IF NOT EXISTS {_qident(db_schema)}.cleanse_profiles (
      id SERIAL PRIMARY KEY,
      name TEXT UNIQUE NOT NULL,
      config JSONB NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """

    upsert_sql = text(f"""
      INSERT INTO {_qident(db_schema)}.cleanse_profiles (name, config)
      VALUES (:name, CAST(:config AS jsonb))
      ON CONFLICT (name)
      DO UPDATE SET config = EXCLUDED.config;
    """)

    try:
        with ENGINE.begin() as conn:
            conn.execute(text(ddl))
            conn.execute(upsert_sql, {"name": name, "config": config_json})
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/cleanse/profiles save failed: {e}")


@router.post("/api/history/ingest-cleansed")
def api_ingest_cleansed_history(
    body: CleansedIngestRequest,
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    _ensure_engine()

    _ensure_cleansed_history_table(db_schema)

    period_ui = _period_ui_from_slug(body.period)
    rows = body.rows or []
    if not rows:
        return {"ok": True, "count": 0}

    sid = int(getattr(body, "scenario_id", 0) or 0)

    table_qual = _qualified(db_schema, CLEANSED_HISTORY_TABLE)

    sql = text(f"""
        INSERT INTO {table_qual}
           ("scenario_id","ProductID","ChannelID","LocationID","StartDate","EndDate","Period","Qty","NetPrice","ListPrice","Level")
        VALUES
           (:scenario_id, :ProductID, :ChannelID, :LocationID,
            CAST(:StartDate AS date), CAST(:EndDate AS date),
            :Period, :Qty, :NetPrice, :ListPrice, :Level)
       ON CONFLICT ("scenario_id","ProductID","ChannelID","LocationID","StartDate","Period")
       DO UPDATE SET
           "EndDate"    = EXCLUDED."EndDate",
           "Qty"        = EXCLUDED."Qty",
           "NetPrice"   = EXCLUDED."NetPrice",
           "ListPrice"  = EXCLUDED."ListPrice",
           "Level"      = EXCLUDED."Level";
    """)

    def _clean_date_str(x: object) -> Optional[str]:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        if s.lower() in ("none", "null", "undefined", "nan"):
            return None
        return s

    def _float_or_none(x: object):
        if x is None:
            return None
        if isinstance(x, str):
            s = x.strip()
            if not s or s.lower() in ("none", "null", "undefined", "nan"):
                return None
        try:
            return float(x)
        except Exception:
            return None

    payload = []
    for r in rows:
        start = _clean_date_str(r.get("StartDate"))
        end = _clean_date_str(r.get("EndDate"))

        if not start:
            continue

        if not end:
            end = start

        payload.append({
            "scenario_id": sid,
            "ProductID": str(r.get("ProductID", "")).strip(),
            "ChannelID": str(r.get("ChannelID", "")).strip(),
            "LocationID": str(r.get("LocationID", "")).strip(),
            "StartDate": start,
            "EndDate": end,
            "Period": period_ui,
            "Qty": float(r.get("Qty", 0) or 0),
            "NetPrice": _float_or_none(r.get("NetPrice")),
            "ListPrice": _float_or_none(r.get("ListPrice")),
            "Level": (str(r.get("Level", "")).strip() or None),
        })

    deduped = {}
    for p in payload:
        k = (
            p["scenario_id"],
            p["ProductID"],
            p["ChannelID"],
            p["LocationID"],
            p["StartDate"],
            p["Period"],
        )
        deduped[k] = p

    payload = list(deduped.values())

    if not payload:
        return {"ok": True, "count": 0}

    bad = [i for i, p in enumerate(payload) if not p["StartDate"] or not p["EndDate"]]
    if bad:
        raise HTTPException(status_code=400, detail=f"Missing StartDate/EndDate in rows at indexes: {bad[:20]}")

    try:
        with ENGINE.begin() as conn:
            chunk_size = 5000
            for i in range(0, len(payload), chunk_size):
                conn.execute(sql, payload[i:i + chunk_size])

        return {"ok": True, "count": len(payload), "scenario_id": sid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/history/ingest-cleansed failed: {e}")