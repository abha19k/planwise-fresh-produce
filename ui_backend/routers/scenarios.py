from __future__ import annotations

import math
import json
import numpy as np

from datetime import datetime, date

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy import text
from services.auth_service import require_roles
from core.db import ENGINE, get_engine, _qualified
from core.config import DEFAULT_SCHEMA, SCENARIO_TABLE, SCENARIO_OVERRIDE_TABLE
from services.audit_service import write_audit_log

from models.schemas import (
    ScenarioCopyIn,
    OverrideUpsertIn,
)

from services.scenario_service import (
    _ensure_scenario_tables,
)

router = APIRouter()


def _ensure_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()


@router.get("/api/scenarios")
def api_scenarios(
    db_schema: str = Query(DEFAULT_SCHEMA),
    current_user=Depends(require_roles("admin", "planner", "viewer")),
):
    _ensure_engine()

    _ensure_scenario_tables(db_schema)
    qual = _qualified(db_schema, SCENARIO_TABLE)

    sql = text(f"""
      SELECT scenario_id, name, parent_scenario_id, is_base,
             created_at, created_by, status
      FROM {qual}
      WHERE status = 'active'
      ORDER BY is_base DESC, created_at DESC, scenario_id DESC;
    """)

    try:
        with ENGINE.begin() as conn:
            rows = conn.execute(sql).mappings().all()

        cleaned = []

        for row in rows:
            item = dict(row)

            for k, v in item.items():

                if isinstance(v, np.generic):
                    v = v.item()

                if isinstance(v, (datetime, date)):
                    v = v.isoformat(sep=" ")

                if isinstance(v, float):
                    if math.isnan(v) or math.isinf(v):
                        v = None

                item[k] = v

            if item.get("scenario_id") is not None:
                item["scenario_id"] = int(item["scenario_id"])

            if item.get("parent_scenario_id") is not None:
                item["parent_scenario_id"] = int(item["parent_scenario_id"])

            cleaned.append(item)

        return cleaned

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/scenarios failed: {e}")


@router.post("/api/scenarios/{scenario_id}/copy")
def api_copy_scenario(
    scenario_id: int,
    body: ScenarioCopyIn,
    db_schema: str = Query(DEFAULT_SCHEMA),
    current_user=Depends(require_roles("admin", "planner")),
):
    _ensure_engine()

    _ensure_scenario_tables(db_schema)

    name = (body.name or "").strip()

    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    qual = _qualified(db_schema, SCENARIO_TABLE)

    chk = text(f"""
        SELECT 1
        FROM {qual}
        WHERE scenario_id = :sid;
    """)

    with ENGINE.begin() as conn:
        ok = conn.execute(chk, {"sid": int(scenario_id)}).scalar()

    if not ok:
        raise HTTPException(status_code=404, detail=f"scenario_id {scenario_id} not found")

    sql = text(f"""
      INSERT INTO {qual}
        (name, parent_scenario_id, is_base, created_by)
      VALUES
        (:name, :parent, false, :created_by)
      RETURNING
        scenario_id,
        name,
        parent_scenario_id,
        is_base,
        created_at::text AS created_at,
        created_by,
        status;
    """)
   

    try:
        with ENGINE.begin() as conn:
            row = conn.execute(sql, {
                "name": name,
                "parent": int(scenario_id),
                "created_by": body.created_by
            }).mappings().first()

        item = dict(row)

        for k, v in item.items():

            if isinstance(v, np.generic):
                v = v.item()

            if isinstance(v, (datetime, date)):
                v = v.isoformat(sep=" ")

            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    v = None

            item[k] = v

        if item.get("scenario_id") is not None:
            item["scenario_id"] = int(item["scenario_id"])

        if item.get("parent_scenario_id") is not None:
            item["parent_scenario_id"] = int(item["parent_scenario_id"])

        write_audit_log(
            action="scenario.copy",
            entity="scenario",
            user_id=str(current_user["id"]),
            entity_id=str(item["scenario_id"]),
            details={
                "name": item["name"],
                "parent_scenario_id": item["parent_scenario_id"],
            },
            db_schema=db_schema,
        ) 

        return item
    
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/scenarios/{scenario_id}/copy failed: {e}")
    
    

@router.post("/api/scenarios/{scenario_id}/override")
def api_upsert_override(
    scenario_id: int,
    body: OverrideUpsertIn,
    db_schema: str = Query(DEFAULT_SCHEMA),
    current_user=Depends(require_roles("admin", "planner")),
):
    _ensure_engine()

    _ensure_scenario_tables(db_schema)

    tname = (body.table_name or "").strip()

    if not tname:
        raise HTTPException(status_code=400, detail="table_name is required")

    if not isinstance(body.pk, dict) or not body.pk:
        raise HTTPException(status_code=400, detail="pk must be a non-empty object")

    pk_json = json.dumps(body.pk, sort_keys=True, ensure_ascii=False)
    row_json = None if body.row is None else json.dumps(body.row, sort_keys=True, ensure_ascii=False)

    qual = _qualified(db_schema, SCENARIO_OVERRIDE_TABLE)

    sql = text(f"""
      INSERT INTO {qual}
        (scenario_id, table_name, pk, row, is_deleted, updated_at, updated_by)
      VALUES
        (:scenario_id, :table_name, CAST(:pk AS jsonb), CAST(:row AS jsonb),
         :is_deleted, now(), :updated_by)

      ON CONFLICT (scenario_id, table_name, pk)
      DO UPDATE SET
        row = EXCLUDED.row,
        is_deleted = EXCLUDED.is_deleted,
        updated_at = now(),
        updated_by = EXCLUDED.updated_by;
    """)


    try:
        with ENGINE.begin() as conn:
            conn.execute(sql, {
                "scenario_id": int(scenario_id),
                "table_name": tname,
                "pk": pk_json,
                "row": row_json,
                "is_deleted": bool(body.is_deleted),
                "updated_by": body.updated_by
            })

        write_audit_log(
            action="scenario.override",
            entity="scenario_override",
            user_id=str(current_user["id"]),
            entity_id=str(scenario_id),
            details={
                "table_name": tname,
                "pk": body.pk,
                "is_deleted": body.is_deleted,
            },
            db_schema=db_schema,
        )

        return {"ok": True}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/scenarios/{scenario_id}/override failed: {e}")