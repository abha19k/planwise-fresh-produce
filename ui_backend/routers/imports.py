from __future__ import annotations

from typing import Any, Dict, List
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
from sqlalchemy import text

from core.db import ENGINE, get_engine, _qualified
from core.config import DEFAULT_SCHEMA
from services.auth_service import require_roles
from services.audit_service import write_audit_log

router = APIRouter()


def _ensure_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()


def _clean_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "none", "null", "undefined"):
        return ""
    return s


def _to_float(x: Any):
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none", "null", "undefined"):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _to_date_str(x: Any) -> str:
    if x is None:
        return ""

    try:
        if isinstance(x, datetime):
            return x.date().isoformat()

        dt = pd.to_datetime(x, errors="coerce")
        if pd.isna(dt):
            return ""

        return dt.date().isoformat()
    except Exception:
        return ""


@router.post("/api/import/history")
def import_history_excel(
    file: UploadFile = File(...),
    period: str = Query(..., description="Daily | Weekly | Monthly"),
    db_schema: str = Query(DEFAULT_SCHEMA),
    current_user=Depends(require_roles("admin", "planner")),
):
    _ensure_engine()

    p = (period or "").strip()
    if p not in ("Daily", "Weekly", "Monthly"):
        raise HTTPException(status_code=400, detail="period must be Daily, Weekly, or Monthly")

    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only Excel files .xlsx/.xls are supported")

    try:
        df = pd.read_excel(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read Excel file: {e}")

    required_cols = [
        "ProductID",
        "ChannelID",
        "LocationID",
        "StartDate",
        "EndDate",
        "Qty",
        "Level",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    table_name = {
        "Daily": "history_daily",
        "Weekly": "history_weekly",
        "Monthly": "history_monthly",
    }[p]

    table_qual = _qualified(db_schema, table_name)

    payload: List[Dict[str, Any]] = []

    for _, r in df.iterrows():
        start = _to_date_str(r.get("StartDate"))
        end = _to_date_str(r.get("EndDate")) or start

        if not start:
            continue

        payload.append({
            "ProductID": _clean_str(r.get("ProductID")),
            "ChannelID": _clean_str(r.get("ChannelID")),
            "LocationID": _clean_str(r.get("LocationID")),
            "StartDate": start,
            "EndDate": end,
            "Period": p,
            "Qty": _to_float(r.get("Qty")) or 0,
            "NetPrice": _to_float(r.get("NetPrice")),
            "ListPrice": _to_float(r.get("ListPrice")),
            "Level": _clean_str(r.get("Level")),
            "Type": _clean_str(r.get("Type")) or "Normal-History",
        })

    if not payload:
        return {"ok": True, "rows_inserted": 0, "message": "No valid rows found"}

    sql = text(f"""
        INSERT INTO {table_qual}
          ("ProductID","ChannelID","LocationID","StartDate","EndDate","Period",
           "Qty","NetPrice","ListPrice","Level","Type")
        VALUES
          (:ProductID,:ChannelID,:LocationID,
           CAST(:StartDate AS date), CAST(:EndDate AS date),
           :Period,:Qty,:NetPrice,:ListPrice,:Level,:Type)
    """)

    try:
        with ENGINE.begin() as conn:
            chunk_size = 5000
            for i in range(0, len(payload), chunk_size):
                conn.execute(sql, payload[i:i + chunk_size])

        write_audit_log(
            action="history.import_excel",
            entity="history",
            user_id=str(current_user["id"]),
            entity_id=f"{table_name}",
            details={
                "filename": file.filename,
                "period": p,
                "rows_inserted": len(payload),
                "table": table_name,
            },
            db_schema=db_schema,
        )

        return {
            "ok": True,
            "period": p,
            "table": table_name,
            "rows_inserted": len(payload),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History import failed: {e}")