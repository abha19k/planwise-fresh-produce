from __future__ import annotations

from typing import Dict

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from core.db import ENGINE, get_engine, _qident, _qualified
from core.config import (
    DEFAULT_SCHEMA,
    PRODUCT_TABLE,
    CHANNEL_TABLE,
    LOCATION_TABLE,
    FORECASTELEMENT_TABLE,
    PRODUCT_COLS,
    CHANNEL_COLS,
    LOCATION_COLS,
    FORECASTELEMENT_COLS,
)
from services.data_service import _normalize_str_df

router = APIRouter()


def _ensure_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()


@router.get("/api/products")
def api_products(db_schema: str = Query(DEFAULT_SCHEMA)):
    _ensure_engine()

    qual = _qualified(db_schema, PRODUCT_TABLE)
    cols = ", ".join(_qident(c) for c in PRODUCT_COLS)
    sql = text(f"SELECT {cols} FROM {qual} ORDER BY {_qident('ProductID')};")

    try:
        df = pd.read_sql_query(sql, ENGINE)
        df = _normalize_str_df(df, PRODUCT_COLS)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/products failed: {e}")


@router.get("/api/channels")
def api_channels(db_schema: str = Query(DEFAULT_SCHEMA)):
    _ensure_engine()

    qual = _qualified(db_schema, CHANNEL_TABLE)
    cols = ", ".join(_qident(c) for c in CHANNEL_COLS)
    sql = text(f"SELECT {cols} FROM {qual} ORDER BY {_qident('Level')}, {_qident('ChannelID')};")

    try:
        df = pd.read_sql_query(sql, ENGINE)
        df = _normalize_str_df(df, CHANNEL_COLS)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/channels failed: {e}")


@router.get("/api/locations")
def api_locations(db_schema: str = Query(DEFAULT_SCHEMA)):
    _ensure_engine()

    qual = _qualified(db_schema, LOCATION_TABLE)
    cols = ", ".join(_qident(c) for c in LOCATION_COLS)
    sql = text(f"SELECT {cols} FROM {qual} ORDER BY {_qident('Level')}, {_qident('LocationID')};")

    try:
        df = pd.read_sql_query(sql, ENGINE)
        df = _normalize_str_df(df, LOCATION_COLS)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/locations failed: {e}")


@router.get("/api/forecastelements")
def api_forecastelements(
    productid: str = Query("", description="typed filter"),
    channelid: str = Query("", description="typed filter"),
    locationid: str = Query("", description="typed filter"),
    level: str = Query("", description="typed filter"),
    isactive: str = Query("", description="true/false typed filter"),
    limit: int = Query(20000, ge=1, le=50000),
    offset: int = Query(0, ge=0),
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    _ensure_engine()

    qual = _qualified(db_schema, FORECASTELEMENT_TABLE)

    where = ["1=1"]
    params: Dict[str, object] = {"limit": limit, "offset": offset}

    def add_like(col: str, val: str, pname: str):
        v = (val or "").strip()
        if not v:
            return
        v = v.replace("*", "%")
        if "%" not in v:
            v = f"%{v}%"
        where.append(f'CAST(t.{_qident(col)} AS TEXT) ILIKE :{pname}')
        params[pname] = v

    add_like("ProductID", productid, "p")
    add_like("ChannelID", channelid, "c")
    add_like("LocationID", locationid, "l")
    add_like("Level", level, "lev")

    ia = (isactive or "").strip().lower()
    if ia in ("true", "false"):
        where.append(f"t.{_qident('IsActive')} = :ia")
        params["ia"] = (ia == "true")
    elif ia:
        add_like("IsActive", isactive, "ia_like")

    where_sql = " AND ".join(where)
    cols_sql = ", ".join([f't.{_qident(c)} AS {_qident(c)}' for c in FORECASTELEMENT_COLS])

    sql_count = text(f"""
      SELECT COUNT(*) AS n
      FROM {qual} t
      WHERE {where_sql};
    """)

    sql_rows = text(f"""
      SELECT {cols_sql}
      FROM {qual} t
      WHERE {where_sql}
      ORDER BY t.{_qident("Level")}, t.{_qident("ProductID")}, t.{_qident("ChannelID")}, t.{_qident("LocationID")}
      LIMIT :limit OFFSET :offset;
    """)

    try:
        with ENGINE.begin() as conn:
            total = int(conn.execute(sql_count, params).scalar_one())
            rows = conn.execute(sql_rows, params).mappings().all()
        return {"count": total, "rows": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/forecastelements failed: {e}")