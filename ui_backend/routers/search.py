from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from core.db import ENGINE, get_engine, _qident, _qualified
from core.config import (
    DEFAULT_SCHEMA,
    TRIPLET_TABLE,
    PRODUCT_TABLE,
    CHANNEL_TABLE,
    LOCATION_TABLE,
)
from models.schemas import SavedSearchIn
from services.search_service import _build_where_from_query
from services.data_service import _ensure_saved_searches_table

router = APIRouter()


def _ensure_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()


@router.get("/api/saved-searches")
def api_saved_searches(db_schema: str = Query(DEFAULT_SCHEMA)):
    _ensure_engine()

    try:
        _ensure_saved_searches_table(db_schema)
        sql = text(f"""
          SELECT id, name, query, created_at::text AS created_at
          FROM {_qident(db_schema)}.saved_searches
          ORDER BY created_at DESC, id DESC;
        """)
        df = pd.read_sql_query(sql, ENGINE)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/saved-searches failed: {e}")


@router.post("/api/saved-searches")
def api_create_saved_search(
    body: SavedSearchIn,
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    _ensure_engine()

    _ensure_saved_searches_table(db_schema)

    name = (body.name or "").strip()
    query = (body.query or "").strip()

    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    _build_where_from_query(query)

    sql = text(f"""
      INSERT INTO {_qident(db_schema)}.saved_searches (name, query)
      VALUES (:name, :query)
      RETURNING id, name, query, created_at::text AS created_at;
    """)

    try:
        with ENGINE.begin() as conn:
            row = conn.execute(
                sql,
                {"name": name, "query": query}
            ).mappings().first()
        return dict(row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/saved-searches POST failed: {e}")


@router.delete("/api/saved-searches/{search_id}")
def api_delete_saved_search(
    search_id: int,
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    _ensure_engine()

    _ensure_saved_searches_table(db_schema)

    sql = text(f"""
      DELETE FROM {_qident(db_schema)}.saved_searches
      WHERE id = :id;
    """)

    try:
        with ENGINE.begin() as conn:
            conn.execute(sql, {"id": int(search_id)})
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/saved-searches DELETE failed: {e}")


@router.get("/api/search")
def api_search(
    q: str = Query(...),
    limit: int = Query(20000, ge=1, le=50000),
    offset: int = Query(0, ge=0),
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    _ensure_engine()

    where_sql, params = _build_where_from_query(q)

    trip_qual = _qualified(db_schema, TRIPLET_TABLE)
    prod_qual = _qualified(db_schema, PRODUCT_TABLE)
    chan_qual = _qualified(db_schema, CHANNEL_TABLE)
    loc_qual = _qualified(db_schema, LOCATION_TABLE)

    sql_keys = text(f"""
      SELECT DISTINCT
        t.{_qident("ProductID")}  AS "ProductID",
        t.{_qident("ChannelID")}  AS "ChannelID",
        t.{_qident("LocationID")} AS "LocationID"
      FROM {trip_qual} t
      JOIN {prod_qual} p ON p.{_qident("ProductID")} = t.{_qident("ProductID")}
      JOIN {chan_qual} c ON c.{_qident("ChannelID")} = t.{_qident("ChannelID")}
      JOIN {loc_qual}  l ON l.{_qident("LocationID")} = t.{_qident("LocationID")}
      WHERE {where_sql}
      ORDER BY 1,2,3
      LIMIT :limit OFFSET :offset;
    """)

    sql_count = text(f"""
      SELECT COUNT(*) AS n
      FROM (
        SELECT DISTINCT
          t.{_qident("ProductID")},
          t.{_qident("ChannelID")},
          t.{_qident("LocationID")}
        FROM {trip_qual} t
        JOIN {prod_qual} p ON p.{_qident("ProductID")} = t.{_qident("ProductID")}
        JOIN {chan_qual} c ON c.{_qident("ChannelID")} = t.{_qident("ChannelID")}
        JOIN {loc_qual}  l ON l.{_qident("LocationID")} = t.{_qident("LocationID")}
        WHERE {where_sql}
      ) x;
    """)

    try:
        with ENGINE.begin() as conn:
            total = int(conn.execute(sql_count, params).scalar_one())
            params2 = dict(params)
            params2["limit"] = limit
            params2["offset"] = offset
            rows = conn.execute(sql_keys, params2).mappings().all()

        return {
            "query": q,
            "count": total,
            "keys": [dict(r) for r in rows],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/search failed: {e}")