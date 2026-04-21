from __future__ import annotations

from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from core.db import ENGINE, get_engine, _qident, _qualified
from core.config import (
    DEFAULT_SCHEMA,
    HISTORY_VIEW,
    HISTORY_COLS,
    HISTORY_FIELDS,
)
from models.schemas import KeysRequest
from services.search_service import (
    FIELD_VALUE_RE,
    _tokenize_query,
    _normalize_like,
)
from services.data_service import _history_by_keys

router = APIRouter()


def _ensure_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()


@router.get("/api/history/search")
def api_history_search(
    field: str = Query(..., description="ProductID | ChannelID | LocationID"),
    term: str = Query("", description="Typed search term"),
    period: Optional[str] = Query(None, description="Daily | Weekly | Monthly (optional)"),
    level: Optional[str] = Query(None, description="111/121/221 (optional)"),
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    _ensure_engine()

    f = (field or "").strip()
    if f not in HISTORY_FIELDS:
        raise HTTPException(status_code=400, detail=f"field must be one of {sorted(HISTORY_FIELDS)}")

    t = (term or "").strip()
    if not t:
        return {"field": f, "term": "", "count": 0, "rows": []}

    like = t.replace("*", "%")
    if "%" not in like:
        like = f"%{like}%"

    view_qual = _qualified(db_schema, HISTORY_VIEW)

    where = [f'CAST(h.{_qident(f)} AS TEXT) ILIKE :like']
    params: Dict[str, object] = {"like": like, "limit": limit, "offset": offset}

    if period:
        where.append(f'LOWER(TRIM(h.{_qident("Period")})) = LOWER(TRIM(:period))')
        params["period"] = period

    if level:
        where.append(f'LOWER(TRIM(h.{_qident("Level")})) = LOWER(TRIM(:level))')
        params["level"] = str(level)

    where_sql = " AND ".join(where)
    cols_sql = ", ".join([f'h.{_qident(c)} AS {_qident(c)}' for c in HISTORY_COLS])

    sql_count = text(f"""
      SELECT COUNT(*) AS n
      FROM {view_qual} h
      WHERE {where_sql};
    """)

    sql_rows = text(f"""
      SELECT {cols_sql}
      FROM {view_qual} h
      WHERE {where_sql}
      ORDER BY h.{_qident("StartDate")} DESC
      LIMIT :limit OFFSET :offset;
    """)

    try:
        with ENGINE.begin() as conn:
            total = int(conn.execute(sql_count, params).scalar_one())
            rows = conn.execute(sql_rows, params).mappings().all()
        return {"field": f, "term": t, "count": total, "rows": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/history/search failed: {e}")


@router.get("/api/history/by-query")
def api_history_by_query(
    q: str = Query(..., description="Query language: productid:.. AND (channelid:.. OR locationid:..)"),
    period: Optional[str] = Query(None, description="Daily | Weekly | Monthly (optional)"),
    level: Optional[str] = Query(None, description="111/121/221 (optional)"),
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    _ensure_engine()

    tokens = _tokenize_query(q)
    if not tokens:
        return {"q": q, "count": 0, "rows": []}

    if not any(FIELD_VALUE_RE.match(t) for t in tokens):
        raise HTTPException(status_code=400, detail='Invalid query. Use field:value (e.g., productid:*A*)')

    history_field_map: Dict[str, str] = {
        "productid": "ProductID",
        "channelid": "ChannelID",
        "locationid": "LocationID",
        "level": "Level",
        "period": "Period",
    }

    pos = 0
    param_index = 0
    params: Dict[str, object] = {}

    def peek() -> str:
        return tokens[pos] if pos < len(tokens) else ""

    def consume(expected: Optional[str] = None) -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise HTTPException(status_code=400, detail="Unexpected end of query.")
        t = tokens[pos]
        if expected and t != expected:
            raise HTTPException(status_code=400, detail=f"Expected {expected} but found {t}")
        pos += 1
        return t

    def clause_for_history(token: str) -> str:
        nonlocal param_index, params
        field, value = token.split(":", 1)
        f = field.lower().strip()
        if f not in history_field_map:
            raise HTTPException(status_code=400, detail=f"Unsupported field for history: {field}")

        col = history_field_map[f]
        pname = f"v{param_index}"
        param_index += 1
        params[pname] = _normalize_like(value)
        return f'CAST(h.{_qident(col)} AS TEXT) ILIKE :{pname}'

    def parse_factor() -> str:
        t = peek()
        if t == "(":
            consume("(")
            inner = parse_expr()
            if peek() != ")":
                raise HTTPException(status_code=400, detail="Missing ')'")
            consume(")")
            return f"({inner})"

        t = consume()
        if not FIELD_VALUE_RE.match(t):
            raise HTTPException(status_code=400, detail=f"Invalid token: {t}")
        return clause_for_history(t)

    def parse_term() -> str:
        left = parse_factor()
        while peek() == "AND":
            consume("AND")
            right = parse_factor()
            left = f"({left} AND {right})"
        return left

    def parse_expr() -> str:
        left = parse_term()
        while peek() == "OR":
            consume("OR")
            right = parse_term()
            left = f"({left} OR {right})"
        return left

    where_sql = parse_expr()
    if pos != len(tokens):
        raise HTTPException(status_code=400, detail=f"Unexpected token: {tokens[pos]}")

    extra = []
    if period:
        extra.append(f'LOWER(TRIM(h.{_qident("Period")})) = LOWER(TRIM(:period))')
        params["period"] = period

    if level:
        extra.append(f'LOWER(TRIM(h.{_qident("Level")})) = LOWER(TRIM(:level))')
        params["level"] = str(level)

    full_where = where_sql
    if extra:
        full_where = f"({where_sql}) AND " + " AND ".join(extra)

    view_qual = _qualified(db_schema, HISTORY_VIEW)
    cols_sql = ", ".join([f'h.{_qident(c)} AS {_qident(c)}' for c in HISTORY_COLS])

    params["limit"] = limit
    params["offset"] = offset

    sql_count = text(f"""
      SELECT COUNT(*) AS n
      FROM {view_qual} h
      WHERE {full_where};
    """)

    sql_rows = text(f"""
      SELECT {cols_sql}
      FROM {view_qual} h
      WHERE {full_where}
      ORDER BY h.{_qident("StartDate")} DESC
      LIMIT :limit OFFSET :offset;
    """)

    try:
        with ENGINE.begin() as conn:
            total = int(conn.execute(sql_count, params).scalar_one())
            rows = conn.execute(sql_rows, params).mappings().all()
        return {"q": q, "count": total, "rows": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/history/by-query failed: {e}")


@router.post("/api/history/daily-by-keys")
def api_history_daily_by_keys(
    body: KeysRequest,
    db_schema: str = Query(DEFAULT_SCHEMA),
    limit_per_key: int = Query(5000, ge=1, le=20000),
):
    return _history_by_keys(body, period_ui="Daily", db_schema=db_schema, limit_per_key=limit_per_key)


@router.post("/api/history/weekly-by-keys")
def api_history_weekly_by_keys(
    body: KeysRequest,
    db_schema: str = Query(DEFAULT_SCHEMA),
    limit_per_key: int = Query(5000, ge=1, le=20000),
):
    return _history_by_keys(body, period_ui="Weekly", db_schema=db_schema, limit_per_key=limit_per_key)


@router.post("/api/history/monthly-by-keys")
def api_history_monthly_by_keys(
    body: KeysRequest,
    db_schema: str = Query(DEFAULT_SCHEMA),
    limit_per_key: int = Query(5000, ge=1, le=20000),
):
    return _history_by_keys(body, period_ui="Monthly", db_schema=db_schema, limit_per_key=limit_per_key)