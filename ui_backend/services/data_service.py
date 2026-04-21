from __future__ import annotations

import pandas as pd
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy import text

from core.db import ENGINE, get_engine, _qident, _qualified
from core.config import (
    CLEANSED_HISTORY_TABLE,
    HISTORY_PREFIX,
    HISTORY_VIEW,
    CLASSIFIED_FE_TABLE,
    PRODUCT_COLS,
    CHANNEL_COLS,
    LOCATION_COLS,
)
from services.scenario_service import _get_scenario_chain, _read_overrides_for_chain


def _ensure_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()


def _history_table_name(level: str, period: str) -> str:
    pl = period.lower()
    if pl not in ("daily", "weekly", "monthly"):
        raise HTTPException(status_code=400, detail=f"Invalid period: {period}")
    return f"{HISTORY_PREFIX}_{level}_{pl}"


def _read_table_df(schema: str, table: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    _ensure_engine()
    qual = _qualified(schema, table)
    col_sql = ", ".join(_qident(c) for c in columns) if columns else "*"
    sql = f"SELECT {col_sql} FROM {qual};"
    try:
        return pd.read_sql_query(text(sql), ENGINE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB read failed for {schema}.{table}: {e}")


def _normalize_str_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("").map(
                lambda x: x.strip() if isinstance(x, str) else str(x)
            )
    return df


def _normalize_required_history_cols(df: pd.DataFrame, period: str) -> pd.DataFrame:
    req = ["ProductID", "ChannelID", "LocationID", "StartDate", "EndDate", "Period", "Qty"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"History table missing columns: {missing}. Required: {req}"
        )

    df = df.copy()
    df["Period"] = df["Period"].astype(str).str.strip()
    return df


def _normalize_weather_cols(df: pd.DataFrame) -> pd.DataFrame:
    req = ["LocationID", "Date"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Weather table missing columns: {missing}. Required: {req}"
        )
    return df.copy()


def _normalize_promo_cols(df: pd.DataFrame) -> pd.DataFrame:
    req = ["ProductID", "ChannelID", "LocationID", "StartDate", "EndDate"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Promo table missing columns: {missing}. Required: {req}"
        )
    return df.copy()


def _ensure_saved_searches_table(db_schema: str):
    _ensure_engine()

    ddl = f"""
    CREATE TABLE IF NOT EXISTS {_qident(db_schema)}.saved_searches (
      id SERIAL PRIMARY KEY,
      name TEXT NOT NULL,
      query TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """
    with ENGINE.begin() as conn:
        conn.execute(text(ddl))


def _ensure_cleansed_history_table(db_schema: str):
    _ensure_engine()

    table_qual = f'{_qident(db_schema)}.{_qident(CLEANSED_HISTORY_TABLE)}'

    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {table_qual} (
              id BIGSERIAL PRIMARY KEY,
              "scenario_id" INTEGER NOT NULL DEFAULT 0,
              "ProductID" TEXT NOT NULL,
              "ChannelID" TEXT NOT NULL,
              "LocationID" TEXT NOT NULL,
              "StartDate" DATE NOT NULL,
              "EndDate"   DATE NOT NULL,
              "Period"    TEXT NOT NULL,
              "Qty"       DOUBLE PRECISION NOT NULL,
              "NetPrice"  DOUBLE PRECISION NULL,
              "ListPrice" DOUBLE PRECISION NULL,
              "Level"     TEXT NULL,
              "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """))

        conn.execute(text(f'''
            ALTER TABLE {table_qual}
            ADD COLUMN IF NOT EXISTS "scenario_id" INTEGER NOT NULL DEFAULT 0;
        '''))

        conn.execute(text(f'''
            ALTER TABLE {table_qual}
            ADD COLUMN IF NOT EXISTS "NetPrice" DOUBLE PRECISION NULL;
        '''))

        conn.execute(text(f'''
            ALTER TABLE {table_qual}
            ADD COLUMN IF NOT EXISTS "ListPrice" DOUBLE PRECISION NULL;
        '''))

        try:
            conn.execute(text(f'''
                ALTER TABLE {table_qual}
                DROP CONSTRAINT IF EXISTS history_cleansed_ProductID_ChannelID_LocationID_StartDate_Period_key;
            '''))
        except Exception:
            pass

        conn.execute(text(f'''
            DROP INDEX IF EXISTS {_qident(db_schema)}.{_qident("idx_history_cleansed_keys")};
        '''))

        conn.execute(text(f'''
            DROP INDEX IF EXISTS {_qident(db_schema)}.{_qident("cleansed_history_uq_idx")};
        '''))

        conn.execute(text(f'''
            CREATE UNIQUE INDEX IF NOT EXISTS cleansed_history_uq_idx
            ON {table_qual}
            ("scenario_id","ProductID","ChannelID","LocationID","StartDate","Period");
        '''))

        conn.execute(text(f'''
            CREATE INDEX IF NOT EXISTS idx_history_cleansed_keys
            ON {table_qual}
            ("scenario_id","ProductID","ChannelID","LocationID","Period");
        '''))


def _period_ui_from_slug(slug: str) -> str:
    s = (slug or "").strip().lower()
    if s == "daily":
        return "Daily"
    if s == "weekly":
        return "Weekly"
    if s == "monthly":
        return "Monthly"
    raise HTTPException(status_code=400, detail="period must be one of: daily, weekly, monthly")


def _scenario_read_table(
    db_schema: str,
    table_name: str,
    scenario_id: int,
    pk_cols: List[str],
    where_sql: str = "1=1",
    where_params: Optional[Dict[str, Any]] = None,
    order_by_sql: Optional[str] = None,
) -> List[Dict[str, Any]]:
    _ensure_engine()

    chain = _get_scenario_chain(db_schema, int(scenario_id))
    overrides = _read_overrides_for_chain(db_schema, table_name, chain)

    base_qual = _qualified(db_schema, table_name)
    params = dict(where_params or {})

    sql = text(f"""
      SELECT t.*
      FROM {base_qual} t
      WHERE {where_sql}
      {("ORDER BY " + order_by_sql) if order_by_sql else ""};
    """)

    with ENGINE.begin() as conn:
        base_rows = conn.execute(sql, params).mappings().all()

    def pk_dict(row: Dict[str, Any]) -> Dict[str, Any]:
        return {c: row.get(c) for c in pk_cols}

    from services.scenario_service import _json_key
    import json

    out_map: Dict[str, Dict[str, Any]] = {}

    for r in base_rows:
        d = dict(r)
        out_map[_json_key(pk_dict(d))] = d

    for k, o in overrides.items():
        if bool(o.get("is_deleted")):
            out_map.pop(k, None)
            continue

        row_json = o.get("row") or {}
        if isinstance(row_json, str):
            try:
                row_json = json.loads(row_json)
            except Exception:
                row_json = {}

        if not isinstance(row_json, dict):
            row_json = {}

        if k in out_map:
            merged = dict(out_map[k])
            merged.update(row_json)
            out_map[k] = merged
        else:
            out_map[k] = dict(row_json)

    return list(out_map.values())


def _scenario_weather_df(
    db_schema: str,
    scenario_id: int,
    table_name: str,
) -> pd.DataFrame:
    rows = _scenario_read_table(
        db_schema=db_schema,
        table_name=table_name,
        scenario_id=scenario_id,
        pk_cols=["LocationID", "Date"],
        where_sql="1=1",
        where_params={},
        order_by_sql='"LocationID", "Date"',
    )
    return pd.DataFrame(rows)


def _scenario_promotions_df(
    db_schema: str,
    scenario_id: int,
    table_name: str,
) -> pd.DataFrame:
    rows = _scenario_read_table(
        db_schema=db_schema,
        table_name=table_name,
        scenario_id=scenario_id,
        pk_cols=["PromoID"],
        where_sql="1=1",
        where_params={},
        order_by_sql='"StartDate", "PromoID"',
    )
    return pd.DataFrame(rows)


def _scenario_cleansed_history_df(
    db_schema: str,
    scenario_id: int,
    period: str,
    level: Optional[str] = None,
) -> pd.DataFrame:
    _ensure_engine()
    _ensure_cleansed_history_table(db_schema)

    table_qual = _qualified(db_schema, CLEANSED_HISTORY_TABLE)
    sid = int(scenario_id or 0)

    where = ['"scenario_id" = :scenario_id', '"Period" = :period']
    params: Dict[str, Any] = {
        "scenario_id": sid,
        "period": str(period).strip(),
    }

    if level is not None and str(level).strip():
        where.append('"Level" = :level')
        params["level"] = str(level).strip()

    where_sql = " AND ".join(where)

    sql = text(f"""
        SELECT
            "ProductID",
            "ChannelID",
            "LocationID",
            "StartDate",
            "EndDate",
            "Period",
            "Qty",
            "NetPrice",
            "ListPrice",
            "Level"
        FROM {table_qual}
        WHERE {where_sql}
        ORDER BY "ProductID", "ChannelID", "LocationID", "StartDate";
    """)

    return pd.read_sql_query(sql, ENGINE, params=params)


def _get_history_with_fallback(
    db_schema: str,
    scenario_id: int,
    level: str,
    period: str,
) -> pd.DataFrame:
    try:
        hist_df = _scenario_cleansed_history_df(
            db_schema=db_schema,
            scenario_id=scenario_id,
            period=period,
            level=level,
        )
        if hist_df is not None and not hist_df.empty:
            return _normalize_required_history_cols(hist_df, period)
    except Exception:
        pass

    hist_table = _history_table_name(level, period)
    return _normalize_required_history_cols(
        _read_table_df(db_schema, hist_table),
        period
    )


def _history_by_keys(body, period_ui: str, db_schema: str, limit_per_key: int):
    _ensure_engine()

    keys = body.keys or []
    if not keys:
        return []

    view_qual = _qualified(db_schema, HISTORY_VIEW)

    clean_keys = []
    for k in keys:
        p = str(k.get("ProductID", "")).strip()
        c = str(k.get("ChannelID", "")).strip()
        l = str(k.get("LocationID", "")).strip()
        if p and c and l:
            clean_keys.append((p, c, l))

    if not clean_keys:
        return []

    values_sql = []
    params: Dict[str, object] = {"period": period_ui, "limit_per_key": int(limit_per_key)}

    for i, (p, c, l) in enumerate(clean_keys):
        values_sql.append(f"(:p{i}, :c{i}, :l{i})")
        params[f"p{i}"] = p
        params[f"c{i}"] = c
        params[f"l{i}"] = l

    values_block = ", ".join(values_sql)

    cols = [
        "ProductID", "ChannelID", "LocationID",
        "StartDate", "EndDate", "Qty",
        "Level", "Period",
        "NetPrice", "ListPrice",
    ]
    cols_sql = ", ".join([f'h.{_qident(c)} AS {_qident(c)}' for c in cols])

    sql = text(f"""
      WITH k("ProductID","ChannelID","LocationID") AS (
        VALUES {values_block}
      )
      SELECT
        {cols_sql},
        'Normal-History'::text AS "Type"
      FROM {view_qual} h
      JOIN k
        ON k."ProductID"  = h.{_qident("ProductID")}
       AND k."ChannelID"  = h.{_qident("ChannelID")}
       AND k."LocationID" = h.{_qident("LocationID")}
      WHERE LOWER(TRIM(h."Period")) = LOWER(TRIM(:period))
      ORDER BY
        h.{_qident("ProductID")},
        h.{_qident("ChannelID")},
        h.{_qident("LocationID")},
        h.{_qident("StartDate")} ASC
      LIMIT :limit_per_key * {len(clean_keys)};
    """)

    try:
        with ENGINE.begin() as conn:
            rows = conn.execute(sql, params).mappings().all()
        return [dict(r) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/history/*-by-keys failed: {e}")