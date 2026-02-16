# main.py
from __future__ import annotations

import os
import re
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi import Body
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime, timezone
from fastapi import Query, HTTPException
from sqlalchemy import text

import forecast  # must expose run_one_job_df + JOBS + GRAIN_CFG


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

OUT_DIR = Path(os.getenv("FORECAST_OUT_DIR", str(BASE_DIR / "outputs"))).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".csv", ".png", ".txt", ".log"}

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DEFAULT_SCHEMA = os.getenv("DB_SCHEMA", "planwise_fresh_produce")

PRODUCT_TABLE = os.getenv("PRODUCT_TABLE", "product")
CHANNEL_TABLE = os.getenv("CHANNEL_TABLE", "channel")
LOCATION_TABLE = os.getenv("LOCATION_TABLE", "location")
TRIPLET_TABLE = os.getenv("TRIPLET_TABLE", "forecastelement")

DEFAULT_WEATHER_TABLE = os.getenv("WEATHER_TABLE", "weather_daily")
DEFAULT_PROMO_TABLE = os.getenv("PROMO_TABLE", "promotions")
HISTORY_PREFIX = os.getenv("HISTORY_PREFIX", "history")

FORECASTELEMENT_TABLE = os.getenv("FORECASTELEMENT_TABLE", "forecastelement")
FORECASTELEMENT_COLS = ["ProductID", "ChannelID", "LocationID", "Level", "IsActive"]

# ✅ view name for history page
HISTORY_VIEW = os.getenv("HISTORY_VIEW", "v_history")

PRODUCT_COLS = ["ProductID", "ProductDescr", "Level", "BusinessUnit", "IsDailyForecastRequired", "IsNew", "ProductFamily"]
CHANNEL_COLS = ["ChannelID", "ChannelDescr", "Level"]

# ✅ views for forecast page
FORECAST_BASELINE_VIEW = os.getenv("FORECAST_BASELINE_VIEW", "v_forecast_baseline")
FORECAST_FEAT_VIEW = os.getenv("FORECAST_FEAT_VIEW", "v_forecast_feat")

FORECAST_COLS = [
    "Level", "Model",
    "ProductID", "ChannelID", "LocationID",
    "StartDate", "EndDate", "Period",
    "ForecastQty", "UOM",
    "NetPrice", "ListPrice",
    "Method",
]
FORECAST_FIELDS = {"ProductID", "ChannelID", "LocationID"}  # typed search fields

# --- Cleanse: profiles + cleansed storage ---
CLEANSE_PROFILES_TABLE = os.getenv("CLEANSE_PROFILES_TABLE", "cleanse_profiles")

CLEANSED_HISTORY_TABLE = os.getenv("CLEANSED_HISTORY_TABLE", "history_cleansed")
CLEANSED_HISTORY_COLS = ["ProductID","ChannelID","LocationID","StartDate","EndDate","Period","Qty","NetPrice","ListPrice","Level"]

CLASSIFIED_FE_TABLE = "classified_forecast_elements"

# ✅ match your DB: location table has IsActive (no Geography)
LOCATION_COLS = ["LocationID", "LocationDescr", "Level", "IsActive"]

ENGINE = None


def get_engine():
    if not DATABASE_URL:
        raise HTTPException(
            status_code=500,
            detail="DATABASE_URL is not set. Example: postgresql+psycopg2://user:pass@host:5432/dbname",
        )
    return create_engine(DATABASE_URL, pool_pre_ping=True)


# ------------------------------------------------------------
# APP
# ------------------------------------------------------------
app = FastAPI(title="PlanWise API (DB tables)", version="3.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# REQUEST MODELS
# ------------------------------------------------------------
class RunOneDBRequest(BaseModel):
    db_schema: str = Field(default=DEFAULT_SCHEMA)

    level: Optional[str] = Field(default=None, description="e.g. 111 / 121 / 221")
    period: str = Field(..., description="Daily / Weekly / Monthly")
    horizon: int = Field(..., ge=1)

    history_table: Optional[str] = Field(
        default=None,
        description="Explicit table name (without schema). If omitted, we build history_{level}_{periodlower}.",
    )

    weather_table: str = Field(default=DEFAULT_WEATHER_TABLE)
    promo_table: str = Field(default=DEFAULT_PROMO_TABLE)
    tag: Optional[str] = Field(default=None, description="Output tag used in filenames")


class RunAllDBRequest(BaseModel):
    db_schema: str = Field(default=DEFAULT_SCHEMA)
    weather_table: str = Field(default=DEFAULT_WEATHER_TABLE)
    promo_table: str = Field(default=DEFAULT_PROMO_TABLE)


class KeysRequest(BaseModel):
    keys: List[Dict[str, str]]  # expects ProductID/ChannelID/LocationID


class CleanseProfileIn(BaseModel):
    name: str
    config: dict


class CleansedIngestRequest(BaseModel):
    period: str  # daily/weekly/monthly (from Angular)
    rows: List[Dict[str, object]]


class SavedSearchIn(BaseModel):
    name: str
    query: str

# Keep a simple in-memory "last computed" timestamp per schema+period
_LAST_CLASSIFY_COMPUTED: Dict[str, str] = {}  # key = f"{schema}:{period_slug}" -> iso ts

class ClassifyComputeRequest(BaseModel):
    period: str  # 'daily' | 'weekly' | 'monthly'
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
    CreatedAt: Optional[str] = None  # from UI (ComputedAt) if you pass it

class ClassifySaveRequest(BaseModel):
    period: str  # 'daily' | 'weekly' | 'monthly'
    rows: List[ClassifySaveRow] = []

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def _safe_filename(name: str) -> str:
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    return name


def _qident(s: str) -> str:
    s = str(s).replace('"', '""')
    return f'"{s}"'


def _qualified(schema: str, table: str) -> str:
    return f"{_qident(schema)}.{_qident(table)}"


def _history_table_name(level: str, period: str) -> str:
    pl = period.lower()
    if pl not in ("daily", "weekly", "monthly"):
        raise HTTPException(status_code=400, detail=f"Invalid period: {period}")
    return f"{HISTORY_PREFIX}_{level}_{pl}"


def _read_table_df(schema: str, table: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

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
            df[c] = df[c].astype(str).fillna("").map(lambda x: x.strip() if isinstance(x, str) else str(x))
    return df


def _collect_outputs(tag: str) -> List[str]:
    outs = [
        f"Forecast_{tag}_baseline.csv",
        f"Forecast_{tag}_feat.csv",
        f"Backtest_{tag}_summary.csv",
        f"Compare_{tag}_plots.png",
        f"Compare_{tag}_wmape.png",
    ]
    return [f for f in outs if (OUT_DIR / f).exists()]


def _normalize_required_history_cols(df: pd.DataFrame, period: str) -> pd.DataFrame:
    req = ["ProductID", "ChannelID", "LocationID", "StartDate", "EndDate", "Period", "Qty"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"History table missing columns: {missing}. Required: {req}")

    df = df.copy()
    df["Period"] = df["Period"].astype(str).str.strip()
    return df


def _normalize_weather_cols(df: pd.DataFrame) -> pd.DataFrame:
    req = ["LocationID", "Date"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Weather table missing columns: {missing}. Required: {req}")
    return df.copy()


def _normalize_promo_cols(df: pd.DataFrame) -> pd.DataFrame:
    req = ["ProductID", "ChannelID", "LocationID", "StartDate", "EndDate"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Promo table missing columns: {missing}. Required: {req}")
    return df.copy()


def _ensure_saved_searches_table(db_schema: str):
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

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
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

    ddl = f"""
    CREATE TABLE IF NOT EXISTS {_qident(db_schema)}.{_qident(CLEANSED_HISTORY_TABLE)} (
      id BIGSERIAL PRIMARY KEY,
      "ProductID" TEXT NOT NULL,
      "ChannelID" TEXT NOT NULL,
      "LocationID" TEXT NOT NULL,
      "StartDate" DATE NOT NULL,
      "EndDate"   DATE NOT NULL,
      "Period"    TEXT NOT NULL,
      "Qty"       DOUBLE PRECISION NOT NULL,
      "Level"     TEXT NULL,
      "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT now(),
      UNIQUE("ProductID","ChannelID","LocationID","StartDate","Period")
    );

    CREATE INDEX IF NOT EXISTS idx_history_cleansed_keys
      ON {_qident(db_schema)}.{_qident(CLEANSED_HISTORY_TABLE)}
      ("ProductID","ChannelID","LocationID","Period");
    """
    with ENGINE.begin() as conn:
        for stmt in ddl.split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s + ";"))


def _period_ui_from_slug(slug: str) -> str:
    s = (slug or "").strip().lower()
    if s == "daily":
        return "Daily"
    if s == "weekly":
        return "Weekly"
    if s == "monthly":
        return "Monthly"
    raise HTTPException(status_code=400, detail="period must be one of: daily, weekly, monthly")

def _ensure_classified_fe_table(db_schema: str):
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

    qual = _qualified(db_schema, CLASSIFIED_FE_TABLE)

    ddl = text(f"""
    CREATE TABLE IF NOT EXISTS {qual} (
      "ProductID"  text NOT NULL,
      "ChannelID"  text NOT NULL,
      "LocationID" text NOT NULL,
      "Period"     text NOT NULL, -- 'Daily'/'Weekly'/'Monthly'
      "ADI"        double precision NULL,
      "CV2"        double precision NULL,
      "Category"   text NOT NULL,
      "Algorithm"  text NOT NULL,
      "CreatedAt"  timestamp with time zone NOT NULL DEFAULT now(),
      "UpdatedAt"  timestamp with time zone NOT NULL DEFAULT now(),
      PRIMARY KEY ("ProductID","ChannelID","LocationID","Period")
    );
    """)
    with ENGINE.begin() as conn:
        conn.execute(ddl)

# ------------------------------------------------------------
# QUERY LANGUAGE (AND/OR + parentheses)
# ------------------------------------------------------------
# field:value where value may be "quoted string" or bare token
FIELD_VALUE_RE = re.compile(r'^[A-Za-z_]\w*:(?:"[^"]*"|\S+)$')

def _tokenize_query(q: str) -> List[str]:
    # tokens: '(', ')', 'AND', 'OR', or field:value
    q = (q or "").strip()
    if not q:
        return []
    raw = re.findall(r'\(|\)|"[^"]*"|\S+', q)
    tokens: List[str] = []
    for t in raw:
        up = t.upper()
        if up in ("AND", "OR") or t in ("(", ")"):
            tokens.append(up if up in ("AND", "OR") else t)
        else:
            tokens.append(t)
    return tokens


def _normalize_like(v: str) -> str:
    v = (v or "").strip()
    if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        v = v[1:-1]
    v = v.replace("*", "%")
    if "%" not in v:
        v = f"%{v}%"
    return v


# field -> (alias, column) for the /api/search join query
ALLOWED_FIELDS: Dict[str, Tuple[str, str]] = {
    # product (p)
    "productid": ("p", "ProductID"),
    "productdescr": ("p", "ProductDescr"),
    "productlevel": ("p", "Level"),
    "businessunit": ("p", "BusinessUnit"),
    "isdailyforecastrequired": ("p", "IsDailyForecastRequired"),
    "isnew": ("p", "IsNew"),
    "productfamily": ("p", "ProductFamily"),
    # channel (c)
    "channelid": ("c", "ChannelID"),
    "channeldescr": ("c", "ChannelDescr"),
    "channellevel": ("c", "Level"),
    # location (l)
    "locationid": ("l", "LocationID"),
    "locationdescr": ("l", "LocationDescr"),
    "locationlevel": ("l", "Level"),
    "isactive": ("l", "IsActive"),
}


def _clause_for_field_value(token: str, i: int) -> Tuple[str, Dict[str, str]]:
    field, value = token.split(":", 1)
    f = field.lower().strip()
    if f not in ALLOWED_FIELDS:
        raise HTTPException(status_code=400, detail=f"Unsupported field: {field}")

    alias, col = ALLOWED_FIELDS[f]
    pname = f"v{i}"
    clause = f'CAST({alias}.{_qident(col)} AS TEXT) ILIKE :{pname}'
    params = {pname: _normalize_like(value)}
    return clause, params


def _build_where_from_query(q: str) -> Tuple[str, Dict[str, str]]:
    """
    Grammar:
      expr  := term (OR term)*
      term  := factor (AND factor)*
      factor:= field:value | '(' expr ')'
    AND has higher precedence than OR.
    """
    tokens = _tokenize_query(q)
    if not tokens:
        return "1=1", {}

    if not any(FIELD_VALUE_RE.match(t) for t in tokens):
        raise HTTPException(status_code=400, detail='Invalid query. Use field:value (e.g., productid:*A*)')

    pos = 0
    param_index = 0
    params: Dict[str, str] = {}

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

    def parse_factor() -> str:
        nonlocal param_index, params
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
        clause, p = _clause_for_field_value(t, param_index)
        param_index += 1
        params.update(p)
        return clause

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
    return where_sql, params

def _history_by_keys(body: KeysRequest, period_ui: str, db_schema: str, limit_per_key: int):
    """
    Fetch raw history rows from v_history for many keys for one Period (Daily/Weekly/Monthly).
    Returns rows with extra 'Type' column set to 'Normal-History' (frontend expects it).
    """
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

    keys = body.keys or []
    if not keys:
        return []

    view_qual = _qualified(db_schema, HISTORY_VIEW)

    # sanitize + normalize keys
    clean_keys = []
    for k in keys:
        p = str(k.get("ProductID", "")).strip()
        c = str(k.get("ChannelID", "")).strip()
        l = str(k.get("LocationID", "")).strip()
        if p and c and l:
            clean_keys.append((p, c, l))

    if not clean_keys:
        return []

    # Build a VALUES list: (:p0,:c0,:l0), (:p1,:c1,:l1) ...
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

    # For each key, we take up to limit_per_key rows (avoid exploding response)
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
# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "default_schema": DEFAULT_SCHEMA,
        "tables": {
            "product": PRODUCT_TABLE,
            "channel": CHANNEL_TABLE,
            "location": LOCATION_TABLE,
            "triplets": TRIPLET_TABLE,
            "weather": DEFAULT_WEATHER_TABLE,
            "promotions": DEFAULT_PROMO_TABLE,
            "history_view": HISTORY_VIEW,
        },
        "forecast_jobs": [{"level": j[0], "period": j[1], "horizon": j[2]} for j in forecast.JOBS],
    }


# -------------------------
# MASTER DATA (Angular)
# -------------------------
@app.get("/api/products")
def api_products(db_schema: str = Query(DEFAULT_SCHEMA)):
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

    qual = _qualified(db_schema, PRODUCT_TABLE)
    cols = ", ".join(_qident(c) for c in PRODUCT_COLS)
    sql = text(f"SELECT {cols} FROM {qual} ORDER BY {_qident('ProductID')};")
    try:
        df = pd.read_sql_query(sql, ENGINE)
        df = _normalize_str_df(df, PRODUCT_COLS)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/products failed: {e}")


@app.get("/api/channels")
def api_channels(db_schema: str = Query(DEFAULT_SCHEMA)):
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

    qual = _qualified(db_schema, CHANNEL_TABLE)
    cols = ", ".join(_qident(c) for c in CHANNEL_COLS)
    sql = text(f"SELECT {cols} FROM {qual} ORDER BY {_qident('Level')}, {_qident('ChannelID')};")
    try:
        df = pd.read_sql_query(sql, ENGINE)
        df = _normalize_str_df(df, CHANNEL_COLS)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/channels failed: {e}")


@app.get("/api/locations")
def api_locations(db_schema: str = Query(DEFAULT_SCHEMA)):
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

    qual = _qualified(db_schema, LOCATION_TABLE)
    cols = ", ".join(_qident(c) for c in LOCATION_COLS)
    sql = text(f"SELECT {cols} FROM {qual} ORDER BY {_qident('Level')}, {_qident('LocationID')};")
    try:
        df = pd.read_sql_query(sql, ENGINE)
        df = _normalize_str_df(df, LOCATION_COLS)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/locations failed: {e}")


@app.get("/api/forecastelements")
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
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

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


# -------------------------
# SAVED SEARCHES (CRUD)
# -------------------------
@app.get("/api/saved-searches")
def api_saved_searches(db_schema: str = Query(DEFAULT_SCHEMA)):
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

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


@app.post("/api/saved-searches")
def api_create_saved_search(body: SavedSearchIn, db_schema: str = Query(DEFAULT_SCHEMA)):
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

    _ensure_saved_searches_table(db_schema)

    name = (body.name or "").strip()
    query = (body.query or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    # validate query early (so bad ones don’t get stored)
    _build_where_from_query(query)

    sql = text(f"""
      INSERT INTO {_qident(db_schema)}.saved_searches (name, query)
      VALUES (:name, :query)
      RETURNING id, name, query, created_at::text AS created_at;
    """)
    try:
        with ENGINE.begin() as conn:
            row = conn.execute(sql, {"name": name, "query": query}).mappings().first()
        return dict(row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/saved-searches POST failed: {e}")


@app.delete("/api/saved-searches/{search_id}")
def api_delete_saved_search(search_id: int, db_schema: str = Query(DEFAULT_SCHEMA)):
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

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


# -------------------------
# /api/search  (saved-search keys against forecastelement join)
# -------------------------
@app.get("/api/search")
def api_search(
    q: str = Query(...),
    limit: int = Query(20000, ge=1, le=50000),
    offset: int = Query(0, ge=0),
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

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
        return {"query": q, "count": total, "keys": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/search failed: {e}")


# -------------------------
# HISTORY (v_history) API
# -------------------------
HISTORY_COLS = [
  "Level","ProductID","ChannelID","LocationID",
  "StartDate","EndDate","Period",
  "Qty","NetPrice","ListPrice",
  "Type",
]

HISTORY_FIELDS = {"ProductID", "ChannelID", "LocationID"}  # typed search fields


@app.get("/api/history/search")
def api_history_search(
    field: str = Query(..., description="ProductID | ChannelID | LocationID"),
    term: str = Query("", description="Typed search term"),
    period: Optional[str] = Query(None, description="Daily | Weekly | Monthly (optional)"),
    level: Optional[str] = Query(None, description="111/121/221 (optional)"),
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

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
        where.append(f'LOWER(TRIM(f.{_qident("Level")})) = LOWER(TRIM(:level))')
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


# ✅ NEW: run saved-search query language on v_history
@app.get("/api/history/by-query")
def api_history_by_query(
    q: str = Query(..., description="Query language: productid:.. AND (channelid:.. OR locationid:..)"),
    period: Optional[str] = Query(None, description="Daily | Weekly | Monthly (optional)"),
    level: Optional[str] = Query(None, description="111/121/221 (optional)"),
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    """
    Applies the SAME saved-search query language to v_history.
    We must map aliases to h.<Column> for history.
    """
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

    # Build where using the parser, but we need history alias "h" instead of p/c/l.
    # Easiest: rebuild clauses token-by-token directly to h.<col>.
    tokens = _tokenize_query(q)
    if not tokens:
        return {"q": q, "count": 0, "rows": []}

    # validate field:value exists
    if not any(FIELD_VALUE_RE.match(t) for t in tokens):
        raise HTTPException(status_code=400, detail='Invalid query. Use field:value (e.g., productid:*A*)')

    # For history, valid fields are just productid/channelid/locationid/level/period/isactive isn't in history
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


# -------------------------
# CLEANSE PROFILES (unchanged minimal)
# -------------------------
@app.get("/api/cleanse/profiles")
def list_cleanse_profiles(db_schema: str = Query(DEFAULT_SCHEMA)):
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

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


@app.post("/api/cleanse/profiles")
def upsert_cleanse_profile(body: CleanseProfileIn, db_schema: str = Query(DEFAULT_SCHEMA)):
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

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

@app.post("/api/history/ingest-cleansed")
def api_ingest_cleansed_history(
    body: CleansedIngestRequest,
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

    _ensure_cleansed_history_table(db_schema)

    period_ui = _period_ui_from_slug(body.period)
    rows = body.rows or []
    if not rows:
        return {"ok": True, "count": 0}

    table_qual = _qualified(db_schema, CLEANSED_HISTORY_TABLE)

    sql = text(f"""
        INSERT INTO {table_qual}
            ("ProductID","ChannelID","LocationID","StartDate","EndDate","Period","Qty","NetPrice","ListPrice","Level")
        VALUES
            (:ProductID, :ChannelID, :LocationID,
             CAST(:StartDate AS date), CAST(:EndDate AS date),
             :Period, :Qty, :NetPrice, :ListPrice, :Level)
        ON CONFLICT ("ProductID","ChannelID","LocationID","StartDate","Period")
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

    def _float_or_none(x: object) -> Optional[float]:
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
            # StartDate is mandatory; skip bad rows
            continue

        if not end:
            # fallback: use StartDate when EndDate missing
            end = start

        payload.append({
            "ProductID": str(r.get("ProductID", "")).strip(),
            "ChannelID": str(r.get("ChannelID", "")).strip(),
            "LocationID": str(r.get("LocationID", "")).strip(),
            "StartDate": start,
            "EndDate": end,
            "Period": period_ui,
            "Qty": float(r.get("Qty", 0) or 0),

            # ✅ MUST EXIST for SQL bind params (can be None)
            "NetPrice": _float_or_none(r.get("NetPrice")),
            "ListPrice": _float_or_none(r.get("ListPrice")),

            "Level": (str(r.get("Level", "")).strip() or None),
        })

    # ✅ this must be OUTSIDE the loop
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

        return {"ok": True, "count": len(payload)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/history/ingest-cleansed failed: {e}")

  

# -------------------------
# HISTORY by keys (for Cleanse page)
# -------------------------
@app.post("/api/history/daily-by-keys")
def api_history_daily_by_keys(
    body: KeysRequest,
    db_schema: str = Query(DEFAULT_SCHEMA),
    limit_per_key: int = Query(5000, ge=1, le=20000),
):
    return _history_by_keys(body, period_ui="Daily", db_schema=db_schema, limit_per_key=limit_per_key)

@app.post("/api/history/weekly-by-keys")
def api_history_weekly_by_keys(
    body: KeysRequest,
    db_schema: str = Query(DEFAULT_SCHEMA),
    limit_per_key: int = Query(5000, ge=1, le=20000),
):
    return _history_by_keys(body, period_ui="Weekly", db_schema=db_schema, limit_per_key=limit_per_key)

@app.post("/api/history/monthly-by-keys")
def api_history_monthly_by_keys(
    body: KeysRequest,
    db_schema: str = Query(DEFAULT_SCHEMA),
    limit_per_key: int = Query(5000, ge=1, le=20000),
):
    return _history_by_keys(body, period_ui="Monthly", db_schema=db_schema, limit_per_key=limit_per_key)

@app.post("/api/classify/compute")
def api_classify_compute(
    body: ClassifyComputeRequest,
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    # We keep this lightweight: mark a compute timestamp
    period = (body.period or "").strip().lower()
    if period not in ("daily", "weekly", "monthly"):
        raise HTTPException(status_code=400, detail="period must be one of: daily, weekly, monthly")

    ts = datetime.now(timezone.utc).isoformat()
    _LAST_CLASSIFY_COMPUTED[f"{db_schema}:{period}"] = ts

    return {"ok": True, "period": period, "computed_at": ts}

@app.get("/api/classify/results")
def api_classify_results(
    period: str = Query(..., description="daily|weekly|monthly"),
    include_inactive: bool = Query(True),
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

    p = (period or "").strip().lower()
    if p not in ("daily", "weekly", "monthly"):
        raise HTTPException(status_code=400, detail="period must be one of: daily, weekly, monthly")

    # your history_cleansed stores Period as UI strings like 'Daily'/'Weekly'/'Monthly'
    period_ui = _period_ui_from_slug(p)  # you already have this helper

    table_qual = _qualified(db_schema, CLEANSED_HISTORY_TABLE)

    # Distinct keys that exist in cleansed history for this period
    sql = text(f"""
      SELECT DISTINCT
        "ProductID"  AS "ProductID",
        "ChannelID"  AS "ChannelID",
        "LocationID" AS "LocationID"
      FROM {table_qual}
      WHERE "Period" = :period_ui
      ORDER BY 1,2,3
      LIMIT 50000;
    """)

    computed_at = _LAST_CLASSIFY_COMPUTED.get(f"{db_schema}:{p}") \
                  or datetime.now(timezone.utc).isoformat()

    try:
        with ENGINE.begin() as conn:
            keys = conn.execute(sql, {"period_ui": period_ui}).mappings().all()

        # return in the shape your Angular expects
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
            })
        return out

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/classify/results failed: {e}")

@app.post("/api/classify/save")
def api_classify_save(
    body: ClassifySaveRequest,
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

    _ensure_classified_fe_table(db_schema)

    period = (body.period or "").strip().lower()
    if period not in ("daily", "weekly", "monthly"):
        raise HTTPException(status_code=400, detail="period must be one of: daily, weekly, monthly")
    period_ui = _period_ui_from_slug(period)

    rows = body.rows or []
    if not rows:
        return {"ok": True, "count": 0}

    qual = _qualified(db_schema, CLASSIFIED_FE_TABLE)

    sql = text(f"""
      INSERT INTO {qual}
        ("ProductID","ChannelID","LocationID","Period","ADI","CV2","Category","Algorithm","CreatedAt","UpdatedAt")
      VALUES
        (:ProductID,:ChannelID,:LocationID,:Period,:ADI,:CV2,:Category,:Algorithm,
         COALESCE(CAST(:CreatedAt AS timestamptz), now()),
         now())
      ON CONFLICT ("ProductID","ChannelID","LocationID","Period")
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

    try:
        with ENGINE.begin() as conn:
            chunk = 5000
            for i in range(0, len(payload), chunk):
                conn.execute(sql, payload[i:i+chunk])
        return {"ok": True, "count": len(payload)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/classify/save failed: {e}")

@app.get("/api/classify/saved")
def api_classify_saved(
    period: str = Query(..., description="daily|weekly|monthly"),
    include_inactive: bool = Query(False, description="if false, only rows that are active in cleansed history"),
    limit: int = Query(20000, ge=1, le=50000),
    offset: int = Query(0, ge=0),
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

    p = (period or "").strip().lower()
    if p not in ("daily", "weekly", "monthly"):
        raise HTTPException(status_code=400, detail="period must be one of: daily, weekly, monthly")

    period_ui = _period_ui_from_slug(p)  # 'Daily'/'Weekly'/'Monthly'

    cls_qual = _qualified(db_schema, CLASSIFIED_FE_TABLE)
    hist_qual = _qualified(db_schema, CLEANSED_HISTORY_TABLE)

    # Active keys = those currently present in history_cleansed for this period
    sql = text(f"""
      WITH active_keys AS (
        SELECT DISTINCT "ProductID","ChannelID","LocationID"
        FROM {hist_qual}
        WHERE "Period" = :period_ui
      )
      SELECT
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
      WHERE c."Period" = :period_ui
        AND (:include_inactive = TRUE OR ak."ProductID" IS NOT NULL)
      ORDER BY c."UpdatedAt" DESC, c."ProductID", c."ChannelID", c."LocationID"
      LIMIT :limit OFFSET :offset;
    """)

    try:
        with ENGINE.begin() as conn:
            rows = conn.execute(sql, {
                "period_ui": period_ui,
                "include_inactive": include_inactive,
                "limit": limit,
                "offset": offset
            }).mappings().all()
        return [dict(r) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/classify/saved failed: {e}")


# -------------------------
# FORECAST (views) API (unchanged)
# -------------------------
@app.get("/api/forecast")
def api_forecast(
    variant: str = Query("baseline", description="baseline | feat"),
    productid: str = Query(...),
    channelid: str = Query(...),
    locationid: str = Query(...),
    period: str = Query(..., description="Daily/Weekly/Monthly"),
    model: Optional[str] = Query(None),
    method: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

    v = (variant or "baseline").strip().lower()
    view_name = FORECAST_BASELINE_VIEW if v in ("baseline", "base") else FORECAST_FEAT_VIEW
    view_qual = _qualified(db_schema, view_name)

    cols_sql = ", ".join([f'f.{_qident(c)} AS {_qident(c)}' for c in FORECAST_COLS])

    where = [
        f"f.{_qident('ProductID')} = :p",
        f"f.{_qident('ChannelID')} = :c",
        f"f.{_qident('LocationID')} = :l",
        f"LOWER(TRIM(f.{_qident('Period')})) = LOWER(TRIM(:period))",
    ]
    params: Dict[str, object] = {
        "p": productid,
        "c": channelid,
        "l": locationid,
        "period": period,
        "limit": limit,
        "offset": offset,
    }

    if model:
        where.append(f"f.{_qident('Model')} = :model")
        params["model"] = model.strip()

    if method:
        where.append(f"f.{_qident('Method')} = :method")
        params["method"] = method.strip()

    where_sql = " AND ".join(where)

    sql_count = text(f"""
      SELECT COUNT(*) AS n
      FROM {view_qual} f
      WHERE {where_sql};
    """)

    sql_rows = text(f"""
      SELECT {cols_sql}
      FROM {view_qual} f
      WHERE {where_sql}
      ORDER BY f.{_qident("StartDate")} ASC
      LIMIT :limit OFFSET :offset;
    """)

    try:
        with ENGINE.begin() as conn:
            total = int(conn.execute(sql_count, params).scalar_one())
            rows = conn.execute(sql_rows, params).mappings().all()
        return {"variant": v, "count": total, "rows": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/forecast failed: {e}")


@app.get("/api/forecast/search")
def api_forecast_search(
    variant: str = Query("baseline", description="baseline | feat"),
    field: str = Query(..., description="ProductID | ChannelID | LocationID"),
    term: str = Query("", description="Typed search term"),
    period: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    method: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

    v = (variant or "baseline").strip().lower()
    view_name = FORECAST_BASELINE_VIEW if v in ("baseline", "base") else FORECAST_FEAT_VIEW
    view_qual = _qualified(db_schema, view_name)

    f = (field or "").strip()
    if f not in FORECAST_FIELDS:
        raise HTTPException(status_code=400, detail=f"field must be one of {sorted(FORECAST_FIELDS)}")

    t = (term or "").strip()
    if not t:
        return {"variant": v, "field": f, "term": "", "count": 0, "rows": []}

    like = t.replace("*", "%")
    if "%" not in like:
        like = f"%{like}%"

    where = [f'CAST(f.{_qident(f)} AS TEXT) ILIKE :like']
    params: Dict[str, object] = {"like": like, "limit": limit, "offset": offset}

    if period:
        where.append(f'LOWER(TRIM(f.{_qident("Period")})) = LOWER(TRIM(:period))')
        params["period"] = period


    if level:
        where.append(f'LOWER(TRIM(h.{_qident("Level")})) = LOWER(TRIM(:level))')
        params["level"] = str(level)

    if model:
        mm = model.strip()
        mm_like = mm.replace("*", "%")
        if "%" not in mm_like:
            mm_like = f"%{mm_like}%"
        where.append(f'CAST(f.{_qident("Model")} AS TEXT) ILIKE :model')
        params["model"] = mm_like

    if method:
        mt = method.strip()
        mt_like = mt.replace("*", "%")
        if "%" not in mt_like:
            mt_like = f"%{mt_like}%"
        where.append(f'CAST(f.{_qident("Method")} AS TEXT) ILIKE :method')
        params["method"] = mt_like

    where_sql = " AND ".join(where)
    cols_sql = ", ".join([f'f.{_qident(c)} AS {_qident(c)}' for c in FORECAST_COLS])

    sql_count = text(f"""
      SELECT COUNT(*) AS n
      FROM {view_qual} f
      WHERE {where_sql};
    """)

    sql_rows = text(f"""
      SELECT {cols_sql}
      FROM {view_qual} f
      WHERE {where_sql}
      ORDER BY f.{_qident("StartDate")} DESC
      LIMIT :limit OFFSET :offset;
    """)

    try:
        with ENGINE.begin() as conn:
            total = int(conn.execute(sql_count, params).scalar_one())
            rows = conn.execute(sql_rows, params).mappings().all()
        return {"variant": v, "field": f, "term": t, "count": total, "rows": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/forecast/search failed: {e}")


# -------------------------
# FORECAST RUNNERS (unchanged)
# -------------------------
@app.get("/forecast/jobs")
def list_jobs():
    return [{"level": j[0], "period": j[1], "horizon": j[2]} for j in forecast.JOBS]

@app.post("/forecast/run-one-db")
def run_one_db(req: RunOneDBRequest):
    try:
        global ENGINE
        if ENGINE is None:
            ENGINE = get_engine()

        schema = req.db_schema.strip()
        period = req.period.strip()        # must be "Daily"/"Weekly"/"Monthly"
        horizon = int(req.horizon)

        if period not in forecast.GRAIN_CFG:
            raise HTTPException(status_code=400, detail=f"Invalid period: {period}")

        if req.history_table:
            # If user provides an explicit history table, we still need a LEVEL to choose the output forecast tables
            # Best: infer from history table name history_{level}_{period}
            hist_table = req.history_table.strip()

            m = re.match(rf"^{HISTORY_PREFIX}_(\d+)_({period.lower()})$", hist_table.lower())
            if not m:
                raise HTTPException(
                    status_code=400,
                    detail="When using history_table, it must look like history_<level>_<daily|weekly|monthly> so we can infer level."
                )
            level = m.group(1)

            tag = req.tag or hist_table
        else:
            if not req.level:
                raise HTTPException(status_code=400, detail="Provide either history_table or level.")
            level = str(req.level).strip()
            hist_table = _history_table_name(level, period)
            tag = req.tag or f"{level}_{period}"

        hist_df = _normalize_required_history_cols(_read_table_df(schema, hist_table), period)
        weather_df = _normalize_weather_cols(_read_table_df(schema, req.weather_table.strip()))
        promo_df = _normalize_promo_cols(_read_table_df(schema, req.promo_table.strip()))

        result = forecast.run_one_job_df(
            hist_df=hist_df,
            period=period,
            horizon=horizon,
            weather_daily=weather_df,
            promos=promo_df,
            tag=tag,
            db_engine=ENGINE,
            db_schema=schema,
            level=level,
            write_to_db=True,
            return_frames=False,
        )

        return {
            "ok": True,
            "message": f"Ran {schema}.{hist_table} -> wrote forecasts to forecast_{level}_{period.lower()} (+ _baseline)",
            "tag": tag,
            "db_result": result.get("db", {}),
            "rows_baseline": result.get("rows_baseline"),
            "rows_feat": result.get("rows_feat"),
            "backtest_rows": result.get("backtest_rows"),
            "mean_wmape_base": result.get("mean_wmape_base"),
            "mean_wmape_feat": result.get("mean_wmape_feat"),
        }

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Run failed: {e}\n\n{tb}")


@app.post("/forecast/run-all-db")
def run_all_db(req: RunAllDBRequest):
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

    schema = req.db_schema.strip()
    weather_table = req.weather_table.strip()
    promo_table = req.promo_table.strip()

    # shared exog (read once)
    weather_df = _normalize_weather_cols(_read_table_df(schema, weather_table))
    promo_df = _normalize_promo_cols(_read_table_df(schema, promo_table))

    results = []
    for level, period, horizon in forecast.JOBS:
        hist_table = _history_table_name(level, period)   # history_<level>_<daily|weekly|monthly>
        tag = f"{level}_{period}"                         # purely for logging / return payload

        try:
            hist_df = _normalize_required_history_cols(_read_table_df(schema, hist_table), period)

            run_result = forecast.run_one_job_df(
                hist_df=hist_df,
                period=period,            # "Daily" / "Weekly" / "Monthly"
                horizon=horizon,
                weather_daily=weather_df,
                promos=promo_df,
                tag=tag,

                # ✅ new args for DB-first forecast.py
                db_engine=ENGINE,
                db_schema=schema,
                level=level,
                write_to_db=True,
                return_frames=False,
            )

            results.append({
                "level": level,
                "period": period,
                "horizon": horizon,
                "ok": True,
                "tag": tag,

                # ✅ DB write summary (shape depends on your forecast.py implementation)
                "db": run_result.get("db", {}),
                "rows_baseline": run_result.get("rows_baseline"),
                "rows_feat": run_result.get("rows_feat"),
                "backtest_rows": run_result.get("backtest_rows"),
                "mean_wmape_base": run_result.get("mean_wmape_base"),
                "mean_wmape_feat": run_result.get("mean_wmape_feat"),
            })

        except Exception as e:
            results.append({
                "level": level,
                "period": period,
                "horizon": horizon,
                "ok": False,
                "tag": tag,
                "error": str(e),
            })

    # No OUT_DIR / outputs anymore (DB-first)
    return {"ok": True, "db_schema": schema, "results": results}




@app.get("/forecast/files")
def list_files():
    files = []
    for p in sorted(OUT_DIR.glob("*")):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            files.append({"name": p.name, "size_bytes": p.stat().st_size, "modified": int(p.stat().st_mtime)})
    return {"out_dir": str(OUT_DIR), "files": files}


@app.get("/forecast/file/{filename}")
def download_file(filename: str):
    filename = _safe_filename(filename)
    p = OUT_DIR / filename
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    if p.suffix.lower() not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail="File type not allowed.")
    return FileResponse(str(p), filename=p.name)

# -------------------------
# KPI (History vs Forecast) API
# -------------------------
from math import sqrt

class KpiRequest(BaseModel):
    ProductID: str
    ChannelID: str
    LocationID: str

@app.get("/api/kpi/single")
def api_kpi_single(
    variant: str = Query("feat", description="baseline | feat"),
    productid: str = Query(...),
    channelid: str = Query(...),
    locationid: str = Query(...),
    period: str = Query(..., description="Daily/Weekly/Monthly"),
    model: Optional[str] = Query(None),
    method: Optional[str] = Query(None),
    limit: int = Query(5000, ge=1, le=50000),
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    """
    Join v_history with v_forecast_* on keys+StartDate+Period and compute KPIs.
    """
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

    v = (variant or "feat").strip().lower()
    view_name = FORECAST_BASELINE_VIEW if v in ("baseline", "base") else FORECAST_FEAT_VIEW

    hist_qual = _qualified(db_schema, HISTORY_VIEW)
    fc_qual = _qualified(db_schema, view_name)

    where_fc = [
        'f."ProductID" = :p',
        'f."ChannelID" = :c',
        'f."LocationID" = :l',
        'f."Period" = :period',
    ]
    params: Dict[str, object] = {
        "p": productid,
        "c": channelid,
        "l": locationid,
        "period": period,
        "limit": limit,
    }
    if model:
        where_fc.append('f."Model" = :model')
        params["model"] = model.strip()
    if method:
        where_fc.append('f."Method" = :method')
        params["method"] = method.strip()

    where_fc_sql = " AND ".join(where_fc)

    sql = text(f"""
      WITH fc AS (
        SELECT
          f."ProductID", f."ChannelID", f."LocationID",
          f."StartDate"::date AS "StartDate",
          f."Period",
          f."ForecastQty"::double precision AS "ForecastQty"
        FROM {fc_qual} f
        WHERE {where_fc_sql}
        ORDER BY f."StartDate" ASC
        LIMIT :limit
      ),
      hist AS (
        SELECT
          h."ProductID", h."ChannelID", h."LocationID",
          h."StartDate"::date AS "StartDate",
          h."Period",
          h."Qty"::double precision AS "Qty"
        FROM {hist_qual} h
        WHERE h."ProductID" = :p
          AND h."ChannelID" = :c
          AND h."LocationID" = :l
          AND h."Period" = :period
      ),
      j AS (
        SELECT
          fc."StartDate",
          hist."Qty" AS "A",
          fc."ForecastQty" AS "F",
          (fc."ForecastQty" - hist."Qty") AS "E"
        FROM fc
        JOIN hist
          ON hist."StartDate" = fc."StartDate"
      )
      SELECT
        COUNT(*)::int AS n,
        COALESCE(SUM(ABS("E")),0) AS sae,
        COALESCE(SUM(ABS("A")),0) AS saa,
        COALESCE(SUM(ABS("E")) / NULLIF(SUM(ABS("A")),0) * 100.0, NULL) AS wape,
        COALESCE(AVG(ABS("E")), NULL) AS mae,
        COALESCE(SQRT(AVG(("E")*("E"))), NULL) AS rmse,
        COALESCE(AVG(
          CASE WHEN ABS("A") > 0 THEN ABS("E") / ABS("A") END
        ) * 100.0, NULL) AS mape,
        COALESCE(AVG(
          CASE WHEN (ABS("A")+ABS("F")) > 0 THEN (2*ABS("E")) / (ABS("A")+ABS("F")) END
        ) * 100.0, NULL) AS smape,
        COALESCE(SUM("E") / NULLIF(SUM("A"),0) * 100.0, NULL) AS bias_pct
      FROM j;
    """)

    try:
        with ENGINE.begin() as conn:
            row = conn.execute(sql, params).mappings().first()
        if not row:
            return {"n": 0, "metrics": None}
        out = dict(row)
        return {"n": out.pop("n", 0), "metrics": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/kpi/single failed: {e}")


@app.get("/api/kpi/by-query")
def api_kpi_by_query(
    q: str = Query(...),
    variant: str = Query("feat"),
    period: str = Query(..., description="Daily/Weekly/Monthly"),
    model: Optional[str] = Query(None),
    method: Optional[str] = Query(None),
    limit_keys: int = Query(200, ge=1, le=5000),
    limit_fc_per_key: int = Query(5000, ge=10, le=50000),
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    """
    q -> keys using the SAME tables as /api/search, then compute a single aggregated KPI over all matched pairs.
    """
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

    # reuse your query parser join logic:
    where_sql, params = _build_where_from_query(q)

    trip_qual = _qualified(db_schema, TRIPLET_TABLE)
    prod_qual = _qualified(db_schema, PRODUCT_TABLE)
    chan_qual = _qualified(db_schema, CHANNEL_TABLE)
    loc_qual  = _qualified(db_schema, LOCATION_TABLE)

    keys_sql = text(f"""
      SELECT DISTINCT
        t."ProductID"  AS "ProductID",
        t."ChannelID"  AS "ChannelID",
        t."LocationID" AS "LocationID"
      FROM {trip_qual} t
      JOIN {prod_qual} p ON p."ProductID" = t."ProductID"
      JOIN {chan_qual} c ON c."ChannelID" = t."ChannelID"
      JOIN {loc_qual}  l ON l."LocationID" = t."LocationID"
      WHERE {where_sql}
      ORDER BY 1,2,3
      LIMIT :limit_keys;
    """)

    params_keys = dict(params)
    params_keys["limit_keys"] = int(limit_keys)

    try:
        with ENGINE.begin() as conn:
            keys = conn.execute(keys_sql, params_keys).mappings().all()
        keys = [dict(k) for k in keys]
        if not keys:
            return {"q": q, "keys": 0, "n": 0, "metrics": None}

        # Compute KPI by summing across keys (loop -> single SQL per key, safe & easy)
        # If you want extreme performance later, we can do one big VALUES join like history-by-keys.
        agg = {
            "pairs": 0,
            "sae": 0.0,
            "saa": 0.0,
            "sum_e": 0.0,
            "sum_abs_e_over_a": 0.0,
            "cnt_mape": 0,
            "sum_smape": 0.0,
            "cnt_smape": 0,
            "sum_sq_e": 0.0,
            "sum_abs_e": 0.0,
        }

        for k in keys:
            res = api_kpi_single(
                variant=variant,
                productid=k["ProductID"],
                channelid=k["ChannelID"],
                locationid=k["LocationID"],
                period=period,
                model=model,
                method=method,
                limit=limit_fc_per_key,
                db_schema=db_schema,
            )
            m = res.get("metrics")
            n = res.get("n", 0)
            if not m or n <= 0:
                continue

            # We have aggregate-ready parts:
            # sae = SUM(|E|), saa = SUM(|A|), rmse needs sum_sq_e, mae needs sum_abs_e & n
            agg["pairs"] += n
            agg["sae"] += float(m.get("sae") or 0)
            agg["saa"] += float(m.get("saa") or 0)
            agg["sum_abs_e"] += float(m.get("sae") or 0)  # same
            # rmse proxy: need sum_sq_e; not returned currently.
            # Keep rmse/mae from combined formula approximations? We'll compute only WAPE/BIAS/sMAPE/MAPE/MAE/RMSE later in single SQL if needed.
            # For now: WAPE/Bias stable. We'll return WAPE/Bias only for query page unless you want full.
            agg["sum_e"] += (float(m.get("bias_pct") or 0) * 0)  # placeholder

        if agg["pairs"] <= 0:
            return {"q": q, "keys": len(keys), "n": 0, "metrics": None}

        wape = (agg["sae"] / agg["saa"] * 100.0) if agg["saa"] else None

        return {
            "q": q,
            "keys": len(keys),
            "n": agg["pairs"],
            "metrics": {
                "WAPE": wape,
                "SAE": agg["sae"],
                "SAA": agg["saa"],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/kpi/by-query failed: {e}")



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
