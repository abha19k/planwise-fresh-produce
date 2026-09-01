from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any, Dict, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy import text
from services.auth_service import require_roles
from core.db import ENGINE, get_engine, _qualified
from core.config import (
    DEFAULT_SCHEMA,
    DEFAULT_WEATHER_TABLE,
    DEFAULT_PROMO_TABLE,
    HISTORY_VIEW,
)
from services.scenario_service import (
    _get_base_scenario_id,
    _get_scenario_chain,
    _json_key,
    _read_overrides_for_chain,
)

router = APIRouter()


def _ensure_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()


@router.get("/api/weather_daily")
def api_weather_daily(
    db_schema: str = Query(DEFAULT_SCHEMA),
    scenario_id: int = Query(0, description="Scenario id; 0 = base"),
    locationid: str = Query("", description="optional"),
    date_from: str = Query(..., description="YYYY-MM-DD"),
    date_to: str = Query(..., description="YYYY-MM-DD"),
    current_user=Depends(require_roles("admin", "planner", "viewer")),
):
    _ensure_engine()

    table_name = DEFAULT_WEATHER_TABLE
    base_qual = _qualified(db_schema, table_name)

    where = [
        'w."Date"::date >= CAST(:date_from AS date)',
        'w."Date"::date <= CAST(:date_to AS date)',
    ]
    params: Dict[str, Any] = {"date_from": date_from, "date_to": date_to}

    if (locationid or "").strip():
        where.append('w."LocationID" = :locationid')
        params["locationid"] = locationid.strip()

    where_sql = " AND ".join(where)

    sql = text(f"""
      SELECT
        w."LocationID",
        w."Date"::date AS "Date",
        w."TavgC",
        w."TminC",
        w."TmaxC",
        w."PrecipMM",
        w."WindMaxMS",
        w."SunHours"
      FROM {base_qual} w
      WHERE {where_sql}
      ORDER BY w."LocationID", w."Date";
    """)

    with ENGINE.begin() as conn:
        base_rows = conn.execute(sql, params).mappings().all()

    rows = [dict(r) for r in base_rows]

    chain = _get_scenario_chain(db_schema, int(scenario_id))
    ov = _read_overrides_for_chain(db_schema, table_name, chain)

    def pk_key(loc: str, d: str) -> str:
        return _json_key({"LocationID": loc, "Date": str(d)})

    out: Dict[str, Dict[str, Any]] = {
        pk_key(r["LocationID"], r["Date"]): r for r in rows
    }

    for _, o in ov.items():
        pk = dict(o.get("pk") or {})
        loc = pk.get("LocationID")
        d = pk.get("Date")
        if not loc or not d:
            continue

        k = pk_key(loc, d)

        if bool(o.get("is_deleted")):
            out.pop(k, None)
            continue

        rj = dict(o.get("row") or {})
        if not rj:
            continue

        if (locationid or "").strip() and str(rj.get("LocationID", "")).strip() != locationid.strip():
            continue

        ds = str(rj.get("Date", "")).strip()
        if ds < date_from or ds > date_to:
            continue

        out[k] = {
            "LocationID": str(rj.get("LocationID", loc)),
            "Date": ds,
            "TavgC": rj.get("TavgC"),
            "TminC": rj.get("TminC"),
            "TmaxC": rj.get("TmaxC"),
            "PrecipMM": rj.get("PrecipMM"),
            "WindMaxMS": rj.get("WindMaxMS"),
            "SunHours": rj.get("SunHours"),
        }

    base_id = _get_base_scenario_id(db_schema)
    sid = int(scenario_id or 0)
    final_rows = list(out.values())
    final_rows.sort(key=lambda r: (str(r.get("LocationID", "")), str(r.get("Date", ""))))

    return {
        "scenario_id": sid or base_id,
        "count": len(final_rows),
        "rows": final_rows,
    }


@router.get("/api/promotions")
def api_promotions(
    db_schema: str = Query(DEFAULT_SCHEMA),
    scenario_id: int = Query(0, description="Scenario id; 0 = base"),
    productid: str = Query("", description="optional"),
    channelid: str = Query("", description="optional"),
    locationid: str = Query("", description="optional"),
    date_from: str = Query(..., description="YYYY-MM-DD"),
    date_to: str = Query(..., description="YYYY-MM-DD"),
    current_user=Depends(require_roles("admin", "planner", "viewer")),
):
    _ensure_engine()

    table_name = DEFAULT_PROMO_TABLE
    base_qual = _qualified(db_schema, table_name)

    DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    date_from = (date_from or "").strip()
    date_to = (date_to or "").strip()

    if not DATE_RE.match(date_from) or not DATE_RE.match(date_to):
        raise HTTPException(status_code=400, detail="date_from/date_to must be YYYY-MM-DD")

    try:
        df = date.fromisoformat(date_from)
        dt = date.fromisoformat(date_to)
    except Exception:
        raise HTTPException(status_code=400, detail="date_from/date_to must be valid ISO dates YYYY-MM-DD")

    if df > dt:
        raise HTTPException(status_code=400, detail="date_from must be <= date_to")

    where = [
        'p."StartDate" <= CAST(:date_to AS date)',
        'p."EndDate"   >= CAST(:date_from AS date)',
    ]
    params: Dict[str, Any] = {"date_from": date_from, "date_to": date_to}

    productid = (productid or "").strip()
    channelid = (channelid or "").strip()
    locationid = (locationid or "").strip()

    if productid:
        where.append('p."ProductID" = :productid')
        params["productid"] = productid
    if channelid:
        where.append('p."ChannelID" = :channelid')
        params["channelid"] = channelid
    if locationid:
        where.append('p."LocationID" = :locationid')
        params["locationid"] = locationid

    where_sql = " AND ".join(where)

    sql = text(f"""
      SELECT
        p."PromoID",
        p."PromoName",
        p."StartDate"::date AS "StartDate",
        p."EndDate"::date   AS "EndDate",
        p."ProductID",
        p."ChannelID",
        p."LocationID",
        p."PromoLevel",
        p."DiscountPct",
        p."UpliftPct",
        p."Notes"
      FROM {base_qual} p
      WHERE {where_sql}
      ORDER BY p."StartDate", p."PromoID";
    """)

    with ENGINE.begin() as conn:
        base_rows = conn.execute(sql, params).mappings().all()

    base_out = []
    for r in base_rows:
        rr = dict(r)
        rr["StartDate"] = rr["StartDate"].isoformat() if rr.get("StartDate") else None
        rr["EndDate"] = rr["EndDate"].isoformat() if rr.get("EndDate") else None
        base_out.append(rr)

    chain = _get_scenario_chain(db_schema, int(scenario_id))
    ov = _read_overrides_for_chain(db_schema, table_name, chain)

    def pk_key(promoid: str) -> str:
        return _json_key({"PromoID": str(promoid)})

    out: Dict[str, Dict[str, Any]] = {}
    for r in base_out:
        pid = str(r.get("PromoID", "")).strip()
        if pid:
            out[pk_key(pid)] = r

    def _parse_date(s: str) -> Optional[date]:
        s = (s or "").strip()
        if not DATE_RE.match(s):
            return None
        try:
            return date.fromisoformat(s)
        except Exception:
            return None

    for _, o in ov.items():
        pk = dict(o.get("pk") or {})
        promoid = str(pk.get("PromoID", "")).strip()
        if not promoid:
            continue

        k_pk = pk_key(promoid)

        if bool(o.get("is_deleted")):
            out.pop(k_pk, None)
            continue

        rj = dict(o.get("row") or {})
        if not rj:
            continue

        rj_promoid = str(rj.get("PromoID") or promoid).strip()
        if not rj_promoid:
            continue
        k = pk_key(rj_promoid)

        if productid and str(rj.get("ProductID", "")).strip() != productid:
            continue
        if channelid and str(rj.get("ChannelID", "")).strip() != channelid:
            continue
        if locationid and str(rj.get("LocationID", "")).strip() != locationid:
            continue

        sd = _parse_date(str(rj.get("StartDate", "")))
        ed = _parse_date(str(rj.get("EndDate", "")))
        if not sd or not ed:
            continue

        if sd > dt or ed < df:
            continue

        out[k] = {
            "PromoID": rj_promoid,
            "PromoName": rj.get("PromoName"),
            "StartDate": sd.isoformat(),
            "EndDate": ed.isoformat(),
            "ProductID": rj.get("ProductID"),
            "ChannelID": rj.get("ChannelID"),
            "LocationID": rj.get("LocationID"),
            "PromoLevel": rj.get("PromoLevel"),
            "DiscountPct": rj.get("DiscountPct"),
            "UpliftPct": rj.get("UpliftPct"),
            "Notes": rj.get("Notes"),
        }

    final_rows = list(out.values())
    final_rows.sort(key=lambda r: (str(r.get("StartDate") or ""), str(r.get("PromoID") or "")))

    base_id = _get_base_scenario_id(db_schema)
    sid = int(scenario_id or 0)

    return {
        "scenario_id": sid or base_id,
        "count": len(final_rows),
        "rows": final_rows,
    }

# ============================================================
# WEATHER / SALES CORRELATION
# ============================================================

WEATHER_METRIC_MAP = {
    # Frontend key -> current DB column
    "TempAvg": "TavgC",
    "TempMin": "TminC",
    "TempMax": "TmaxC",
    "RainMm": "PrecipMM",
    "SnowCm": None,       # Current weather table has no snowfall column
    "WindMax": "WindMaxMS",
    "SunHours": "SunHours",
}


def _normalise_period(period: str) -> str:
    p = (period or "").strip().lower()

    if p == "daily":
        return "Daily"
    if p == "weekly":
        return "Weekly"
    if p == "monthly":
        return "Monthly"

    raise HTTPException(
        status_code=400,
        detail="period must be Daily, Weekly or Monthly",
    )


def _load_weather_sales_rows(
    db_schema: str,
    product_id: str,
    channel_id: str,
    location_id: str,
    period: str,
    weather_column: str,
):
    """
    Load and align weather + sales history for one
    Product / Channel / Location / Period.

    History already exists at Daily / Weekly / Monthly grain
    in v_history, so Qty is read directly at the requested grain.

    Weather is daily and therefore aggregated to the requested grain.
    """

    _ensure_engine()

    period_ui = _normalise_period(period)

    history_qual = _qualified(db_schema, HISTORY_VIEW)
    weather_qual = _qualified(db_schema, DEFAULT_WEATHER_TABLE)

    params = {
        "product_id": product_id.strip(),
        "channel_id": channel_id.strip(),
        "location_id": location_id.strip(),
        "period": period_ui,
    }

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history_sql = text(
        f"""
        SELECT
            h."StartDate"::date AS "Date",
            SUM(h."Qty")::double precision AS "Qty"
        FROM {history_qual} h
        WHERE TRIM(h."ProductID") = :product_id
          AND TRIM(h."ChannelID") = :channel_id
          AND TRIM(h."LocationID") = :location_id
          AND LOWER(TRIM(h."Period")) = LOWER(:period)
        GROUP BY h."StartDate"::date
        ORDER BY h."StartDate"::date
        """
    )

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    if period_ui == "Daily":

        weather_sql = text(
            f"""
            SELECT
                w."Date"::date AS "Date",
                AVG(w."{weather_column}")::double precision AS "WeatherMetric"
            FROM {weather_qual} w
            WHERE TRIM(w."LocationID") = :location_id
              AND w."{weather_column}" IS NOT NULL
            GROUP BY w."Date"::date
            ORDER BY w."Date"::date
            """
        )

    elif period_ui == "Weekly":

        # PlanWise weekly history uses Monday as StartDate.
        weather_sql = text(
            f"""
            SELECT
                date_trunc('week', w."Date")::date AS "Date",
                AVG(w."{weather_column}")::double precision AS "WeatherMetric"
            FROM {weather_qual} w
            WHERE TRIM(w."LocationID") = :location_id
              AND w."{weather_column}" IS NOT NULL
            GROUP BY date_trunc('week', w."Date")::date
            ORDER BY date_trunc('week', w."Date")::date
            """
        )

    else:  # Monthly

        weather_sql = text(
            f"""
            SELECT
                date_trunc('month', w."Date")::date AS "Date",
                AVG(w."{weather_column}")::double precision AS "WeatherMetric"
            FROM {weather_qual} w
            WHERE TRIM(w."LocationID") = :location_id
              AND w."{weather_column}" IS NOT NULL
            GROUP BY date_trunc('month', w."Date")::date
            ORDER BY date_trunc('month', w."Date")::date
            """
        )

    try:
        with ENGINE.begin() as conn:
            history_rows = conn.execute(
                history_sql,
                params,
            ).mappings().all()

            weather_rows = conn.execute(
                weather_sql,
                params,
            ).mappings().all()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed loading weather/sales data: {e}",
        )

    # --------------------------------------------------------
    # Join in Python
    # --------------------------------------------------------

    history_map = {}

    for r in history_rows:
        d = r["Date"]

        if d is None:
            continue

        qty = r["Qty"]

        if qty is None:
            continue

        history_map[d] = float(qty)

    weather_map = {}

    for r in weather_rows:
        d = r["Date"]

        if d is None:
            continue

        value = r["WeatherMetric"]

        if value is None:
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        if not math.isfinite(value):
            continue

        weather_map[d] = value

    common_dates = sorted(
        set(history_map.keys()) &
        set(weather_map.keys())
    )

    result = []

    for d in common_dates:

        qty = history_map[d]
        weather_value = weather_map[d]

        if not math.isfinite(qty):
            continue

        result.append(
            {
                "date": d.isoformat(),
                "weatherMetric": weather_value,
                "quantity": qty,
            }
        )

    return result


# ============================================================
# WEATHER + SALES SERIES
# ============================================================

@router.get("/api/weather-sales")
def api_weather_sales(
    productId: str = Query(...),
    channelId: str = Query(...),
    locationId: str = Query(...),
    period: str = Query("Daily"),
    metric: str = Query("TempAvg"),
    db_schema: str = Query(DEFAULT_SCHEMA),
    current_user=Depends(
        require_roles("admin", "planner", "viewer")
    ),
):
    """
    Return aligned weather + sales observations for the selected
    Product / Channel / Location / Period.
    """

    metric = (metric or "").strip()

    if metric not in WEATHER_METRIC_MAP:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown weather metric '{metric}'. "
                f"Allowed: {list(WEATHER_METRIC_MAP.keys())}"
            ),
        )

    weather_column = WEATHER_METRIC_MAP[metric]

    # Snowfall doesn't currently exist in weather_daily.
    if weather_column is None:
        return []

    rows = _load_weather_sales_rows(
        db_schema=db_schema,
        product_id=productId,
        channel_id=channelId,
        location_id=locationId,
        period=period,
        weather_column=weather_column,
    )

    return rows


# ============================================================
# WEATHER CORRELATIONS
# ============================================================

@router.get("/api/weather-correlations")
def api_weather_correlations(
    productId: str = Query(...),
    channelId: str = Query(...),
    locationId: str = Query(...),
    period: str = Query("Daily"),
    db_schema: str = Query(DEFAULT_SCHEMA),
    current_user=Depends(
        require_roles("admin", "planner", "viewer")
    ),
):
    """
    Pearson correlation between historical sales and each
    available weather metric.
    """

    result = {
        "TempAvg": None,
        "TempMin": None,
        "TempMax": None,
        "RainMm": None,
        "SnowCm": None,
        "WindMax": None,
        "SunHours": None,
    }

    for frontend_metric, weather_column in WEATHER_METRIC_MAP.items():

        if weather_column is None:
            continue

        rows = _load_weather_sales_rows(
            db_schema=db_schema,
            product_id=productId,
            channel_id=channelId,
            location_id=locationId,
            period=period,
            weather_column=weather_column,
        )

        if len(rows) < 2:
            continue

        x = np.asarray(
            [r["weatherMetric"] for r in rows],
            dtype=float,
        )

        y = np.asarray(
            [r["quantity"] for r in rows],
            dtype=float,
        )

        valid = np.isfinite(x) & np.isfinite(y)

        x = x[valid]
        y = y[valid]

        if len(x) < 2:
            continue

        # Pearson correlation is undefined when either series
        # has zero variance.
        if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
            continue

        corr = float(np.corrcoef(x, y)[0, 1])

        if math.isfinite(corr):
            result[frontend_metric] = corr

    return result