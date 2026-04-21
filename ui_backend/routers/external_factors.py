from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any, Dict, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from core.db import ENGINE, get_engine, _qualified
from core.config import (
    DEFAULT_SCHEMA,
    DEFAULT_WEATHER_TABLE,
    DEFAULT_PROMO_TABLE,
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