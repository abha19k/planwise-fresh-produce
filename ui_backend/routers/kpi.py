from __future__ import annotations

from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy import text
from services.auth_service import require_roles
from core.db import ENGINE, get_engine, _qualified
from core.config import (
    DEFAULT_SCHEMA,
    FORECAST_BASELINE_VIEW,
    FORECAST_FEAT_VIEW,
    HISTORY_VIEW,
    TRIPLET_TABLE,
    PRODUCT_TABLE,
    CHANNEL_TABLE,
    LOCATION_TABLE,
)
from services.search_service import _build_where_from_query

router = APIRouter()


def _ensure_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()


@router.get("/api/kpi/summary")
def api_kpi_summary(
    variant: str = Query("feat", description="baseline | feat"),
    period: str = Query("Weekly"),
    level: str = Query("111"),
    lag: int = Query(1, ge=0, le=52),
    scenario_id: int = Query(1, ge=1),
    db_schema: str = Query(DEFAULT_SCHEMA),
    current_user=Depends(require_roles("admin", "planner", "viewer")),
    productid: str | None = Query(None),
    channelid: str | None = Query(None),
    locationid: str | None = Query(None),
):
    """
    Overall KPI summary for the dashboard.

    Uses the same lag-alignment principle as the KPI page:

        forecast date -> shift backwards by lag
        -> compare against historical actual

    Example:
        Weekly, lag=1
        forecast 2026-08-10
        compares with actual 2026-08-03
    """

    _ensure_engine()

    v = (variant or "feat").strip().lower()
    p = (period or "Weekly").strip()

    view_name = (
        FORECAST_BASELINE_VIEW
        if v in ("baseline", "base")
        else FORECAST_FEAT_VIEW
    )

    hist_qual = _qualified(db_schema, HISTORY_VIEW)
    fc_qual = _qualified(db_schema, view_name)

    # ---------------------------------------------------------
    # Build the same date shift used by the Angular KPI page
    # ---------------------------------------------------------

    if p.lower() == "daily":
        hist_date_expr = (
            f'(f."StartDate"::date - '
            f"(:lag * INTERVAL '1 day'))::date"
        )

    elif p.lower() == "weekly":
        hist_date_expr = (
            f'(f."StartDate"::date - '
            f"(:lag * INTERVAL '7 day'))::date"
        )

    elif p.lower() == "monthly":
        hist_date_expr = (
            f'(f."StartDate"::date - '
            f"(:lag * INTERVAL '1 month'))::date"
        )

    else:
        raise HTTPException(
            status_code=400,
            detail="period must be Daily, Weekly or Monthly"
        )

    sql = text(f"""
        WITH joined AS (
            SELECT
                h."Qty"::double precision AS actual,
                f."ForecastQty"::double precision AS forecast,

                (
                    f."ForecastQty"::double precision
                    - h."Qty"::double precision
                ) AS error

            FROM {fc_qual} f

            JOIN {hist_qual} h
              ON h."ProductID" = f."ProductID"
             AND h."ChannelID" = f."ChannelID"
             AND h."LocationID" = f."LocationID"

             AND h."StartDate"::date = {hist_date_expr}

             AND LOWER(TRIM(h."Period"))
                 = LOWER(TRIM(f."Period"))

            WHERE f.scenario_id = :scenario_id

              AND LOWER(TRIM(f."Period"))
                  = LOWER(TRIM(:period))

              AND CAST(f."Level" AS TEXT) = :level

              AND (
                :productid IS NULL
                OR f."ProductID" = :productid
              )

              AND (
                :channelid IS NULL
                OR f."ChannelID" = :channelid
              )

              AND (
                :locationid IS NULL
                OR f."LocationID" = :locationid
              )
        )

        SELECT
            COUNT(*)::int AS n,

            COALESCE(
                SUM(ABS(error)),
                0
            ) AS sae,

            COALESCE(
                SUM(ABS(actual)),
                0
            ) AS saa,

            COALESCE(
                SUM(ABS(error))
                / NULLIF(SUM(ABS(actual)), 0)
                * 100.0,
                NULL
            ) AS wape,

            COALESCE(
                SUM(error)
                / NULLIF(SUM(ABS(actual)), 0)
                * 100.0,
                NULL
            ) AS bias_pct

        FROM joined;
    """)

    try:

        with ENGINE.begin() as conn:
            row = conn.execute(
                sql,
                {
                    "scenario_id": int(scenario_id),
                    "period": p,
                    "level": str(level),
                    "lag": int(lag),

                    "productid": productid,
                    "channelid": channelid,
                    "locationid": locationid,
                }
            ).mappings().first()

        if not row:
            return {
                "scenario_id": int(scenario_id),
                "period": p,
                "level": level,
                "n": 0,
                "wape": None,
                "accuracy": None,
                "bias_pct": None,
                "lag": int(lag),
                "productid": productid,
                "channelid": channelid,
                "locationid": locationid,

            }

        n = int(row["n"] or 0)

        wape = (
            float(row["wape"])
            if row["wape"] is not None
            else None
        )

        bias = (
            float(row["bias_pct"])
            if row["bias_pct"] is not None
            else None
        )

        accuracy = (
            max(0.0, 100.0 - wape)
            if wape is not None
            else None
        )

        return {
            "scenario_id": int(scenario_id),
            "period": p,
            "level": level,
            "lag": int(lag),
            "productid": productid,
            "channelid": channelid,
            "locationid": locationid,

            "n": n,

            "sae": float(row["sae"] or 0),
            "saa": float(row["saa"] or 0),

            "wape": wape,
            "accuracy": accuracy,
            "bias_pct": bias,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"/api/kpi/summary failed: {e}"
        )

@router.get("/api/kpi/dashboard-series")
def api_kpi_dashboard_series(
    variant: str = Query("feat", description="baseline | feat"),
    period: str = Query("Weekly"),
    level: str = Query("111"),
    lag: int = Query(1, ge=0, le=52),
    scenario_id: int = Query(1, ge=1),
    limit: int = Query(16, ge=4, le=52),
    db_schema: str = Query(DEFAULT_SCHEMA),
    current_user=Depends(require_roles("admin", "planner", "viewer")),
):
    """
    Aggregate Actual vs Forecast series for the dashboard.

    Uses the same lag alignment as the dashboard KPI summary.
    """

    _ensure_engine()

    v = (variant or "feat").strip().lower()
    p = (period or "Weekly").strip()

    view_name = (
        FORECAST_BASELINE_VIEW
        if v in ("baseline", "base")
        else FORECAST_FEAT_VIEW
    )

    hist_qual = _qualified(db_schema, HISTORY_VIEW)
    fc_qual = _qualified(db_schema, view_name)

    if p.lower() == "daily":
        hist_date_expr = (
            f'(f."StartDate"::date - '
            f"(:lag * INTERVAL '1 day'))::date"
        )

    elif p.lower() == "weekly":
        hist_date_expr = (
            f'(f."StartDate"::date - '
            f"(:lag * INTERVAL '7 day'))::date"
        )

    elif p.lower() == "monthly":
        hist_date_expr = (
            f'(f."StartDate"::date - '
            f"(:lag * INTERVAL '1 month'))::date"
        )

    else:
        raise HTTPException(
            status_code=400,
            detail="period must be Daily, Weekly or Monthly"
        )

    sql = text(f"""
        WITH matched AS (
            SELECT
                f."StartDate"::date AS forecast_date,
                h."Qty"::double precision AS actual,
                f."ForecastQty"::double precision AS forecast

            FROM {fc_qual} f

            JOIN {hist_qual} h
              ON h."ProductID" = f."ProductID"
             AND h."ChannelID" = f."ChannelID"
             AND h."LocationID" = f."LocationID"

             AND h."StartDate"::date = {hist_date_expr}

             AND LOWER(TRIM(h."Period"))
                 = LOWER(TRIM(f."Period"))

            WHERE f.scenario_id = :scenario_id

              AND LOWER(TRIM(f."Period"))
                  = LOWER(TRIM(:period))

              AND CAST(f."Level" AS TEXT) = :level
        ),

        aggregated AS (
            SELECT
                forecast_date AS date,
                SUM(actual) AS actual,
                SUM(forecast) AS forecast
            FROM matched
            GROUP BY forecast_date
        ),

        latest AS (
            SELECT
                date,
                actual,
                forecast
            FROM aggregated
            ORDER BY date DESC
            LIMIT :limit
        )

        SELECT
            date,
            actual,
            forecast
        FROM latest
        ORDER BY date ASC;
    """)

    try:

        with ENGINE.begin() as conn:
            rows = conn.execute(
                sql,
                {
                    "scenario_id": int(scenario_id),
                    "period": p,
                    "level": str(level),
                    "lag": int(lag),
                    "limit": int(limit),
                }
            ).mappings().all()

        return {
            "scenario_id": int(scenario_id),
            "period": p,
            "level": level,
            "lag": int(lag),
            "count": len(rows),
            "rows": [
                {
                    "date": str(r["date"]),
                    "actual": float(r["actual"] or 0),
                    "forecast": float(r["forecast"] or 0),
                }
                for r in rows
            ],
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"/api/kpi/dashboard-series failed: {e}"
        )

@router.get("/api/kpi/single")
def api_kpi_single(
    variant: str = Query("feat", description="baseline | feat"),
    productid: str = Query(...),
    channelid: str = Query(...),
    locationid: str = Query(...),
    period: str = Query(..., description="Daily/Weekly/Monthly"),
    model: Optional[str] = Query(None),
    method: Optional[str] = Query(None),
    scenario_id: int = Query(1, ge=1),
    limit: int = Query(5000, ge=1, le=50000),
    db_schema: str = Query(DEFAULT_SCHEMA),
    current_user=Depends(require_roles("admin", "planner", "viewer")),
):
    _ensure_engine()

    v = (variant or "feat").strip().lower()
    view_name = FORECAST_BASELINE_VIEW if v in ("baseline", "base") else FORECAST_FEAT_VIEW

    hist_qual = _qualified(db_schema, HISTORY_VIEW)
    fc_qual = _qualified(db_schema, view_name)

    where_fc = [
        'f.scenario_id = :scenario_id',
        'f."ProductID" = :p',
        'f."ChannelID" = :c',
        'f."LocationID" = :l',
        'f."Period" = :period',
    ]
    params: Dict[str, object] = {
        "scenario_id": int(scenario_id),
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
            return {"scenario_id": int(scenario_id), "n": 0, "metrics": None}
        out = dict(row)
        return {"scenario_id": int(scenario_id), "n": out.pop("n", 0), "metrics": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/kpi/single failed: {e}")


@router.get("/api/kpi/by-query")
def api_kpi_by_query(
    q: str = Query(...),
    variant: str = Query("feat"),
    period: str = Query(..., description="Daily/Weekly/Monthly"),
    model: Optional[str] = Query(None),
    method: Optional[str] = Query(None),
    limit_keys: int = Query(200, ge=1, le=5000),
    limit_fc_per_key: int = Query(5000, ge=10, le=50000),
    scenario_id: int = Query(1),
    db_schema: str = Query(DEFAULT_SCHEMA),
    current_user=Depends(require_roles("admin", "planner", "viewer")),
):
    _ensure_engine()

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
            return {"q": q, "scenario_id": int(scenario_id), "keys": 0, "n": 0, "metrics": None}

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
                scenario_id=scenario_id,
                limit=limit_fc_per_key,
                db_schema=db_schema,
            )

            m = res.get("metrics")
            n = res.get("n", 0)
            if not m or n <= 0:
                continue

            agg["pairs"] += n
            agg["sae"] += float(m.get("sae") or 0)
            agg["saa"] += float(m.get("saa") or 0)
            agg["sum_abs_e"] += float(m.get("sae") or 0)
            agg["sum_e"] += (float(m.get("bias_pct") or 0) * 0)

        if agg["pairs"] <= 0:
            return {"q": q, "keys": len(keys), "n": 0, "metrics": None}

        wape = (agg["sae"] / agg["saa"] * 100.0) if agg["saa"] else None

        return {
            "q": q,
            "scenario_id": int(scenario_id),
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