from __future__ import annotations

import os
import re
import traceback
from typing import Dict, Optional
import pandas as pd

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from sqlalchemy import text
from services.auth_service import require_roles
import forecast
from services.audit_service import write_audit_log
import json

from core.db import ENGINE, get_engine, _qident, _qualified
from core.config import (
    DEFAULT_SCHEMA,
    OUT_DIR,
    ALLOWED_EXTS,
    HISTORY_PREFIX,
    DEFAULT_WEATHER_TABLE,
    DEFAULT_PROMO_TABLE,
    FORECAST_BASELINE_VIEW,
    FORECAST_FEAT_VIEW,
    FORECAST_COLS,
    FORECAST_FIELDS,
)
from models.schemas import RunOneDBRequest, RunAllDBRequest
from services.data_service import (
    _history_table_name,
    _normalize_weather_cols,
    _normalize_promo_cols,
    _scenario_weather_df,
    _scenario_promotions_df,
    _get_history_with_fallback,
)

router = APIRouter()


def _ensure_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()

def _load_future_weather_df(
    db_schema: str,
    table_name: str = "weather_forecast_daily",
):
    """
    Load future weather forecasts.

    This is intentionally separate from historical weather_daily so
    forecast weather cannot accidentally enter model training/backtesting.
    """
    _ensure_engine()

    table_qual = _qualified(db_schema, table_name)

    sql = text(f"""
        SELECT
            "LocationID",
            "Date",
            "TavgC",
            "TminC",
            "TmaxC",
            "PrecipMM",
            "WindMaxMS",
            "SunHours",
            "Provider",
            "Model",
            "ForecastRun",
            "LoadedAt"
        FROM {table_qual}
        ORDER BY "LocationID", "Date";
    """)

    with ENGINE.begin() as conn:
        rows = conn.execute(sql).mappings().all()

    import pandas as pd

    if not rows:
        return pd.DataFrame(
            columns=[
                "LocationID",
                "Date",
                "TavgC",
                "TminC",
                "TmaxC",
                "PrecipMM",
                "WindMaxMS",
                "SunHours",
                "Provider",
                "Model",
                "ForecastRun",
                "LoadedAt",
            ]
        )

    return pd.DataFrame([dict(r) for r in rows])


def _safe_filename(name: str) -> str:
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    return name


@router.get("/api/forecast")
def api_forecast(
    variant: str = Query("baseline", description="baseline | feat"),
    productid: str = Query(...),
    channelid: str = Query(...),
    locationid: str = Query(...),
    period: str = Query(..., description="Daily/Weekly/Monthly"),
    model: Optional[str] = Query(None),
    method: Optional[str] = Query(None),
    scenario_id: int = Query(1, ge=1),
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    db_schema: str = Query(DEFAULT_SCHEMA),
    current_user=Depends(require_roles("admin", "planner", "viewer")),
):
    _ensure_engine()

    v = (variant or "baseline").strip().lower()
    view_name = FORECAST_BASELINE_VIEW if v in ("baseline", "base") else FORECAST_FEAT_VIEW
    view_qual = _qualified(db_schema, view_name)

    cols_sql = ", ".join([f'f.{_qident(c)} AS {_qident(c)}' for c in FORECAST_COLS])

    where = [
        'f.scenario_id = :scenario_id',
        f'f.{_qident("ProductID")} = :p',
        f'f.{_qident("ChannelID")} = :c',
        f'f.{_qident("LocationID")} = :l',
        f'LOWER(TRIM(f.{_qident("Period")})) = LOWER(TRIM(:period))',
    ]
    params: Dict[str, object] = {
        "scenario_id": int(scenario_id),
        "p": productid,
        "c": channelid,
        "l": locationid,
        "period": period,
        "limit": limit,
        "offset": offset,
    }

    if model:
        where.append(f'f.{_qident("Model")} = :model')
        params["model"] = model.strip()

    if method:
        where.append(f'f.{_qident("Method")} = :method')
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
        return {
            "variant": v,
            "scenario_id": int(scenario_id),
            "count": total,
            "rows": [dict(r) for r in rows]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/forecast failed: {e}")

@router.get("/api/forecast/backtest-selected")
def api_forecast_backtest_selected(
    productid: str = Query(...),
    channelid: str = Query(...),
    locationid: str = Query(...),
    period: str = Query("Weekly"),
    level: str = Query("111"),
    variant: str = Query("feat", description="baseline | feat"),
    scenario_id: int = Query(1, ge=1),
    db_schema: str = Query(DEFAULT_SCHEMA),
    current_user=Depends(require_roles("admin", "planner", "viewer")),
):
    """
    Backtest ONE selected forecast element.

    Does not write forecasts to DB.
    Does not affect normal Run Forecast / Save Forecast.
    """

    try:
        _ensure_engine()

        p = (period or "").strip()
        lvl = str(level).strip()
        v = (variant or "feat").strip().lower()

        if p not in forecast.GRAIN_CFG:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid period: {p}"
            )

        if v not in ("baseline", "base", "feat"):
            raise HTTPException(
                status_code=400,
                detail="variant must be baseline or feat"
            )

        # --------------------------------------------------
        # Load scenario history
        # --------------------------------------------------

        hist_df = _get_history_with_fallback(
            db_schema=db_schema,
            scenario_id=scenario_id,
            level=lvl,
            period=p,
        )

        if hist_df is None or hist_df.empty:
            raise HTTPException(
                status_code=404,
                detail="No history found."
            )

        # --------------------------------------------------
        # Weather
        # --------------------------------------------------

        weather_df = _scenario_weather_df(
            db_schema=db_schema,
            scenario_id=scenario_id,
            table_name=DEFAULT_WEATHER_TABLE,
        )

        weather_df = _normalize_weather_cols(
            weather_df
        )

        # --------------------------------------------------
        # Promotions
        # --------------------------------------------------

        promo_df = _scenario_promotions_df(
            db_schema=db_schema,
            scenario_id=scenario_id,
            table_name=DEFAULT_PROMO_TABLE,
        )

        promo_df = _normalize_promo_cols(
            promo_df
        )

        # --------------------------------------------------
        # Prepare history exactly as forecast.py expects
        # --------------------------------------------------

        df = hist_df.copy()

        for c in forecast.KEY_COLS:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
            )

        df["Qty"] = (
            forecast.to_num(df["Qty"])
            .fillna(0.0)
        )

        if "NetPrice" not in df.columns:
            df["NetPrice"] = None

        if "ListPrice" not in df.columns:
            df["ListPrice"] = None

        df["NetPrice"] = forecast.to_num(
            df["NetPrice"]
        )

        df["ListPrice"] = forecast.to_num(
            df["ListPrice"]
        )

        if "UOM" not in df.columns:
            df["UOM"] = "KG"

        df["UOM"] = (
            df["UOM"]
            .astype(str)
            .fillna("KG")
        )

        df["StartDate_dt"] = forecast.parse_date(
            df["StartDate"]
        )

        df["Period"] = (
            df["Period"]
            .astype(str)
            .str.strip()
        )

        df = df[
            (df["Period"] == p)
            & df["StartDate_dt"].notna()
        ].copy()

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail="No history after period filtering."
            )

        df = df.sort_values(
            forecast.KEY_COLS + ["StartDate_dt"]
        )

        df["NetPrice"] = (
            df.groupby(forecast.KEY_COLS)["NetPrice"]
            .ffill()
            .bfill()
        )

        df["ListPrice"] = (
            df.groupby(forecast.KEY_COLS)["ListPrice"]
            .ffill()
            .bfill()
        )

        # --------------------------------------------------
        # Build exogenous inputs
        # --------------------------------------------------

        weather_hist_agg, clim_map = (
            forecast.build_weather_climatology(
                weather_df,
                p,
            )
        )

        promo_daily = (
            forecast.expand_promotions_to_daily(
                promo_df
            )
        )

        promo_agg = (
            forecast.aggregate_promotions_to_period(
                promo_daily,
                p,
            )
        )

        promo_clim = (
            forecast.build_promo_climatology(
                promo_agg,
                p,
            )
        )

        df_exog = forecast.attach_exog_to_history(
            df,
            weather_hist_agg,
            promo_agg,
        )

        # --------------------------------------------------
        # Run selected-key backtest
        # --------------------------------------------------

        result = forecast.rolling_backtest_selected_key(
            df_exog=df_exog,
            period=p,
            forecast_key=(
                productid,
                channelid,
                locationid,
            ),
            use_exog=(v == "feat"),
            weather_hist_agg=weather_hist_agg,
            clim_map=clim_map,
            promo_agg=promo_agg,
            promo_clim=promo_clim,
        )

        # --------------------------------------------------
        # Save successful backtest summary
        # --------------------------------------------------

        if result.get("ok"):
            insert_sql = text(f"""
                INSERT INTO {_qualified(db_schema, "forecast_backtest_result")} (
                    scenario_id,
                    level,
                    period,
                    product_id,
                    channel_id,
                    location_id,
                    variant,
                    accuracy,
                    wape,
                    bias_pct,
                    n,
                    folds
                )
                VALUES (
                    :scenario_id,
                    :level,
                    :period,
                    :product_id,
                    :channel_id,
                    :location_id,
                    :variant,
                    :accuracy,
                    :wape,
                    :bias_pct,
                    :n,
                    :folds
                );
            """)

            with ENGINE.begin() as conn:
                conn.execute(
                    insert_sql,
                    {
                        "scenario_id": int(scenario_id),
                        "level": lvl,
                        "period": p,
                        "product_id": productid,
                        "channel_id": channelid,
                        "location_id": locationid,
                        "variant": v,
                        "accuracy": result.get("accuracy"),
                        "wape": result.get("wape"),
                        "bias_pct": result.get("bias_pct"),
                        "n": int(result.get("n") or 0),
                        "folds": int(result.get("folds") or 0),
                    },
                )

        return {
            "scenario_id": scenario_id,
            "level": lvl,
            "variant": v,
            **result,
        }

    except HTTPException:
        raise

    except Exception as e:
        tb = traceback.format_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                f"Selected-key backtest failed: {e}\n\n{tb}"
            ),
        )
    
@router.get("/api/forecast/backtest-latest")
def api_forecast_backtest_latest(
    productid: str = Query(...),
    channelid: str = Query(...),
    locationid: str = Query(...),
    period: str = Query("Weekly"),
    level: str = Query("111"),
    variant: str = Query("feat", description="baseline | feat"),
    scenario_id: int = Query(1, ge=1),
    db_schema: str = Query(DEFAULT_SCHEMA),
    current_user=Depends(require_roles("admin", "planner", "viewer")),
):
    """
    Return the most recently saved backtest result.

    This endpoint does NOT run a backtest.
    It only reads the latest stored result.
    """

    try:
        _ensure_engine()

        v = (variant or "feat").strip().lower()
        p = (period or "").strip()
        lvl = str(level).strip()

        sql = text(f"""
            SELECT
                scenario_id,
                level,
                period,
                product_id,
                channel_id,
                location_id,
                variant,
                accuracy,
                wape,
                bias_pct,
                n,
                folds,
                run_at
            FROM {_qualified(db_schema, "forecast_backtest_result")}
            WHERE scenario_id = :scenario_id
              AND level = :level
              AND period = :period
              AND product_id = :product_id
              AND channel_id = :channel_id
              AND location_id = :location_id
              AND variant = :variant
            ORDER BY run_at DESC
            LIMIT 1;
        """)

        params = {
            "scenario_id": int(scenario_id),
            "level": lvl,
            "period": p,
            "product_id": productid,
            "channel_id": channelid,
            "location_id": locationid,
            "variant": v,
        }

        with ENGINE.begin() as conn:
            row = conn.execute(sql, params).mappings().first()

        if row is None:
            return {
                "ok": False,
                "scenario_id": int(scenario_id),
                "level": lvl,
                "period": p,
                "variant": v,
                "reason": "No saved backtest result found.",
            }

        return {
            "ok": True,
            **dict(row),
        }

    except Exception as e:
        tb = traceback.format_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Latest backtest lookup failed: {e}\n\n{tb}",
        )

@router.get("/api/forecast/search")
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
    scenario_id: int = Query(1, ge=1),
    db_schema: str = Query(DEFAULT_SCHEMA),
    current_user=Depends(require_roles("admin", "planner", "viewer")),
):
    _ensure_engine()

    v = (variant or "baseline").strip().lower()
    view_name = FORECAST_BASELINE_VIEW if v in ("baseline", "base") else FORECAST_FEAT_VIEW
    view_qual = _qualified(db_schema, view_name)

    f = (field or "").strip()
    if f not in FORECAST_FIELDS:
        raise HTTPException(status_code=400, detail=f"field must be one of {sorted(FORECAST_FIELDS)}")

    t = (term or "").strip()
    if not t:
        return {"variant": v, "scenario_id": int(scenario_id), "field": f, "term": "", "count": 0, "rows": []}

    like = t.replace("*", "%")
    if "%" not in like:
        like = f"%{like}%"

    where = [
        'f.scenario_id = :scenario_id',
        f'CAST(f.{_qident(f)} AS TEXT) ILIKE :like'
    ]
    params: Dict[str, object] = {
        "scenario_id": int(scenario_id),
        "like": like,
        "limit": limit,
        "offset": offset
    }

    if period:
        where.append(f'LOWER(TRIM(f.{_qident("Period")})) = LOWER(TRIM(:period))')
        params["period"] = period

    if level:
        where.append(f'LOWER(TRIM(f.{_qident("Level")})) = LOWER(TRIM(:level))')
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
        return {
            "variant": v,
            "scenario_id": int(scenario_id),
            "field": f,
            "term": t,
            "count": total,
            "rows": [dict(r) for r in rows]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/forecast/search failed: {e}")


@router.get("/forecast/jobs")
def list_jobs(
    current_user=Depends(require_roles("admin", "planner", "viewer")),
):
    return [{"level": j[0], "period": j[1], "horizon": j[2]} for j in forecast.JOBS]


@router.post("/forecast/run-one-db")
def run_one_db(
    req: RunOneDBRequest,
    current_user=Depends(require_roles("admin", "planner")),
):
    try:
        _ensure_engine()

        schema = req.db_schema.strip()
        print("DEBUG run_one_db scenario_id =", req.scenario_id)

        period = req.period.strip()
        horizon = int(req.horizon)

        if period not in forecast.GRAIN_CFG:
            raise HTTPException(status_code=400, detail=f"Invalid period: {period}")

        if req.history_table:
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

        hist_df = _get_history_with_fallback(
            db_schema=schema,
            scenario_id=req.scenario_id,
            level=level,
            period=period,
        )

        weather_df = _scenario_weather_df(
            db_schema=schema,
            scenario_id=req.scenario_id,
            table_name=req.weather_table.strip(),
        )
        weather_df = _normalize_weather_cols(weather_df)

        # Future weather forecast
        future_weather_df = _load_future_weather_df(
            db_schema=schema,
            table_name="weather_forecast_daily",
        )

        promo_df = _scenario_promotions_df(
            db_schema=schema,
            scenario_id=req.scenario_id,
            table_name=req.promo_table.strip(),
        )
        promo_df = _normalize_promo_cols(promo_df)

        forecast_key = None

        if (
            req.product_id
            and req.channel_id
            and req.location_id
        ):
            forecast_key = (
                req.product_id,
                req.channel_id,
                req.location_id,
            )

        result = forecast.run_one_job_df(
            hist_df=hist_df,
            period=period,
            horizon=horizon,
            weather_daily=weather_df,
            promos=promo_df,
            tag=tag,

            weather_future_daily=future_weather_df,

            db_engine=ENGINE,
            db_schema=schema,
            level=level,
            scenario_id=req.scenario_id,
            write_to_db=req.save_to_db,

            run_backtest=False,

            # Interactive Run Forecast:
            # don't waste time training the baseline model.
            run_baseline=req.save_to_db,

            # Global training, selected-series forecasting
            forecast_key=forecast_key,

            return_frames=(not req.save_to_db),
        )


        write_audit_log(
            action="forecast.run_one",
            entity="forecast",
            user_id=str(current_user["id"]),
            entity_id=f"{level}_{period}_scenario_{req.scenario_id}",
            details={
                "scenario_id": req.scenario_id,
                "level": level,
                "period": period,
                "horizon": horizon,
                "save_to_db": req.save_to_db,
            },
            db_schema=schema,
        )

        response = {
            "ok": True,
            "scenario_id": req.scenario_id,
            "message": f"Ran {schema}.{hist_table} -> wrote forecasts to forecast_{level}_{period.lower()} (+ _baseline)" if req.save_to_db
               else f"Ran {schema}.{hist_table} -> forecast generated in memory",
            "tag": tag,
            "db_result": result.get("db", {}),
            "rows_baseline": result.get("rows_baseline"),
            "rows_feat": result.get("rows_feat"),
            "backtest_rows": result.get("backtest_rows"),
            "mean_wmape_base": result.get("mean_wmape_base"),
            "mean_wmape_feat": result.get("mean_wmape_feat"),
        }

        if not req.save_to_db:
            feat_df = result.get("fc_feat")
            base_df = result.get("fc_base")

            response["forecast_feat_rows"] = (
                json.loads(
                    feat_df.to_json(
                        orient="records",
                        date_format="iso"
                    )
                )
                if feat_df is not None and not feat_df.empty
                else []
            )

            response["forecast_baseline_rows"] = (
                json.loads(
                    base_df.to_json(
                        orient="records",
                        date_format="iso"
                    )
                )
                if base_df is not None and not base_df.empty
                else []
            )



        return response

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Run failed: {e}\n\n{tb}")


@router.post("/forecast/run-all-db")
def run_all_db(
    req: RunAllDBRequest,
    current_user=Depends(require_roles("admin", "planner")),
):
    _ensure_engine()

    schema = req.db_schema.strip()
    weather_table = req.weather_table.strip()
    promo_table = req.promo_table.strip()

    weather_df = _scenario_weather_df(
        db_schema=schema,
        scenario_id=req.scenario_id,
        table_name=weather_table,
    )
    weather_df = _normalize_weather_cols(weather_df)

    future_weather_df = _load_future_weather_df(
        db_schema=schema,
        table_name="weather_forecast_daily",
    )

    promo_df = _scenario_promotions_df(
        db_schema=schema,
        scenario_id=req.scenario_id,
        table_name=promo_table,
    )
    promo_df = _normalize_promo_cols(promo_df)

    results = []

    for level, period, horizon in forecast.JOBS:
        hist_table = _history_table_name(level, period)
        tag = f"{level}_{period}"

        try:
            hist_df = _get_history_with_fallback(
                db_schema=schema,
                scenario_id=req.scenario_id,
                level=level,
                period=period,
            )

            run_result = forecast.run_one_job_df(
                hist_df=hist_df,
                period=period,
                horizon=horizon,
                weather_daily=weather_df,
                promos=promo_df,
                tag=tag,

                weather_future_daily=future_weather_df,

                scenario_id=req.scenario_id,
                db_engine=ENGINE,
                db_schema=schema,
                level=level,
                write_to_db=True,
                run_backtest=False,
                return_frames=False,
            )



            results.append({
                "level": level,
                "period": period,
                "horizon": horizon,
                "ok": True,
                "tag": tag,
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
    
    write_audit_log(
        action="forecast.run_all",
        entity="forecast",
        user_id=str(current_user["id"]),
        entity_id=f"scenario_{req.scenario_id}",
        details={
            "scenario_id": req.scenario_id,
            "results_count": len(results),
        },
        db_schema=schema,
    )

    return {"ok": True, "db_schema": schema, "results": results}


@router.get("/forecast/files")
def list_files(
    current_user=Depends(require_roles("admin", "planner", "viewer")),
):
    files = []
    for p in sorted(OUT_DIR.glob("*")):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            files.append({
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "modified": int(p.stat().st_mtime),
            })
    return {"out_dir": str(OUT_DIR), "files": files}


@router.get("/forecast/file/{filename}")
def download_file(
    filename: str,
    current_user=Depends(require_roles("admin", "planner", "viewer")),
):
    filename = _safe_filename(filename)
    p = OUT_DIR / filename
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    if p.suffix.lower() not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail="File type not allowed.")
    return FileResponse(str(p), filename=p.name)