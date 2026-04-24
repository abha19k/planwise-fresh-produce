from __future__ import annotations

import os
import re
import traceback
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from sqlalchemy import text
from services.auth_service import require_roles
import forecast

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

        promo_df = _scenario_promotions_df(
            db_schema=schema,
            scenario_id=req.scenario_id,
            table_name=req.promo_table.strip(),
        )
        promo_df = _normalize_promo_cols(promo_df)

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
            scenario_id=req.scenario_id,
            write_to_db=req.save_to_db,
            return_frames=(not req.save_to_db),
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
            feat_df = result.get("forecast_feat_df")
            base_df = result.get("forecast_baseline_df")

            response["forecast_feat_rows"] = feat_df.to_dict(orient="records") if feat_df is not None else []
            response["forecast_baseline_rows"] = base_df.to_dict(orient="records") if base_df is not None else []

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
                scenario_id=req.scenario_id,
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