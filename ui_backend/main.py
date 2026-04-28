from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import forecast

from core.config import (
    DEFAULT_SCHEMA,
    PRODUCT_TABLE,
    CHANNEL_TABLE,
    LOCATION_TABLE,
    TRIPLET_TABLE,
    DEFAULT_WEATHER_TABLE,
    DEFAULT_PROMO_TABLE,
    HISTORY_VIEW,
)

from routers.master_data import router as master_data_router
from routers.search import router as search_router
from routers.history import router as history_router
from routers.cleanse import router as cleanse_router
from routers.classify import router as classify_router
from routers.scenarios import router as scenarios_router
from routers.kpi import router as kpi_router
from routers.forecast import router as forecast_router
from routers.external_factors import router as external_factors_router
from routers.auth import router as auth_router
from routers.admin import router as admin_router

app = FastAPI(title="PlanWise API (DB tables)", version="3.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(master_data_router)
app.include_router(search_router)
app.include_router(history_router)
app.include_router(cleanse_router)
app.include_router(classify_router)
app.include_router(scenarios_router)
app.include_router(kpi_router)
app.include_router(forecast_router)
app.include_router(external_factors_router)
app.include_router(auth_router)
app.include_router(admin_router)

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )