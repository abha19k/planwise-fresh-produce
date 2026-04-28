from __future__ import annotations

import os
from pathlib import Path

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

OUT_DIR = Path(
    os.getenv("FORECAST_OUT_DIR", str(BASE_DIR / "outputs"))
).resolve()

OUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".csv", ".png", ".txt", ".log"}

JWT_SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
    "DEV_ONLY_CHANGE_ME_LONG_SECRET"
)

# ------------------------------------------------------------
# DATABASE
# ------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DEFAULT_SCHEMA = os.getenv("DB_SCHEMA", "planwise_fresh_produce")

# ------------------------------------------------------------
# TABLE NAMES
# ------------------------------------------------------------
PRODUCT_TABLE = os.getenv("PRODUCT_TABLE", "product")
CHANNEL_TABLE = os.getenv("CHANNEL_TABLE", "channel")
LOCATION_TABLE = os.getenv("LOCATION_TABLE", "location")
TRIPLET_TABLE = os.getenv("TRIPLET_TABLE", "forecastelement")

DEFAULT_WEATHER_TABLE = os.getenv("WEATHER_TABLE", "weather_daily")
DEFAULT_PROMO_TABLE = os.getenv("PROMO_TABLE", "promotions")

FORECASTELEMENT_TABLE = os.getenv(
    "FORECASTELEMENT_TABLE",
    "forecastelement"
)

HISTORY_PREFIX = os.getenv("HISTORY_PREFIX", "history")

HISTORY_VIEW = os.getenv("HISTORY_VIEW", "v_history")

FORECAST_BASELINE_VIEW = os.getenv(
    "FORECAST_BASELINE_VIEW",
    "v_forecast_baseline"
)

FORECAST_FEAT_VIEW = os.getenv(
    "FORECAST_FEAT_VIEW",
    "v_forecast_feat"
)

CLEANSE_PROFILES_TABLE = os.getenv(
    "CLEANSE_PROFILES_TABLE",
    "cleanse_profiles"
)

CLEANSED_HISTORY_TABLE = os.getenv(
    "CLEANSED_HISTORY_TABLE",
    "history_cleansed"
)

SCENARIO_TABLE = os.getenv("SCENARIO_TABLE", "scenario")

SCENARIO_OVERRIDE_TABLE = os.getenv(
    "SCENARIO_OVERRIDE_TABLE",
    "scenario_override"
)

CLASSIFIED_FE_TABLE = "classified_forecast_elements"

# ------------------------------------------------------------
# COLUMN LISTS
# ------------------------------------------------------------
PRODUCT_COLS = [
    "ProductID",
    "ProductDescr",
    "Level",
    "BusinessUnit",
    "IsDailyForecastRequired",
    "IsNew",
    "ProductFamily",
]

CHANNEL_COLS = [
    "ChannelID",
    "ChannelDescr",
    "Level",
]

LOCATION_COLS = [
    "LocationID",
    "LocationDescr",
    "Level",
    "IsActive",
]

FORECASTELEMENT_COLS = [
    "ProductID",
    "ChannelID",
    "LocationID",
    "Level",
    "IsActive",
]

FORECAST_COLS = [
    "Level",
    "Model",
    "ProductID",
    "ChannelID",
    "LocationID",
    "StartDate",
    "EndDate",
    "Period",
    "ForecastQty",
    "UOM",
    "NetPrice",
    "ListPrice",
    "Method",
]

CLEANSED_HISTORY_COLS = [
    "ProductID",
    "ChannelID",
    "LocationID",
    "StartDate",
    "EndDate",
    "Period",
    "Qty",
    "NetPrice",
    "ListPrice",
    "Level",
]

HISTORY_COLS = [
    "Level",
    "ProductID",
    "ChannelID",
    "LocationID",
    "StartDate",
    "EndDate",
    "Period",
    "Qty",
    "NetPrice",
    "ListPrice",
    "Type",
]

# ------------------------------------------------------------
# ALLOWED SEARCH FIELDS
# ------------------------------------------------------------
FORECAST_FIELDS = {
    "ProductID",
    "ChannelID",
    "LocationID",
}

HISTORY_FIELDS = {
    "ProductID",
    "ChannelID",
    "LocationID",
}