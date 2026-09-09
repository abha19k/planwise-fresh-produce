from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

LOCATION_CSV = "Location.csv"
OUTPUT_CSV = "weather_forecast_daily.csv"

TIMEZONE = "Europe/Amsterdam"

# Open-Meteo Forecast API
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Generic forecast API supports up to 16 days
FORECAST_DAYS = 16

DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
    "precipitation_sum",
    "wind_speed_10m_max",
    "sunshine_duration",
]

SLEEP_BETWEEN_CALLS_SEC = 0.4


# ============================================================
# LOCATION COORDINATES
# Same mapping as historical weather script
# ============================================================

REGION_COORDS: Dict[str, Tuple[float, float]] = {
    "NL-NH": (52.3676, 4.9041),   # Amsterdam / Noord-Holland
    "NL-ZH": (51.9225, 4.4792),   # Rotterdam / Zuid-Holland
    "NL-NB": (51.4416, 5.4697),   # Eindhoven / Noord-Brabant
    "NL-GE": (51.9851, 5.8987),   # Arnhem / Gelderland
}


# ============================================================
# FETCH WEATHER
# ============================================================

def fetch_future_weather(
    lat: float,
    lon: float,
) -> tuple[pd.DataFrame, str]:

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(DAILY_VARS),
        "timezone": TIMEZONE,
        "forecast_days": FORECAST_DAYS,

        # Return wind speed in m/s so it matches our DB column
        "wind_speed_unit": "ms",
    }

    response = requests.get(
        FORECAST_URL,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    daily = data.get("daily")

    if not daily or "time" not in daily:
        raise ValueError(
            f"No daily forecast returned "
            f"for lat={lat}, lon={lon}"
        )

    df = pd.DataFrame({
        "Date": daily["time"]
    })

    mapping = {
        "temperature_2m_mean": "TavgC",
        "temperature_2m_min": "TminC",
        "temperature_2m_max": "TmaxC",
        "precipitation_sum": "PrecipMM",
        "wind_speed_10m_max": "WindMaxMS",
        "sunshine_duration": "SunshineSeconds",
    }

    for api_name, output_name in mapping.items():

        values = daily.get(
            api_name,
            [None] * len(df)
        )

        df[output_name] = values

    # Generic endpoint automatically chooses suitable weather models.
    model_name = "best_match"

    return df, model_name


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        f"[INFO] Fetching next {FORECAST_DAYS} days "
        f"of weather forecasts"
    )

    # --------------------------------------------------------
    # Read planning locations
    # --------------------------------------------------------

    loc = (
        pd.read_csv(
            LOCATION_CSV,
            dtype=str,
        )
        .fillna("")
    )

    loc["Level"] = pd.to_numeric(
        loc["Level"],
        errors="coerce",
    )

    regions = loc.loc[
        loc["Level"] == 1,
        [
            "LocationID",
            "LocationDescr",
        ],
    ].copy()

    if regions.empty:
        raise ValueError(
            "No Level=1 locations found in Location.csv"
        )

    parts: List[pd.DataFrame] = []

    forecast_run = datetime.now()

    # --------------------------------------------------------
    # Fetch each region
    # --------------------------------------------------------

    for _, row in regions.iterrows():

        location_id = (
            row["LocationID"]
            .strip()
        )

        if location_id not in REGION_COORDS:

            raise ValueError(
                f"No coordinates configured for "
                f"LocationID={location_id}. "
                f"Add it to REGION_COORDS mapping."
            )

        lat, lon = REGION_COORDS[
            location_id
        ]

        print(
            f"[INFO] {location_id} "
            f"({row['LocationDescr']}) "
            f"-> lat={lat:.4f}, "
            f"lon={lon:.4f}"
        )

        df_weather, model_name = (
            fetch_future_weather(
                lat,
                lon,
            )
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        df_weather.insert(
            0,
            "LocationID",
            location_id,
        )

        df_weather["Provider"] = "Open-Meteo"
        df_weather["Model"] = model_name
        df_weather["ForecastRun"] = forecast_run

        parts.append(
            df_weather
        )

        time.sleep(
            SLEEP_BETWEEN_CALLS_SEC
        )

    # --------------------------------------------------------
    # Combine locations
    # --------------------------------------------------------

    out = pd.concat(
        parts,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    out["Date"] = pd.to_datetime(
        out["Date"]
    )

    numeric_cols = [
        "TavgC",
        "TminC",
        "TmaxC",
        "PrecipMM",
        "WindMaxMS",
        "SunshineSeconds",
    ]

    for col in numeric_cols:

        out[col] = pd.to_numeric(
            out[col],
            errors="coerce",
        )

    # Open-Meteo returns sunshine duration in seconds.
    out["SunHours"] = (
        out["SunshineSeconds"]
        / 3600.0
    )

    # --------------------------------------------------------
    # Final DB-compatible columns
    # --------------------------------------------------------

    out = out[
        [
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
        ]
    ]

    out = out.sort_values(
        [
            "LocationID",
            "Date",
        ]
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    out.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print(
        f"[DONE] Wrote {OUTPUT_CSV} "
        f"with {len(out):,} rows"
    )

    print(
        f"[INFO] Locations: "
        f"{out['LocationID'].nunique()}"
    )

    print(
        f"[INFO] Date range: "
        f"{out['Date'].min().date()} "
        f"to "
        f"{out['Date'].max().date()}"
    )


if __name__ == "__main__":
    main()