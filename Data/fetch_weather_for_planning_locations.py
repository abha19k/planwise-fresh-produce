from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import requests


# -----------------------------
# CONFIG
# -----------------------------
LOCATION_CSV = "Location.csv"
OUTPUT_CSV = "weather_daily.csv"
TIMEZONE = "Europe/Amsterdam"

# Open-Meteo Historical Weather API
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
    "precipitation_sum",
    "wind_speed_10m_max",
    "sunshine_duration",
]

SLEEP_BETWEEN_CALLS_SEC = 0.4

# Stable coordinate mapping (representative city per province/region)
# You can refine later (multiple points + average).
REGION_COORDS: Dict[str, Tuple[float, float]] = {
    "NL-NH": (52.3676, 4.9041),   # Amsterdam (Noord-Holland)
    "NL-ZH": (51.9225, 4.4792),   # Rotterdam (Zuid-Holland)
    "NL-NB": (51.4416, 5.4697),   # Eindhoven (Noord-Brabant)
    "NL-GE": (51.9851, 5.8987),   # Arnhem (Gelderland)
}


def last_3_years() -> tuple[str, str]:
    """Return (start_date, end_date) for the last 3*365 days, ending yesterday."""
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=365 * 3)
    return start.isoformat(), end.isoformat()


def fetch_daily_weather(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARS),
        "timezone": TIMEZONE,
    }
    r = requests.get(HISTORICAL_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    daily = data.get("daily")
    if not daily or "time" not in daily:
        raise ValueError(f"No 'daily' data returned for lat={lat}, lon={lon}")

    df = pd.DataFrame({"Date": daily["time"]})

    mapping = {
        "temperature_2m_mean": "TavgC",
        "temperature_2m_min": "TminC",
        "temperature_2m_max": "TmaxC",
        "precipitation_sum": "PrecipMM",
        "wind_speed_10m_max": "WindMaxMS",
        "sunshine_duration": "SunshineSeconds",
    }

    for api_name, out_name in mapping.items():
        df[out_name] = daily.get(api_name, [None] * len(df))

    return df


def main():
    start_date, end_date = last_3_years()
    print(f"[INFO] Fetching daily weather from {start_date} to {end_date}")

    loc = pd.read_csv(LOCATION_CSV, dtype=str).fillna("")
    loc["Level"] = pd.to_numeric(loc["Level"], errors="coerce")

    regions = loc.loc[loc["Level"] == 1, ["LocationID", "LocationDescr"]].copy()
    if regions.empty:
        raise ValueError("No Level=1 locations found in Location.csv")

    parts: List[pd.DataFrame] = []

    for _, r in regions.iterrows():
        lid = r["LocationID"].strip()
        if lid not in REGION_COORDS:
            raise ValueError(
                f"No coordinates configured for LocationID={lid}. "
                f"Add it to REGION_COORDS mapping."
            )

        lat, lon = REGION_COORDS[lid]
        print(f"[INFO] {lid} ({r['LocationDescr']}) -> lat={lat:.4f}, lon={lon:.4f}")

        dfw = fetch_daily_weather(lat, lon, start_date, end_date)
        dfw.insert(0, "LocationID", lid)
        parts.append(dfw)

        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    out = pd.concat(parts, ignore_index=True)

    out["Date"] = pd.to_datetime(out["Date"])
    for c in ["TavgC", "TminC", "TmaxC", "PrecipMM", "WindMaxMS", "SunshineSeconds"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["SunHours"] = out["SunshineSeconds"] / 3600.0

    out = out[["LocationID", "Date", "TavgC", "TminC", "TmaxC", "PrecipMM", "WindMaxMS", "SunHours"]]
    out = out.sort_values(["LocationID", "Date"])

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"[DONE] Wrote {OUTPUT_CSV} with {len(out):,} rows")


if __name__ == "__main__":
    main()

