from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(".")
UOM = "KG"
RANDOM_SEED = 42
NOISE_SIGMA = 0.18

OUT_D = "History_111_Daily.csv"
OUT_W = "History_111_Weekly.csv"
OUT_M = "History_111_Monthly.csv"

# Demand knobs
FAMILY_BASE_QTY = {"Apple": 80.0, "Pear": 70.0, "": 75.0}
GROUP_MULT_QTY = {"RETAIL": 1.00, "WHOLESALE": 0.70, "ONLINE": 0.55, "OTHER": 0.60}
DOW_FACTOR_RETAIL = np.array([0.95, 0.98, 1.00, 1.02, 1.05, 1.20, 1.15])
DOW_FACTOR_OTHER  = np.array([1.00, 1.00, 1.00, 1.00, 1.00, 0.90, 0.85])

# Weather factor
WEATHER_TREF = 10.0
WEATHER_A = 0.015
WEATHER_B = 0.020
WEATHER_C = 0.035
MIN_WEATHER_FACTOR = 0.70
MAX_WEATHER_FACTOR = 1.35

# Embedded price + harvest logic
BASE_LIST_PRICE_FAMILY = {"Apple": 2.40, "Pear": 2.80}
HARVEST_WINDOW = {"Apple": (8, 11), "Pear": (8, 10)}
HARV_TO_PRICE = {1: 1.18, 2: 1.08, 3: 1.00, 4: 0.92, 5: 0.85}
DEFAULT_DISCOUNT_IF_MISSING = 10.0


def channel_group(cid: str) -> str:
    c = str(cid).strip().upper()
    if c in {"AH", "JUMBO", "PLUS"}:
        return "RETAIL"
    if c in {"SLIGRO", "HOLLANDFOODSERVICE", "GREENACRES", "MELEDI"}:
        return "WHOLESALE"
    if c in {"PICNIC", "HELLOFRESH", "FLINK"}:
        return "ONLINE"
    return "OTHER"


def seasonal_demand_factor(product_family: str, day_of_year: int) -> float:
    fam = str(product_family).lower()
    if fam.startswith("apple"):
        phase, amp = 280, 0.30
    elif fam.startswith("pear"):
        phase, amp = 240, 0.28
    else:
        phase, amp = 260, 0.25
    return 1.0 + amp * math.sin(2 * math.pi * (day_of_year - phase) / 365.25)


def weather_factor(tavg: float, sun: float, pr: float) -> float:
    f = math.exp(WEATHER_A * (tavg - WEATHER_TREF) + WEATHER_B * sun - WEATHER_C * pr)
    return float(max(MIN_WEATHER_FACTOR, min(MAX_WEATHER_FACTOR, f)))


def harvest_level(family: str, month: int, rng: np.random.Generator) -> int:
    fam = str(family)
    start_m, end_m = HARVEST_WINDOW.get(fam, (8, 10))
    base = 4 if (start_m <= month <= end_m) else 2
    lvl = int(round(base + rng.normal(0, 0.7)))
    return max(1, min(5, lvl))


def seasonal_price_multiplier(family: str, month: int) -> float:
    if family == "Apple":
        return 1.00 if month in [9, 10, 11] else 1.05
    if family == "Pear":
        return 1.00 if month in [8, 9, 10] else 1.06
    return 1.00


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    products = pd.read_csv(BASE_DIR / "Product.csv", dtype=str).fillna("")
    fe = pd.read_csv(BASE_DIR / "ForecastElement.csv", dtype=str).fillna("")
    weather = pd.read_csv(BASE_DIR / "weather_daily.csv", dtype=str).fillna("")
    promos = pd.read_csv(BASE_DIR / "Promotions.csv", dtype=str).fillna("")

    products["Level"] = pd.to_numeric(products["Level"], errors="coerce")
    prod_level = products.set_index("ProductID")["Level"].to_dict()
    prod_family = products.set_index("ProductID")["ProductFamily"].to_dict()

    p1 = products.loc[products["Level"] == 1, ["ProductID", "ProductFamily"]].copy()
    fam_to_p1 = p1.groupby("ProductFamily")["ProductID"].apply(list).to_dict()

    group_to_customers = {
        "RETAIL": ["AH", "JUMBO", "PLUS"],
        "WHOLESALE": ["SLIGRO", "HOLLANDFOODSERVICE", "GREENACRES", "MELEDI"],
        "ONLINE": ["PICNIC", "HELLOFRESH", "FLINK"],
    }

    # ----- Weather prep
    weather["Date"] = pd.to_datetime(weather["Date"])
    for c in ["TavgC", "PrecipMM", "SunHours"]:
        weather[c] = pd.to_numeric(weather[c], errors="coerce")
    weather = weather.sort_values(["LocationID", "Date"])
    weather[["TavgC", "PrecipMM", "SunHours"]] = (
        weather.groupby("LocationID")[["TavgC", "PrecipMM", "SunHours"]]
        .apply(lambda g: g.interpolate(limit_direction="both"))
        .reset_index(level=0, drop=True)
    )
    wsmall = weather[["LocationID", "Date", "TavgC", "PrecipMM", "SunHours"]].copy()
    wsmall["WeatherFactor"] = wsmall.apply(
        lambda r: weather_factor(float(r["TavgC"]), float(r["SunHours"]), float(r["PrecipMM"])),
        axis=1
    )
    wsmall = wsmall[["LocationID", "Date", "WeatherFactor"]]

    dates = pd.date_range(weather["Date"].min(), weather["Date"].max(), freq="D")
    n_days = len(dates)
    dow = dates.dayofweek.values
    doy = dates.dayofyear.values
    months = dates.month.values

    families = sorted(p1["ProductFamily"].unique().tolist())
    demand_season_by_fam = {
        fam: np.array([seasonal_demand_factor(fam, int(x)) for x in doy]) for fam in families
    }

    # ----- Promotions map (factor + discount)
    promos["StartDate"] = pd.to_datetime(promos["StartDate"]).dt.date
    promos["EndDate"] = pd.to_datetime(promos["EndDate"]).dt.date
    promos["UpliftPct"] = pd.to_numeric(promos.get("UpliftPct", 0), errors="coerce").fillna(0.0)
    promos["DiscountPct"] = pd.to_numeric(promos.get("DiscountPct", np.nan), errors="coerce")
    promos["PromoLevel"] = promos["PromoLevel"].astype(str).str.strip().str.upper()

    promo_factor: Dict[Tuple[str, str, str, object], float] = {}
    promo_discount: Dict[Tuple[str, str, str, object], float] = {}

    for _, r in promos.iterrows():
        lvl = r["PromoLevel"]
        sd, ed = r["StartDate"], r["EndDate"]
        pid = str(r["ProductID"]).strip()
        cid = str(r["ChannelID"]).strip().upper()
        lid = str(r["LocationID"]).strip()

        factor = 1.0 + float(r["UpliftPct"]) / 100.0
        disc = r["DiscountPct"]
        if pd.isna(disc):
            disc = max(2.0, min(25.0, DEFAULT_DISCOUNT_IF_MISSING + 0.25 * float(r["UpliftPct"])))

        if lvl == "111":
            target_products = [pid]
            target_channels = [cid]
        elif lvl == "221":
            if prod_level.get(pid) == 2:
                fam = prod_family.get(pid, "")
                target_products = fam_to_p1.get(fam, [])
            else:
                target_products = [pid]
            target_channels = group_to_customers.get(cid, [])
        else:
            continue

        d = sd
        while d <= ed:
            for p_ in target_products:
                for c_ in target_channels:
                    k = (p_, c_, lid, d)
                    promo_factor[k] = factor
                    promo_discount[k] = float(disc)
            d += timedelta(days=1)

    # ----- Forecast elements 111 only
    lvl_col = "Level" if "Level" in fe.columns else ("FLevel" if "FLevel" in fe.columns else None)
    if lvl_col is None:
        raise ValueError("ForecastElement.csv must have 'Level' or 'FLevel' (111/121/221).")

    fe_111 = fe[fe[lvl_col].astype(str).str.strip() == "111"].copy()
    fe_111["ProductID"] = fe_111["ProductID"].astype(str).str.strip()
    fe_111["ChannelID"] = fe_111["ChannelID"].astype(str).str.strip().str.upper()
    fe_111["LocationID"] = fe_111["LocationID"].astype(str).str.strip()

    locs_sorted = sorted(fe_111["LocationID"].unique().tolist())
    region_mult = {lid: (0.95 + 0.03 * i) for i, lid in enumerate(locs_sorted)}

    # ----- Build daily history 111
    parts = []
    for _, row in fe_111.iterrows():
        pid, cid, lid = row["ProductID"], row["ChannelID"], row["LocationID"]
        fam = prod_family.get(pid, "")
        grp = channel_group(cid)

        base = FAMILY_BASE_QTY.get(fam, 75.0) * GROUP_MULT_QTY.get(grp, 0.6) * region_mult.get(lid, 1.0)
        var_mult = 0.85 + (abs(hash(pid)) % 31) / 100.0
        base *= var_mult

        season_vec = demand_season_by_fam.get(fam, np.ones(n_days))
        dow_vec = DOW_FACTOR_RETAIL if grp == "RETAIL" else DOW_FACTOR_OTHER
        noise = rng.lognormal(mean=0.0, sigma=NOISE_SIGMA, size=n_days)

        df = pd.DataFrame({
            "ProductID": pid,
            "ChannelID": cid,
            "LocationID": lid,
            "Date": dates,
            "Qty": base * season_vec * dow_vec[dow] * noise
        })

        df = df.merge(wsmall, how="left", on=["LocationID", "Date"])
        df["WeatherFactor"] = df["WeatherFactor"].fillna(1.0)
        df["Qty"] = df["Qty"] * df["WeatherFactor"]

        # Embedded ListPrice (harvest-driven)
        base_list = BASE_LIST_PRICE_FAMILY.get(fam, 2.60) * region_mult.get(lid, 1.0)
        harvest = np.array([harvest_level(fam, int(m), rng) for m in months], dtype=int)
        list_price = (
            base_list
            * np.array([seasonal_price_multiplier(fam, int(m)) for m in months])
            * np.vectorize(HARV_TO_PRICE.get)(harvest)
        )
        list_price = list_price * rng.normal(1.0, 0.03, size=n_days)
        list_price = np.maximum(0.10, list_price)

        disc_arr = np.zeros(n_days, dtype=float)
        dts = df["Date"].dt.date.values
        for i, d in enumerate(dts):
            disc_arr[i] = promo_discount.get((pid, cid, lid, d), 0.0)

        net_price = np.maximum(0.05, list_price * (1.0 - disc_arr / 100.0))

        # promo uplift to demand
        if promo_factor:
            pf = np.ones(n_days)
            for i, d in enumerate(dts):
                pf[i] = promo_factor.get((pid, cid, lid, d), 1.0)
            df["Qty"] = df["Qty"] * pf

        df["Qty"] = np.round(np.maximum(df["Qty"], 0.0), 2)
        df["ListPrice"] = np.round(list_price, 3)
        df["NetPrice"] = np.round(net_price, 3)
        df["UOM"] = UOM

        parts.append(df[["ProductID","ChannelID","LocationID","Date","Qty","NetPrice","ListPrice","UOM"]])

    daily = pd.concat(parts, ignore_index=True)

    daily["Period"] = "Daily"
    daily["StartDate"] = daily["Date"].dt.date.astype(str)
    daily["EndDate"] = ""
    daily = daily.drop(columns=["Date"])
    daily = daily[["ProductID","ChannelID","LocationID","Qty","Period","StartDate","EndDate","UOM","NetPrice","ListPrice"]]
    daily.to_csv(BASE_DIR / OUT_D, index=False)
    print(f"[DONE] {OUT_D} rows={len(daily):,}")

    # -----------------------------
    # Weekly aggregation (sum qty; weighted avg prices)
    # -----------------------------
    tmp = daily.copy()
    tmp["StartDate_dt"] = pd.to_datetime(tmp["StartDate"])
    iso = tmp["StartDate_dt"].dt.isocalendar()
    tmp["ISOYear"] = iso.year.astype(int)
    tmp["ISOWeek"] = iso.week.astype(int)

    tmp["Qty"] = pd.to_numeric(tmp["Qty"], errors="coerce").fillna(0.0)
    tmp["NetPrice"] = pd.to_numeric(tmp["NetPrice"], errors="coerce")
    tmp["ListPrice"] = pd.to_numeric(tmp["ListPrice"], errors="coerce")

    grp_cols = ["ProductID","ChannelID","LocationID","ISOYear","ISOWeek"]

    weekly = tmp.groupby(grp_cols, as_index=False)["Qty"].sum()

    tmp["NetPrice_x_Qty"] = tmp["NetPrice"] * tmp["Qty"]
    tmp["ListPrice_x_Qty"] = tmp["ListPrice"] * tmp["Qty"]

    num_np = tmp.groupby(grp_cols, as_index=False)["NetPrice_x_Qty"].sum()
    num_lp = tmp.groupby(grp_cols, as_index=False)["ListPrice_x_Qty"].sum()

    weekly = weekly.merge(num_np, on=grp_cols, how="left").merge(num_lp, on=grp_cols, how="left")

    weekly["NetPrice"] = np.where(weekly["Qty"] > 0, weekly["NetPrice_x_Qty"] / weekly["Qty"], np.nan)
    weekly["ListPrice"] = np.where(weekly["Qty"] > 0, weekly["ListPrice_x_Qty"] / weekly["Qty"], np.nan)

    weekly["NetPrice"] = weekly["NetPrice"].round(3)
    weekly["ListPrice"] = weekly["ListPrice"].round(3)

    weekly["StartDate"] = weekly.apply(
        lambda r: pd.Timestamp.fromisocalendar(int(r["ISOYear"]), int(r["ISOWeek"]), 1).date().isoformat(),
        axis=1
    )
    weekly["EndDate"] = weekly.apply(
        lambda r: (pd.Timestamp.fromisocalendar(int(r["ISOYear"]), int(r["ISOWeek"]), 1) + pd.Timedelta(days=6)).date().isoformat(),
        axis=1
    )

    weekly["Period"] = "Weekly"
    weekly["UOM"] = UOM

    weekly = weekly.drop(columns=["NetPrice_x_Qty","ListPrice_x_Qty","ISOYear","ISOWeek"])
    weekly = weekly[["ProductID","ChannelID","LocationID","Qty","Period","StartDate","EndDate","UOM","NetPrice","ListPrice"]]
    weekly.to_csv(BASE_DIR / OUT_W, index=False)
    print(f"[DONE] {OUT_W} rows={len(weekly):,}")

    # -----------------------------
    # Monthly aggregation (sum qty; weighted avg prices)
    # -----------------------------
    tmp2 = daily.copy()
    tmp2["StartDate_dt"] = pd.to_datetime(tmp2["StartDate"])
    tmp2["Month"] = tmp2["StartDate_dt"].dt.to_period("M").astype(str)

    tmp2["Qty"] = pd.to_numeric(tmp2["Qty"], errors="coerce").fillna(0.0)
    tmp2["NetPrice"] = pd.to_numeric(tmp2["NetPrice"], errors="coerce")
    tmp2["ListPrice"] = pd.to_numeric(tmp2["ListPrice"], errors="coerce")

    grp_cols_m = ["ProductID","ChannelID","LocationID","Month"]

    monthly = tmp2.groupby(grp_cols_m, as_index=False)["Qty"].sum()

    tmp2["NetPrice_x_Qty"] = tmp2["NetPrice"] * tmp2["Qty"]
    tmp2["ListPrice_x_Qty"] = tmp2["ListPrice"] * tmp2["Qty"]

    num_np = tmp2.groupby(grp_cols_m, as_index=False)["NetPrice_x_Qty"].sum()
    num_lp = tmp2.groupby(grp_cols_m, as_index=False)["ListPrice_x_Qty"].sum()

    monthly = monthly.merge(num_np, on=grp_cols_m, how="left").merge(num_lp, on=grp_cols_m, how="left")

    monthly["NetPrice"] = np.where(monthly["Qty"] > 0, monthly["NetPrice_x_Qty"] / monthly["Qty"], np.nan)
    monthly["ListPrice"] = np.where(monthly["Qty"] > 0, monthly["ListPrice_x_Qty"] / monthly["Qty"], np.nan)

    monthly["NetPrice"] = monthly["NetPrice"].round(3)
    monthly["ListPrice"] = monthly["ListPrice"].round(3)

    monthly["StartDate"] = pd.to_datetime(monthly["Month"] + "-01").dt.date.astype(str)
    monthly["EndDate"] = (pd.to_datetime(monthly["StartDate"]) + pd.offsets.MonthEnd(0)).dt.date.astype(str)

    monthly["Period"] = "Monthly"
    monthly["UOM"] = UOM

    monthly = monthly.drop(columns=["NetPrice_x_Qty","ListPrice_x_Qty","Month"])
    monthly = monthly[["ProductID","ChannelID","LocationID","Qty","Period","StartDate","EndDate","UOM","NetPrice","ListPrice"]]
    monthly.to_csv(BASE_DIR / OUT_M, index=False)
    print(f"[DONE] {OUT_M} rows={len(monthly):,}")


if __name__ == "__main__":
    main()

