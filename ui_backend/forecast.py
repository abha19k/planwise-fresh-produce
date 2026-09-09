# forecast.py
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# optional DB writing
from sqlalchemy import text
from sqlalchemy.engine import Engine


# ============================================================
# CONFIG
# ============================================================
KEY_COLS = ["ProductID", "ChannelID", "LocationID"]

# Logical jobs (main.py decides which DB tables correspond to these)
JOBS = [
    ("111", "Daily", 28),
    ("121", "Daily", 28),
    ("221", "Daily", 28),

    ("111", "Weekly", 13),
    ("121", "Weekly", 13),
    ("221", "Weekly", 13),

    ("111", "Monthly", 12),
    ("121", "Monthly", 12),
    ("221", "Monthly", 12),
]

# Feature settings by grain
GRAIN_CFG = {
    "Daily":   {"freq": "D",     "lags": [1, 7, 14, 28], "rolls": [7, 14, 28], "min_train_points": 120},
    "Weekly":  {"freq": "W-MON", "lags": [1, 4, 13],     "rolls": [4, 13],     "min_train_points": 78},
    "Monthly": {"freq": "MS",    "lags": [1, 3, 6, 12],  "rolls": [3, 6, 12],  "min_train_points": 24},
}

# Backtest settings (rolling origin)
BACKTEST_CFG = {
    "Daily":   {"n_folds": 6, "step": 28, "bt_horizon": 28},
    "Weekly":  {"n_folds": 6, "step": 13, "bt_horizon": 13},
    "Monthly": {"n_folds": 3, "step": 3,  "bt_horizon": 6},
}

# XGB parameters
RANDOM_SEED = 42
XGB_PARAMS = dict(
    n_estimators=450,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=RANDOM_SEED,
    objective="reg:squarederror",
    n_jobs=8,
)


# ============================================================
# HELPERS
# ============================================================
def parse_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def period_to_bucket_start(period: str, dt: pd.Timestamp) -> pd.Timestamp:
    if period == "Daily":
        return pd.Timestamp(dt.date())
    if period == "Weekly":
        return (dt - pd.Timedelta(days=dt.dayofweek)).normalize()  # Monday
    if period == "Monthly":
        return pd.Timestamp(dt.year, dt.month, 1)
    raise ValueError(period)

def next_start(period: str, dt: pd.Timestamp) -> pd.Timestamp:
    if period == "Daily":
        return dt + pd.Timedelta(days=1)
    if period == "Weekly":
        return dt + pd.Timedelta(weeks=1)
    if period == "Monthly":
        return dt + pd.offsets.MonthBegin(1)
    raise ValueError(period)

def make_end_date(period: str, start: pd.Timestamp) -> pd.Timestamp:
    if period == "Daily":
        return start
    if period == "Weekly":
        return start + pd.Timedelta(days=6)
    if period == "Monthly":
        return start + pd.offsets.MonthEnd(0)
    raise ValueError(period)

def calendar_features(period: str, dt: pd.Timestamp) -> Dict[str, float]:
    feats = {"year": float(dt.year), "month": float(dt.month)}
    if period == "Daily":
        feats["dow"] = float(dt.dayofweek)
        feats["doy"] = float(dt.dayofyear)
    elif period == "Weekly":
        feats["weekofyear"] = float(dt.isocalendar().week)
    elif period == "Monthly":
        feats["month"] = float(dt.month)
    return feats

def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


# ============================================================
# WEATHER CLIMATOLOGY
# ============================================================
WEATHER_COLS = ["TavgC", "PrecipMM", "SunHours", "TminC", "TmaxC", "WindMaxMS"]

def build_weather_climatology(
    weather_daily: pd.DataFrame,
    period: str
) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], Dict[str, float]]]:
    w = weather_daily.copy()
    w["Date"] = parse_date(w["Date"])
    w["LocationID"] = w["LocationID"].astype(str).str.strip()

    keep = ["LocationID", "Date"]
    for c in WEATHER_COLS:
        if c in w.columns:
            w[c] = to_num(w[c])
            keep.append(c)
    w = w[keep].dropna(subset=["LocationID", "Date"]).copy()

    w["StartDate_dt"] = w["Date"].apply(lambda d: period_to_bucket_start(period, d))
    weather_hist_agg = w.groupby(["LocationID", "StartDate_dt"], as_index=False).mean(numeric_only=True)

    if period == "Daily":
        w["season_key"] = w["Date"].dt.dayofyear.astype(int)
    elif period == "Weekly":
        w["season_key"] = w["Date"].dt.isocalendar().week.astype(int)
    elif period == "Monthly":
        w["season_key"] = w["Date"].dt.month.astype(int)
    else:
        raise ValueError(period)

    clim = w.groupby(["LocationID", "season_key"], as_index=False).mean(numeric_only=True)

    clim_map: Dict[Tuple[str, int], Dict[str, float]] = {}
    wcols = [c for c in clim.columns if c not in ("LocationID", "season_key")]
    for _, r in clim.iterrows():
        key = (str(r["LocationID"]), int(r["season_key"]))
        clim_map[key] = {c: float(r[c]) for c in wcols if pd.notna(r[c])}

    return weather_hist_agg, clim_map

def aggregate_future_weather(
    weather_forecast_daily: Optional[pd.DataFrame],
    period: str,
) -> pd.DataFrame:
    """
    Aggregate future daily weather forecasts to the forecasting grain.

    Daily   -> one row per day
    Weekly  -> average of available daily forecasts in each Monday-based week
    Monthly -> average of available daily forecasts in each month
    """
    if weather_forecast_daily is None or weather_forecast_daily.empty:
        return pd.DataFrame(
            columns=["LocationID", "StartDate_dt"] + WEATHER_COLS
        )

    w = weather_forecast_daily.copy()

    w["Date"] = parse_date(w["Date"])
    w["LocationID"] = (
        w["LocationID"]
        .astype(str)
        .str.strip()
    )

    for c in WEATHER_COLS:
        if c in w.columns:
            w[c] = to_num(w[c])

    w = w.dropna(
        subset=["LocationID", "Date"]
    ).copy()

    w["StartDate_dt"] = w["Date"].apply(
        lambda d: period_to_bucket_start(
            period,
            pd.Timestamp(d),
        )
    )

    available_cols = [
        c for c in WEATHER_COLS
        if c in w.columns
    ]

    if not available_cols:
        return pd.DataFrame(
            columns=["LocationID", "StartDate_dt"] + WEATHER_COLS
        )

    return (
        w.groupby(
            ["LocationID", "StartDate_dt"],
            as_index=False,
        )[available_cols]
        .mean()
    )

def build_complete_future_weather(
    weather_daily: pd.DataFrame,
    weather_future_daily: Optional[pd.DataFrame],
    period: str,
    location_id: str,
    future_starts: List[pd.Timestamp],
    use_observed_weather: bool = False,
) -> pd.DataFrame:
    """
    Build complete weather features for forecast periods.

    Priority for every individual day:
      1. observed weather_daily
      2. weather_forecast_daily
      3. historical daily climatology
      4. historical location mean
      5. zero

    Daily values are resolved BEFORE aggregation so Weekly and
    Monthly periods never use incomplete weather buckets.
    """

    location_id = str(location_id).strip()

    if not future_starts:
        return pd.DataFrame(
            columns=["LocationID", "StartDate_dt"] + WEATHER_COLS
        )

    # ---------------------------------------------------------
    # Determine every calendar day required by forecast periods
    # ---------------------------------------------------------

    first_day = pd.Timestamp(future_starts[0]).normalize()

    if period == "Daily":
        last_day = pd.Timestamp(future_starts[-1]).normalize()

    elif period == "Weekly":
        last_day = (
            pd.Timestamp(future_starts[-1]).normalize()
            + pd.Timedelta(days=6)
        )

    elif period == "Monthly":
        last_day = (
            pd.Timestamp(future_starts[-1])
            + pd.offsets.MonthEnd(0)
        ).normalize()

    else:
        raise ValueError(period)

    all_dates = pd.date_range(
        start=first_day,
        end=last_day,
        freq="D",
    )

    out = pd.DataFrame({
        "LocationID": location_id,
        "Date": all_dates,
    })

    # ---------------------------------------------------------
    # Historical observed daily weather
    # ---------------------------------------------------------

    hist = weather_daily.copy()

    hist["Date"] = parse_date(hist["Date"]).dt.normalize()
    hist["LocationID"] = (
        hist["LocationID"].astype(str).str.strip()
    )

    hist = hist[
        hist["LocationID"] == location_id
    ].copy()

    for c in WEATHER_COLS:
        if c in hist.columns:
            hist[c] = to_num(hist[c])

    hist_cols = [
        "LocationID",
        "Date",
    ] + [
        c for c in WEATHER_COLS
        if c in hist.columns
    ]

    hist = hist[hist_cols]

    if use_observed_weather:
        out = out.merge(
            hist,
            on=["LocationID", "Date"],
            how="left",
        )

    for c in WEATHER_COLS:
        if c not in out.columns:
            out[c] = np.nan

    # ---------------------------------------------------------
    # Future daily weather
    # ---------------------------------------------------------

    if (
        weather_future_daily is not None
        and not weather_future_daily.empty
    ):
        future = weather_future_daily.copy()

        future["Date"] = (
            parse_date(future["Date"]).dt.normalize()
        )

        future["LocationID"] = (
            future["LocationID"]
            .astype(str)
            .str.strip()
        )

        future = future[
            future["LocationID"] == location_id
        ].copy()

        for c in WEATHER_COLS:
            if c in future.columns:
                future[c] = to_num(future[c])

        future_cols = [
            "LocationID",
            "Date",
        ] + [
            c for c in WEATHER_COLS
            if c in future.columns
        ]

        future = future[future_cols]

        future = future.rename(
            columns={
                c: f"{c}_future"
                for c in WEATHER_COLS
                if c in future.columns
            }
        )

        out = out.merge(
            future,
            on=["LocationID", "Date"],
            how="left",
        )

        for c in WEATHER_COLS:
            fc = f"{c}_future"

            if fc in out.columns:
                out[c] = out[c].fillna(out[fc])
                out.drop(columns=[fc], inplace=True)

    # ---------------------------------------------------------
    # Daily historical climatology
    # ---------------------------------------------------------

    if not hist.empty:

        hist["season_key"] = hist["Date"].dt.dayofyear

        available_cols = [
            c for c in WEATHER_COLS
            if c in hist.columns
        ]

        daily_clim = (
            hist.groupby("season_key")[available_cols]
            .mean()
        )

        location_means = hist[available_cols].mean()

        for i in out.index:

            season_key = int(
                out.loc[i, "Date"].dayofyear
            )

            for c in WEATHER_COLS:

                if pd.notna(out.loc[i, c]):
                    continue

                if (
                    c in daily_clim.columns
                    and season_key in daily_clim.index
                ):
                    value = daily_clim.loc[
                        season_key,
                        c,
                    ]

                    if pd.notna(value):
                        out.loc[i, c] = value
                        continue

                if (
                    c in location_means.index
                    and pd.notna(location_means[c])
                ):
                    out.loc[i, c] = location_means[c]

    for c in WEATHER_COLS:
        out[c] = to_num(out[c]).fillna(0.0)

    # ---------------------------------------------------------
    # NOW aggregate complete daily weather
    # ---------------------------------------------------------

    out["StartDate_dt"] = out["Date"].apply(
        lambda d: period_to_bucket_start(
            period,
            pd.Timestamp(d),
        )
    )

    result = (
        out.groupby(
            ["LocationID", "StartDate_dt"],
            as_index=False,
        )[WEATHER_COLS]
        .mean()
    )

    required_starts = pd.DataFrame({
        "LocationID": location_id,
        "StartDate_dt": future_starts,
    })


    return required_starts.merge(
        result,
        on=["LocationID", "StartDate_dt"],
        how="left",
    )

def weather_features_for_dates(
    period: str,
    location_id: str,
    dates: List[pd.Timestamp],
    weather_hist_agg: pd.DataFrame,
    clim_map: Dict[Tuple[str, int], Dict[str, float]],
    weather_future_agg: Optional[pd.DataFrame] = None,
    use_observed_weather: bool = False,
) -> pd.DataFrame:
    
        location_id = str(location_id).strip()

        tmp = pd.DataFrame({
            "LocationID": [location_id] * len(dates),
            "StartDate_dt": dates,
        })

        out = tmp.copy()

        # =========================================================
        # 1. Observed historical weather for exact forecast dates
        #
        # Used for normal/replay forecasting only.
        # Must remain OFF during rolling backtests to avoid leakage.
        # =========================================================

        if (
            use_observed_weather
            and weather_hist_agg is not None
            and not weather_hist_agg.empty
        ):
            hist_sub = weather_hist_agg[
                weather_hist_agg["LocationID"]
                .astype(str)
                .str.strip()
                == location_id
            ].copy()

            hist_cols = [
                "LocationID",
                "StartDate_dt",
            ] + [
                c for c in WEATHER_COLS
                if c in hist_sub.columns
            ]

            hist_sub = hist_sub[hist_cols]

            out = out.merge(
                hist_sub,
                on=["LocationID", "StartDate_dt"],
                how="left",
            )

        # Make sure every weather column exists
        for c in WEATHER_COLS:
            if c not in out.columns:
                out[c] = np.nan

        # =========================================================
        # 2. Future weather forecast
        # =========================================================

        if (
            weather_future_agg is not None
            and not weather_future_agg.empty
        ):
            future_sub = weather_future_agg[
                weather_future_agg["LocationID"]
                .astype(str)
                .str.strip()
                == location_id
            ].copy()

            future_cols = [
                "LocationID",
                "StartDate_dt",
            ] + [
                c for c in WEATHER_COLS
                if c in future_sub.columns
            ]

            future_sub = future_sub[future_cols]

            future_sub = future_sub.rename(
                columns={
                    c: f"{c}_future"
                    for c in WEATHER_COLS
                    if c in future_sub.columns
                }
            )

            out = out.merge(
                future_sub,
                on=["LocationID", "StartDate_dt"],
                how="left",
            )

            # Observed weather wins.
            # Future forecast fills only missing observed weather.
            for c in WEATHER_COLS:
                future_col = f"{c}_future"

                if future_col in out.columns:
                    out[c] = out[c].fillna(
                        out[future_col]
                    )

                    out.drop(
                        columns=[future_col],
                        inplace=True,
                    )

        # =========================================================
        # 3. Seasonal climatology
        # =========================================================

        if period == "Daily":
            keys = [
                d.dayofyear
                for d in dates
            ]

        elif period == "Weekly":
            keys = [
                int(d.isocalendar().week)
                for d in dates
            ]

        else:
            keys = [
                d.month
                for d in dates
            ]

        for i, season_key in enumerate(keys):

            for c in WEATHER_COLS:

                # Keep observed/future weather when available
                if pd.notna(out.loc[i, c]):
                    continue

                vals = clim_map.get(
                    (location_id, int(season_key)),
                    {},
                )

                if c in vals:
                    out.loc[i, c] = vals[c]

        # =========================================================
        # 4. Final fallback: historical mean
        # =========================================================

        for c in WEATHER_COLS:

            if out[c].isna().any():

                if c in weather_hist_agg.columns:

                    fallback = to_num(
                        weather_hist_agg[c]
                    ).mean()

                    out[c] = out[c].fillna(
                        fallback
                    )

                out[c] = out[c].fillna(0.0)
        

        return out




# ============================================================
# PROMOTIONS
# ============================================================
PROMO_COLS = ["PromoFlag", "PromoDepth", "PromoUplift"]

def expand_promotions_to_daily(promos: pd.DataFrame) -> pd.DataFrame:
    p = promos.copy()
    for c in KEY_COLS:
        if c in p.columns:
            p[c] = p[c].astype(str).str.strip()
    p["StartDate"] = parse_date(p["StartDate"])
    p["EndDate"] = parse_date(p["EndDate"])

    p["DiscountPct"] = to_num(p["DiscountPct"]) if "DiscountPct" in p.columns else 0.0
    p["UpliftPct"]   = to_num(p["UpliftPct"]) if "UpliftPct" in p.columns else 0.0
    p["DiscountPct"] = pd.Series(p["DiscountPct"]).fillna(0.0)
    p["UpliftPct"]   = pd.Series(p["UpliftPct"]).fillna(0.0)

    rows = []
    for _, r in p.iterrows():
        if pd.isna(r["StartDate"]) or pd.isna(r["EndDate"]):
            continue
        pid = str(r.get("ProductID", "")).strip()
        cid = str(r.get("ChannelID", "")).strip()
        lid = str(r.get("LocationID", "")).strip()
        if pid == "" or cid == "" or lid == "":
            continue

        cur = r["StartDate"].normalize()
        edn = r["EndDate"].normalize()
        while cur <= edn:
            rows.append({
                "ProductID": pid,
                "ChannelID": cid,
                "LocationID": lid,
                "Date": cur,
                "PromoFlag": 1.0,
                "PromoDepth": float(r["DiscountPct"]),
                "PromoUplift": float(r["UpliftPct"]),
            })
            cur += pd.Timedelta(days=1)

    if not rows:
        return pd.DataFrame(columns=KEY_COLS + ["Date"] + PROMO_COLS)
    return pd.DataFrame(rows)


def aggregate_promotions_to_period(promo_daily: pd.DataFrame, period: str) -> pd.DataFrame:
    if promo_daily.empty:
        return pd.DataFrame(columns=KEY_COLS + ["StartDate_dt"] + PROMO_COLS)
    p = promo_daily.copy()
    p["StartDate_dt"] = p["Date"].apply(lambda d: period_to_bucket_start(period, d))
    agg = p.groupby(KEY_COLS + ["StartDate_dt"], as_index=False).max(numeric_only=True)
    return agg


def build_promo_climatology(promo_agg: pd.DataFrame, period: str) -> Dict[Tuple[str, str, str, int], Dict[str, float]]:
    if promo_agg.empty:
        return {}

    p = promo_agg.copy()
    p["StartDate_dt"] = pd.to_datetime(p["StartDate_dt"], errors="coerce")
    for c in PROMO_COLS:
        if c in p.columns:
            p[c] = to_num(p[c]).fillna(0.0)
        else:
            p[c] = 0.0

    if period == "Daily":
        p["season_key"] = p["StartDate_dt"].dt.dayofyear.astype(int)
    elif period == "Weekly":
        p["season_key"] = p["StartDate_dt"].dt.isocalendar().week.astype(int)
    else:
        p["season_key"] = p["StartDate_dt"].dt.month.astype(int)

    clim = p.groupby(KEY_COLS + ["season_key"], as_index=False)[PROMO_COLS].mean()

    out: Dict[Tuple[str, str, str, int], Dict[str, float]] = {}
    for _, r in clim.iterrows():
        key = (str(r["ProductID"]), str(r["ChannelID"]), str(r["LocationID"]), int(r["season_key"]))
        out[key] = {c: float(r[c]) for c in PROMO_COLS}
    return out


def promo_features_for_future(
    period: str,
    keys: Tuple[str, str, str],
    future_starts: List[pd.Timestamp],
    promo_agg: pd.DataFrame,
    promo_clim: Dict[Tuple[str, str, str, int], Dict[str, float]],
) -> pd.DataFrame:
    pid, cid, lid = [str(x).strip() for x in keys]
    tmp = pd.DataFrame({"StartDate_dt": future_starts})
    if promo_agg.empty:
        out = tmp.copy()
        for c in PROMO_COLS:
            out[c] = 0.0
    else:
        sub = promo_agg[
            (promo_agg["ProductID"] == pid) &
            (promo_agg["ChannelID"] == cid) &
            (promo_agg["LocationID"] == lid)
        ][["StartDate_dt"] + PROMO_COLS].copy()
        sub["StartDate_dt"] = pd.to_datetime(sub["StartDate_dt"], errors="coerce")
        out = tmp.merge(sub, on="StartDate_dt", how="left")
        for c in PROMO_COLS:
            out[c] = to_num(out[c]).fillna(np.nan)

    if period == "Daily":
        sk = [d.dayofyear for d in future_starts]
    elif period == "Weekly":
        sk = [int(d.isocalendar().week) for d in future_starts]
    else:
        sk = [d.month for d in future_starts]

    for i, k in enumerate(sk):
        if out.loc[i, PROMO_COLS].isna().all():
            vals = promo_clim.get((pid, cid, lid, int(k)))
            if vals:
                for c in PROMO_COLS:
                    out.loc[i, c] = vals.get(c, 0.0)

    for c in PROMO_COLS:
        out[c] = to_num(out[c]).fillna(0.0)

    return out


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def build_supervised_frame(
    df: pd.DataFrame,
    period: str,
    lags: List[int],
    rolls: List[int],
    use_exog: bool,
) -> pd.DataFrame:
    df = df.sort_values("StartDate_dt").copy()
    y = df["Qty"].astype(float)

    netp = df["NetPrice"].astype(float).values
    listp = df["ListPrice"].astype(float).values
    disc = np.where(listp > 0, 1.0 - (netp / listp), 0.0)

    feats = pd.DataFrame({"NetPrice": netp, "ListPrice": listp, "DiscountRate": disc})

    for L in lags:
        feats[f"lag_{L}"] = y.shift(L)

    for R in rolls:
        feats[f"roll_mean_{R}"] = y.shift(1).rolling(R).mean()
        feats[f"roll_std_{R}"]  = y.shift(1).rolling(R).std()
        feats[f"roll_min_{R}"]  = y.shift(1).rolling(R).min()
        feats[f"roll_max_{R}"]  = y.shift(1).rolling(R).max()

    cal = df["StartDate_dt"].apply(lambda d: calendar_features(period, pd.Timestamp(d))).apply(pd.Series)
    feats = pd.concat([feats.reset_index(drop=True), cal.reset_index(drop=True)], axis=1)

    if use_exog:
        for c in WEATHER_COLS:
            feats[c] = to_num(df[c]).fillna(0.0) if c in df.columns else 0.0
        for c in PROMO_COLS:
            feats[c] = to_num(df[c]).fillna(0.0) if c in df.columns else 0.0

    feats["StartDate_dt"] = df["StartDate_dt"].values
    feats["y"] = y.values

    feats = feats.dropna().reset_index(drop=True)
    return feats


def train_model(train_frame: pd.DataFrame) -> Tuple[XGBRegressor, List[str]]:
    feature_cols = [c for c in train_frame.columns if c not in ("StartDate_dt", "y")]
    X = train_frame[feature_cols].to_numpy()
    y = train_frame["y"].to_numpy()

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)
    return model, feature_cols


# ============================================================
# EXOG ATTACH
# ============================================================
def attach_exog_to_history(
    hist: pd.DataFrame,
    weather_hist_agg: pd.DataFrame,
    promo_agg: pd.DataFrame,
) -> pd.DataFrame:
    df = hist.copy()
    df = df.merge(weather_hist_agg, on=["LocationID", "StartDate_dt"], how="left")

    if not promo_agg.empty:
        df = df.merge(promo_agg, on=KEY_COLS + ["StartDate_dt"], how="left")
    else:
        for c in PROMO_COLS:
            df[c] = 0.0

    for c in PROMO_COLS:
        df[c] = to_num(df[c]).fillna(0.0)

    wcols = [c for c in WEATHER_COLS if c in df.columns]
    for c in wcols:
        df[c] = to_num(df[c])
    if wcols:
        df[wcols] = df.groupby("LocationID")[wcols].transform(lambda x: x.fillna(x.mean()))
        df[wcols] = df[wcols].fillna(df[wcols].mean())

    return df


# ============================================================
# FORECAST ONE SERIES (recursive)
# ============================================================
def forecast_one_series(
    hist: pd.DataFrame,
    period: str,
    horizon: int,
    lags: List[int],
    rolls: List[int],
    model: XGBRegressor,
    feature_cols: List[str],
    use_exog: bool,
    weather_hist_agg: pd.DataFrame,
    clim_map: Dict[Tuple[str, int], Dict[str, float]],
    promo_agg: pd.DataFrame,
    promo_clim: Dict[Tuple[str, str, str, int], Dict[str, float]],
    weather_daily: Optional[pd.DataFrame] = None,
    weather_future_daily: Optional[pd.DataFrame] = None,
    weather_future_agg: Optional[pd.DataFrame] = None,
    use_observed_weather: bool = False,
) -> pd.DataFrame:
    hist = hist.sort_values("StartDate_dt").copy()
    last = hist.iloc[-1]

    pid = str(last["ProductID"])
    cid = str(last["ChannelID"])
    lid = str(last["LocationID"])
    uom = str(last.get("UOM", "KG") or "KG")

    start0 = next_start(period, pd.Timestamp(last["StartDate_dt"]))
    future_starts = [start0]
    for _ in range(1, horizon):
        future_starts.append(next_start(period, future_starts[-1]))
    future_ends = [make_end_date(period, s) for s in future_starts]

    if (
        weather_daily is not None
        and not weather_daily.empty
    ):
        w_future = build_complete_future_weather(
            weather_daily=weather_daily,
            weather_future_daily=weather_future_daily,
            period=period,
            location_id=lid,
            future_starts=future_starts,
            use_observed_weather=use_observed_weather,
        )
    else:
        # Fallback to the existing logic
        w_future = weather_features_for_dates(
            period=period,
            location_id=lid,
            dates=future_starts,
            weather_hist_agg=weather_hist_agg,
            clim_map=clim_map,
            weather_future_agg=weather_future_agg,
            use_observed_weather=use_observed_weather,
        )


    p_future = promo_features_for_future(period, (pid, cid, lid), future_starts, promo_agg, promo_clim)

    list_price = float(last["ListPrice"]) if pd.notna(last["ListPrice"]) else np.nan
    net_price_last = float(last["NetPrice"]) if pd.notna(last["NetPrice"]) else np.nan

    qty_hist = hist["Qty"].astype(float).tolist()
    out_rows = []

    for i, (s, e) in enumerate(zip(future_starts, future_ends)):
        feats: Dict[str, float] = {}

        promo_flag  = float(p_future.loc[i, "PromoFlag"])
        promo_depth = float(p_future.loc[i, "PromoDepth"])

        if np.isfinite(list_price) and list_price > 0:
            net_price = list_price * (1.0 - promo_depth / 100.0) if promo_flag > 0 else (
                net_price_last if np.isfinite(net_price_last) else list_price
            )
            net_price = max(0.01, float(net_price))
        else:
            net_price = net_price_last if np.isfinite(net_price_last) else 0.0

        feats["NetPrice"] = float(net_price)
        feats["ListPrice"] = float(list_price) if np.isfinite(list_price) else 0.0
        feats["DiscountRate"] = float(1.0 - (net_price / list_price)) if (np.isfinite(list_price) and list_price > 0) else 0.0

        for L in lags:
            feats[f"lag_{L}"] = qty_hist[-L] if len(qty_hist) >= L else np.nan

        arr = np.asarray(qty_hist, dtype=float)
        for R in rolls:
            if len(arr) >= R:
                window = arr[-R:]
                feats[f"roll_mean_{R}"] = float(np.nanmean(window))
                feats[f"roll_std_{R}"]  = float(np.nanstd(window, ddof=1)) if R > 1 else 0.0
                feats[f"roll_min_{R}"]  = float(np.nanmin(window))
                feats[f"roll_max_{R}"]  = float(np.nanmax(window))
            else:
                feats[f"roll_mean_{R}"] = np.nan
                feats[f"roll_std_{R}"]  = np.nan
                feats[f"roll_min_{R}"]  = np.nan
                feats[f"roll_max_{R}"]  = np.nan

        feats.update(calendar_features(period, s))

        if use_exog:
            for c in WEATHER_COLS:
                feats[c] = float(w_future.loc[i, c]) if c in w_future.columns else 0.0
            for c in PROMO_COLS:
                feats[c] = float(p_future.loc[i, c])

        Xrow = pd.DataFrame([feats])
        for c in feature_cols:
            if c not in Xrow.columns:
                Xrow[c] = np.nan
        Xrow = Xrow[feature_cols]

        if Xrow.isna().any(axis=1).iloc[0]:
            yhat = float(qty_hist[-1])
        else:
            yhat = float(model.predict(Xrow.to_numpy())[0])

        yhat = max(0.0, yhat)
        qty_hist.append(yhat)

        out_rows.append({
            "ProductID": pid,
            "ChannelID": cid,
            "LocationID": lid,
            "StartDate": s.date().isoformat(),
            "EndDate": e.date().isoformat(),
            "Period": period,
            "Qty": np.nan,
            "UOM": uom,
            "NetPrice": feats["NetPrice"],
            "ListPrice": feats["ListPrice"],
            "ForecastQty": yhat,
            "Method": "xgboost_feat" if use_exog else "xgboost_baseline",
        })

    return pd.DataFrame(out_rows)


# ============================================================
# BACKTEST (rolling origin)
# ============================================================
def rolling_backtest(
    df_exog: pd.DataFrame,
    period: str,
    bt_horizon: int,
    use_exog: bool,
    weather_hist_agg: pd.DataFrame,
    clim_map: Dict[Tuple[str, int], Dict[str, float]],
    promo_agg: pd.DataFrame,
    promo_clim: Dict[Tuple[str, str, str, int], Dict[str, float]],
) -> pd.DataFrame:
    cfg = GRAIN_CFG[period]
    lags = cfg["lags"]
    rolls = cfg["rolls"]
    min_train = cfg["min_train_points"]

    n_folds = BACKTEST_CFG[period]["n_folds"]
    step = BACKTEST_CFG[period]["step"]

    rows = []
    for keys, g in df_exog.groupby(KEY_COLS):
        g = g.sort_values("StartDate_dt").copy()
        if len(g) < min_train + bt_horizon + 5:
            continue

        max_end = len(g) - bt_horizon
        cutoffs = []
        cur = max_end
        for _ in range(n_folds):
            cutoffs.append(cur)
            cur -= step
        cutoffs = [c for c in reversed(cutoffs) if c > min_train]
        if not cutoffs:
            continue

        for ci, cutoff in enumerate(cutoffs, start=1):
            train_hist = g.iloc[:cutoff].copy()
            test_hist  = g.iloc[cutoff:cutoff + bt_horizon].copy()

            tr_frame = build_supervised_frame(train_hist, period, lags, rolls, use_exog=use_exog)
            if tr_frame.empty or len(tr_frame) < 15:
                continue

            model, feature_cols = train_model(tr_frame)

            fc = forecast_one_series(
                hist=train_hist,
                period=period,
                horizon=bt_horizon,
                lags=lags,
                rolls=rolls,
                model=model,
                feature_cols=feature_cols,
                use_exog=use_exog,
                weather_hist_agg=weather_hist_agg,
                clim_map=clim_map,
                promo_agg=promo_agg,
                promo_clim=promo_clim,
            )

            test = test_hist[["StartDate_dt", "Qty"]].copy()
            fc2 = fc.copy()
            fc2["StartDate_dt"] = pd.to_datetime(fc2["StartDate"], errors="coerce")

            merged = test.merge(fc2[["StartDate_dt", "ForecastQty"]], on="StartDate_dt", how="inner")
            if merged.empty:
                continue

            score = wmape(merged["Qty"].values, merged["ForecastQty"].values)
            rows.append({
                "ProductID": keys[0],
                "ChannelID": keys[1],
                "LocationID": keys[2],
                "Fold": ci,
                "WMAPE": score,
            })

    return pd.DataFrame(rows)

def rolling_backtest_selected_key(
    df_exog: pd.DataFrame,
    period: str,
    forecast_key: Tuple[str, str, str],
    use_exog: bool,
    weather_hist_agg: pd.DataFrame,
    clim_map: Dict[Tuple[str, int], Dict[str, float]],
    promo_agg: pd.DataFrame,
    promo_clim: Dict[Tuple[str, str, str, int], Dict[str, float]],
) -> Dict[str, object]:
    """
    Rolling-origin backtest for ONE selected forecast element.

    Important:
    - Models are trained globally using all eligible series available
      before each historical cutoff.
    - Only forecast_key is evaluated.
    - Nothing is written to the database.
    - Returns historical Actual vs Forecast rows plus KPI metrics.
    """

    if period not in GRAIN_CFG:
        raise ValueError(f"Invalid period: {period}")

    cfg = GRAIN_CFG[period]

    lags = cfg["lags"]
    rolls = cfg["rolls"]
    min_train = cfg["min_train_points"]

    bt_cfg = BACKTEST_CFG[period]

    n_folds = bt_cfg["n_folds"]
    step = bt_cfg["step"]
    bt_horizon = bt_cfg["bt_horizon"]

    selected_key = tuple(
        str(x).strip()
        for x in forecast_key
    )

    # --------------------------------------------------------
    # Locate selected series
    # --------------------------------------------------------

    selected_mask = (
        (df_exog["ProductID"].astype(str).str.strip() == selected_key[0])
        & (df_exog["ChannelID"].astype(str).str.strip() == selected_key[1])
        & (df_exog["LocationID"].astype(str).str.strip() == selected_key[2])
    )

    selected = (
        df_exog.loc[selected_mask]
        .sort_values("StartDate_dt")
        .copy()
    )

    if selected.empty:
        return {
            "ok": False,
            "reason": "Selected forecast element was not found.",
            "forecast_key": selected_key,
            "rows": [],
            "n": 0,
            "wape": None,
            "accuracy": None,
            "bias_pct": None,
        }

    if len(selected) < min_train + bt_horizon + 5:
        return {
            "ok": False,
            "reason": (
                "Not enough history for selected forecast element. "
                f"Rows={len(selected)}, minimum training={min_train}, "
                f"backtest horizon={bt_horizon}."
            ),
            "forecast_key": selected_key,
            "rows": [],
            "n": 0,
            "wape": None,
            "accuracy": None,
            "bias_pct": None,
        }

    # --------------------------------------------------------
    # Determine historical cutoffs using selected series
    # --------------------------------------------------------

    max_end = len(selected) - bt_horizon

    cutoffs = []
    cur = max_end

    for _ in range(n_folds):
        cutoffs.append(cur)
        cur -= step

    cutoffs = [
        cutoff
        for cutoff in reversed(cutoffs)
        if cutoff > min_train
    ]

    if not cutoffs:
        return {
            "ok": False,
            "reason": "No valid backtest cutoffs available.",
            "forecast_key": selected_key,
            "rows": [],
            "n": 0,
            "wape": None,
            "accuracy": None,
            "bias_pct": None,
        }

    result_rows = []

    # --------------------------------------------------------
    # Rolling-origin folds
    # --------------------------------------------------------

    for fold_no, cutoff in enumerate(cutoffs, start=1):

        train_selected = selected.iloc[:cutoff].copy()

        test_selected = selected.iloc[
            cutoff:cutoff + bt_horizon
        ].copy()

        if train_selected.empty or test_selected.empty:
            continue

        cutoff_date = pd.Timestamp(
            train_selected["StartDate_dt"].max()
        )

        # ----------------------------------------------------
        # Global training dataset
        #
        # Critical:
        # only data available at this historical cutoff is
        # allowed into the model.
        # ----------------------------------------------------

        global_train = df_exog[
            df_exog["StartDate_dt"] <= cutoff_date
        ].copy()

        frames = []

        for _, g in global_train.groupby(KEY_COLS):

            g = (
                g.sort_values("StartDate_dt")
                .copy()
            )

            if len(g) < max(lags + rolls) + 10:
                continue

            if len(g) < min_train:
                continue

            frame = build_supervised_frame(
                g,
                period,
                lags,
                rolls,
                use_exog=use_exog,
            )

            if not frame.empty:
                frames.append(frame)

        if not frames:
            continue

        train_frame = pd.concat(
            frames,
            ignore_index=True,
        )

        if len(train_frame) < 15:
            continue

        # ----------------------------------------------------
        # Train global model for this historical cutoff
        # ----------------------------------------------------

        model, feature_cols = train_model(
            train_frame
        )

        # ----------------------------------------------------
        # Forecast only selected series
        # ----------------------------------------------------

        fc = forecast_one_series(
            hist=train_selected,
            period=period,
            horizon=len(test_selected),
            lags=lags,
            rolls=rolls,
            model=model,
            feature_cols=feature_cols,
            use_exog=use_exog,
            weather_hist_agg=weather_hist_agg,
            clim_map=clim_map,
            promo_agg=promo_agg,
            promo_clim=promo_clim,
        )

        if fc.empty:
            continue

        fc["StartDate_dt"] = pd.to_datetime(
            fc["StartDate"],
            errors="coerce",
        )

        actual = test_selected[
            ["StartDate_dt", "Qty"]
        ].copy()

        merged = actual.merge(
            fc[
                [
                    "StartDate_dt",
                    "ForecastQty",
                ]
            ],
            on="StartDate_dt",
            how="inner",
        )

        if merged.empty:
            continue

        # ----------------------------------------------------
        # Preserve individual historical predictions
        # ----------------------------------------------------

        for _, row in merged.iterrows():

            actual_qty = float(row["Qty"])
            forecast_qty = float(
                row["ForecastQty"]
            )

            result_rows.append({
                "ProductID": selected_key[0],
                "ChannelID": selected_key[1],
                "LocationID": selected_key[2],
                "Period": period,
                "Fold": fold_no,
                "StartDate": pd.Timestamp(
                    row["StartDate_dt"]
                ).date().isoformat(),
                "ActualQty": actual_qty,
                "ForecastQty": forecast_qty,
                "Error": forecast_qty - actual_qty,
                "AbsError": abs(
                    forecast_qty - actual_qty
                ),
            })

    # --------------------------------------------------------
    # No successful folds
    # --------------------------------------------------------

    if not result_rows:

        return {
            "ok": False,
            "reason": (
                "Backtest did not produce any matched "
                "historical forecast observations."
            ),
            "forecast_key": selected_key,
            "rows": [],
            "n": 0,
            "wape": None,
            "accuracy": None,
            "bias_pct": None,
        }

    # --------------------------------------------------------
    # KPI calculation
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        result_rows
    )

    actual_values = (
        result_df["ActualQty"]
        .astype(float)
        .to_numpy()
    )

    forecast_values = (
        result_df["ForecastQty"]
        .astype(float)
        .to_numpy()
    )

    abs_actual_sum = float(
        np.sum(np.abs(actual_values))
    )

    abs_error_sum = float(
        np.sum(
            np.abs(
                forecast_values
                - actual_values
            )
        )
    )

    error_sum = float(
        np.sum(
            forecast_values
            - actual_values
        )
    )

    if abs_actual_sum > 1e-12:

        wape_pct = (
            abs_error_sum
            / abs_actual_sum
            * 100.0
        )

        bias_pct = (
            error_sum
            / abs_actual_sum
            * 100.0
        )

    else:
        wape_pct = None
        bias_pct = None

    accuracy = (
        max(
            0.0,
            100.0 - wape_pct,
        )
        if wape_pct is not None
        else None
    )

    return {
        "ok": True,

        "forecast_key": {
            "ProductID": selected_key[0],
            "ChannelID": selected_key[1],
            "LocationID": selected_key[2],
        },

        "period": period,

        "folds": int(
            result_df["Fold"].nunique()
        ),

        "n": int(len(result_df)),

        "wape": (
            float(wape_pct)
            if wape_pct is not None
            else None
        ),

        "accuracy": (
            float(accuracy)
            if accuracy is not None
            else None
        ),

        "bias_pct": (
            float(bias_pct)
            if bias_pct is not None
            else None
        ),

        "rows": result_rows,
    }


# ============================================================
# DB WRITE HELPERS
# ============================================================
def _period_slug(period_ui: str) -> str:
    p = (period_ui or "").strip().lower()
    if p == "daily": return "daily"
    if p == "weekly": return "weekly"
    if p == "monthly": return "monthly"
    raise ValueError(f"Invalid period: {period_ui}")

def _forecast_table_name(level: str, period_ui: str, baseline: bool) -> str:
    pl = _period_slug(period_ui)
    base = f"forecast_{level}_{pl}"
    return f"{base}_baseline" if baseline else base

def write_forecast_df_to_db(
    engine: Engine,
    db_schema: str,
    table: str,
    df: pd.DataFrame,
    scenario_id: int,
) -> Dict[str, int]:
    """
    Writes forecast rows into "db_schema"."table" for ONE scenario_id.
    We DELETE existing rows for the same (scenario_id, Period, Method, StartDate window) and INSERT the new batch.
    """
    if df is None or df.empty:
        return {"deleted": 0, "inserted": 0}

    needed = [
        "scenario_id",
        "ProductID","ChannelID","LocationID",
        "StartDate","EndDate","Period",
        "Qty","UOM","NetPrice","ListPrice","ForecastQty","Method"
    ]

    # validate required df cols (except scenario_id which we add)
    required_in_df = [c for c in needed if c != "scenario_id"]
    missing = [c for c in required_in_df if c not in df.columns]
    if missing:
        raise ValueError(f"Forecast df missing columns: {missing}")

    out = df.copy()
    out["scenario_id"] = int(scenario_id)

    # normalize text
    for c in ["ProductID","ChannelID","LocationID","Period","UOM","Method"]:
        out[c] = out[c].astype(str).str.strip()

    # numeric
    out["Qty"] = pd.to_numeric(out["Qty"], errors="coerce")
    out["ForecastQty"] = pd.to_numeric(out["ForecastQty"], errors="coerce")
    out["NetPrice"] = pd.to_numeric(out["NetPrice"], errors="coerce")
    out["ListPrice"] = pd.to_numeric(out["ListPrice"], errors="coerce")

    sdt = pd.to_datetime(out["StartDate"], errors="coerce")
    if sdt.isna().all():
        raise ValueError("StartDate could not be parsed in forecast df.")
    smin = sdt.min()
    smax = sdt.max()

    period_ui = str(out["Period"].iloc[0]).strip()
    method = str(out["Method"].iloc[0]).strip()

    table_qual = f'"{db_schema}"."{table}"'

    del_sql = text(f"""
      DELETE FROM {table_qual}
      WHERE scenario_id = :scenario_id
        AND "Period" = :period
        AND "Method" = :method
        AND "StartDate" BETWEEN CAST(:smin AS date) AND CAST(:smax AS date);
    """)

    ins_sql = text(f"""
      INSERT INTO {table_qual}
        (scenario_id,
         "ProductID","ChannelID","LocationID","StartDate","EndDate","Period",
         "Qty","UOM","NetPrice","ListPrice","ForecastQty","Method")
      VALUES
        (:scenario_id,
         :ProductID,:ChannelID,:LocationID,
         CAST(:StartDate AS date), CAST(:EndDate AS date), :Period,
         :Qty, :UOM, :NetPrice, :ListPrice, :ForecastQty, :Method);
    """)
    print("DEBUG write_forecast_df_to_db scenario_id =", scenario_id, "table =", table)
    payload = out[needed].where(pd.notna(out[needed]), None).to_dict(orient="records")

    deleted = 0
    with engine.begin() as conn:
        res = conn.execute(del_sql, {
            "scenario_id": int(scenario_id),
            "period": period_ui,
            "method": method,
            "smin": str(smin.date()),
            "smax": str(smax.date()),
        })
        try:
            deleted = int(res.rowcount or 0)
        except Exception:
            deleted = 0

        chunk = 5000
        for i in range(0, len(payload), chunk):
            conn.execute(ins_sql, payload[i:i+chunk])

    return {"deleted": deleted, "inserted": len(payload)}



# ============================================================
# MAIN ENTRY: RUN ONE JOB (no CSV, no PNG)
# ============================================================
def run_one_job_df(
    hist_df: pd.DataFrame,
    period: str,
    horizon: int,
    weather_daily: pd.DataFrame,
    promos: pd.DataFrame,
    tag: str,

    weather_future_daily: Optional[pd.DataFrame] = None,

    db_engine: Optional[Engine] = None,
    db_schema: Optional[str] = None,
    level: Optional[str] = None,
    scenario_id: int = 1,
    write_to_db: bool = True,

    # performance
    run_backtest: bool = False,

    # False for interactive Run Forecast.
    # True for full/save/run-all forecasts.
    run_baseline: bool = True,

    # If supplied, train globally but forecast only this series.
    # (ProductID, ChannelID, LocationID)
    forecast_key: Optional[Tuple[str, str, str]] = None,

    return_frames: bool = False,
):
    """
    Produces:
      - baseline forecast rows (Method='xgboost_baseline')
      - feat forecast rows (Method='xgboost_feat')
      - backtest summary DataFrame (WMAPE_base/WMAPE_feat)

    By default, writes baseline+feat to your Postgres forecast tables:
      forecast_{level}_{period}_baseline  and  forecast_{level}_{period}

    No CSV output. No PNG plots.
    """
    df = hist_df.copy()

    # normalize keys
    for c in KEY_COLS:
        df[c] = df[c].astype(str).str.strip()

    # required numeric
    df["Qty"] = to_num(df["Qty"]).fillna(0.0)

    # prices optional
    if "NetPrice" not in df.columns:
        df["NetPrice"] = np.nan
    if "ListPrice" not in df.columns:
        df["ListPrice"] = np.nan
    df["NetPrice"] = to_num(df["NetPrice"])
    df["ListPrice"] = to_num(df["ListPrice"])

    # UOM optional
    if "UOM" not in df.columns:
        df["UOM"] = "KG"
    df["UOM"] = df["UOM"].astype(str).fillna("KG")

    # dates + period filter
    df["StartDate_dt"] = parse_date(df["StartDate"])
    df["Period"] = df["Period"].astype(str).str.strip()
    df = df[(df["Period"] == period) & df["StartDate_dt"].notna()].copy()
    if df.empty:
        return {"ok": False, "tag": tag, "reason": "no rows after filtering"}

    # fill missing prices within series
    df = df.sort_values(KEY_COLS + ["StartDate_dt"])
    df["NetPrice"] = df.groupby(KEY_COLS)["NetPrice"].ffill().bfill()
    df["ListPrice"] = df.groupby(KEY_COLS)["ListPrice"].ffill().bfill()

    # build exog climatologies
    weather_hist_agg, clim_map = build_weather_climatology(weather_daily, period,)
    weather_future_agg = aggregate_future_weather(weather_future_daily, period,)


    promo_daily = expand_promotions_to_daily(promos)
    promo_agg = aggregate_promotions_to_period(promo_daily, period)
    promo_clim = build_promo_climatology(promo_agg, period)

    # attach exog to history
    df_exog = attach_exog_to_history(df, weather_hist_agg, promo_agg)

    cfg = GRAIN_CFG[period]
    lags = cfg["lags"]
    rolls = cfg["rolls"]
    min_train = cfg["min_train_points"]

    # ============================================================
    # TRAIN GLOBAL MODEL(S)
    #
    # Feature model always uses ALL eligible series.
    # Baseline can be skipped for interactive Forecast Tuning.
    # ============================================================

    frames_base = []
    frames_feat = []

    for _, g in df_exog.groupby(KEY_COLS):
        g = g.sort_values("StartDate_dt")

        if len(g) < max(lags + rolls) + 10:
            continue

        if len(g) < min_train:
            continue

        if run_baseline:
            frames_base.append(
                build_supervised_frame(
                    g, period, lags, rolls, use_exog=False
                )
            )

        frames_feat.append(
            build_supervised_frame(
                g, period, lags, rolls, use_exog=True
            )
        )


    if not frames_feat:
        return {
            "ok": False,
            "tag": tag,
            "reason": "not enough history to train global feature model"
        }


    # Feature model: always required
    train_feat = pd.concat(frames_feat, ignore_index=True)
    model_feat, cols_feat = train_model(train_feat)


    # Baseline model: optional
    model_base = None
    cols_base = []

    if run_baseline:

        if not frames_base:
            return {
                "ok": False,
                "tag": tag,
                "reason": "not enough history to train global baseline model"
            }

        train_base = pd.concat(frames_base, ignore_index=True)
        model_base, cols_base = train_model(train_base)

    
    # ============================================================
    # FORECAST
    #
    # forecast_key=None:
    #     forecast every eligible series
    #
    # forecast_key=(ProductID, ChannelID, LocationID):
    #     global model was still trained on ALL series,
    #     but forecast only the selected series.
    # ============================================================

    out_base = []
    out_feat = []

    normalized_forecast_key = None

    if forecast_key is not None:
        normalized_forecast_key = tuple(
            str(x).strip() for x in forecast_key
        )


    for keys, g in df_exog.groupby(KEY_COLS):

        normalized_keys = tuple(
            str(x).strip() for x in keys
        )

        # Interactive forecast:
        # ignore every series except the selected one.
        if (
            normalized_forecast_key is not None
            and normalized_keys != normalized_forecast_key
        ):
            continue

        g = g.sort_values("StartDate_dt").copy()

        if len(g) < min_train:
            continue

        # Baseline is optional
        if run_baseline and model_base is not None:
            out_base.append(
                forecast_one_series(
                    hist=g,
                    period=period,
                    horizon=horizon,
                    lags=lags,
                    rolls=rolls,
                    model=model_base,
                    feature_cols=cols_base,
                    use_exog=False,
                    weather_hist_agg=weather_hist_agg,
                    clim_map=clim_map,
                    promo_agg=promo_agg,
                    promo_clim=promo_clim,
                )
            )

        # Feature forecast
        out_feat.append(
            forecast_one_series(
                hist=g,
                period=period,
                horizon=horizon,
                lags=lags,
                rolls=rolls,
                model=model_feat,
                feature_cols=cols_feat,
                use_exog=True,
                weather_hist_agg=weather_hist_agg,
                clim_map=clim_map,
                promo_agg=promo_agg,
                promo_clim=promo_clim,
                weather_daily=weather_daily,
                weather_future_daily=weather_future_daily,
                weather_future_agg=weather_future_agg,
                use_observed_weather=True,
            )
        )


    fc_base = (
        pd.concat(out_base, ignore_index=True)
        if out_base
        else pd.DataFrame()
    )

    fc_feat = (
        pd.concat(out_feat, ignore_index=True)
        if out_feat
        else pd.DataFrame()
    )


    # DB write (recommended)
    db_results = {}
    if write_to_db:
        if db_engine is None or not db_schema or not level:
            raise ValueError("write_to_db=True requires db_engine, db_schema, and level.")
        t_base = _forecast_table_name(level, period, baseline=True)
        t_feat = _forecast_table_name(level, period, baseline=False)
        if run_baseline and not fc_base.empty:
            db_results["baseline"] = write_forecast_df_to_db(
                db_engine,
                db_schema,
                t_base,
                fc_base,
                scenario_id=scenario_id,
            )

        if not fc_feat.empty:
            db_results["feat"] = write_forecast_df_to_db(
                db_engine,
                db_schema,
                t_feat,
                fc_feat,
                scenario_id=scenario_id,
            )

    # ============================================================
    # BACKTEST
    # Do NOT run this during normal interactive forecasting.
    # Rolling backtesting repeatedly trains XGBoost models and is
    # considerably more expensive than producing the forecast.
    # ============================================================

    summary = pd.DataFrame()

    if run_backtest:
        bt_h = BACKTEST_CFG[period]["bt_horizon"]

        bt_base = rolling_backtest(
            df_exog,
            period,
            bt_h,
            use_exog=False,
            weather_hist_agg=weather_hist_agg,
            clim_map=clim_map,
            promo_agg=promo_agg,
            promo_clim=promo_clim,
        )

        bt_feat = rolling_backtest(
            df_exog,
            period,
            bt_h,
            use_exog=True,
            weather_hist_agg=weather_hist_agg,
            clim_map=clim_map,
            promo_agg=promo_agg,
            promo_clim=promo_clim,
        )

        if (not bt_base.empty) and (not bt_feat.empty):

            bsum = (
                bt_base
                .groupby(KEY_COLS, as_index=False)["WMAPE"]
                .mean()
                .rename(columns={"WMAPE": "WMAPE_base"})
            )

            fsum = (
                bt_feat
                .groupby(KEY_COLS, as_index=False)["WMAPE"]
                .mean()
                .rename(columns={"WMAPE": "WMAPE_feat"})
            )

            summary = bsum.merge(
                fsum,
                on=KEY_COLS,
                how="inner"
            )

            summary["Series"] = (
                summary["ProductID"]
                + " | "
                + summary["ChannelID"]
                + " | "
                + summary["LocationID"]
            )

            summary["bt_horizon"] = bt_h

    result = {
        "ok": True,
        "tag": tag,
        "period": period,
        "horizon": int(horizon),
        "rows_baseline": int(len(fc_base)),
        "rows_feat": int(len(fc_feat)),
        "db": db_results,
        "backtest_rows": int(len(summary)),
        "mean_wmape_base": (float(summary["WMAPE_base"].mean()) if not summary.empty else None),
        "mean_wmape_feat": (float(summary["WMAPE_feat"].mean()) if not summary.empty else None),
    }

    if return_frames:
        result["fc_base"] = fc_base
        result["fc_feat"] = fc_feat
        result["backtest_summary"] = summary


    return result
