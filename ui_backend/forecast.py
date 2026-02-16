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


def weather_features_for_dates(
    period: str,
    location_id: str,
    dates: List[pd.Timestamp],
    weather_hist_agg: pd.DataFrame,
    clim_map: Dict[Tuple[str, int], Dict[str, float]],
) -> pd.DataFrame:
    location_id = str(location_id).strip()
    tmp = pd.DataFrame({"LocationID": [location_id] * len(dates), "StartDate_dt": dates})
    out = tmp.merge(weather_hist_agg, on=["LocationID", "StartDate_dt"], how="left")

    if period == "Daily":
        keys = [d.dayofyear for d in dates]
    elif period == "Weekly":
        keys = [int(d.isocalendar().week) for d in dates]
    else:
        keys = [d.month for d in dates]

    wcols = [c for c in weather_hist_agg.columns if c not in ("LocationID", "StartDate_dt")]
    for i, sk in enumerate(keys):
        if (len(wcols) > 0) and out.loc[i, wcols].isna().all():
            vals = clim_map.get((location_id, int(sk)), {})
            for c in wcols:
                if c in vals:
                    out.loc[i, c] = vals[c]

    for c in wcols:
        if out[c].isna().any():
            out[c] = out[c].fillna(weather_hist_agg[c].mean())

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

    w_future = weather_features_for_dates(period, lid, future_starts, weather_hist_agg, clim_map)
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
) -> Dict[str, int]:
    """
    Writes forecast rows into "db_schema"."table".
    We DELETE existing rows for the same (Period, Method, StartDate window) and INSERT the new batch.
    """
    if df is None or df.empty:
        return {"deleted": 0, "inserted": 0}

    needed = [
        "ProductID","ChannelID","LocationID",
        "StartDate","EndDate","Period",
        "Qty","UOM","NetPrice","ListPrice","ForecastQty","Method"
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Forecast df missing columns: {missing}")

    out = df.copy()

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
      WHERE "Period" = :period
        AND "Method" = :method
        AND "StartDate" BETWEEN CAST(:smin AS date) AND CAST(:smax AS date);
    """)

    ins_sql = text(f"""
      INSERT INTO {table_qual}
        ("ProductID","ChannelID","LocationID","StartDate","EndDate","Period",
         "Qty","UOM","NetPrice","ListPrice","ForecastQty","Method")
      VALUES
        (:ProductID,:ChannelID,:LocationID,
         CAST(:StartDate AS date), CAST(:EndDate AS date), :Period,
         :Qty, :UOM, :NetPrice, :ListPrice, :ForecastQty, :Method);
    """)

    payload = out[needed].where(pd.notna(out[needed]), None).to_dict(orient="records")

    deleted = 0
    with engine.begin() as conn:
        res = conn.execute(del_sql, {
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
    tag: str,  # kept for logging / traceability

    # DB output (recommended)
    db_engine: Optional[Engine] = None,
    db_schema: Optional[str] = None,
    level: Optional[str] = None,
    write_to_db: bool = True,

    # optional: also return results to caller
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
    weather_hist_agg, clim_map = build_weather_climatology(weather_daily, period)

    promo_daily = expand_promotions_to_daily(promos)
    promo_agg = aggregate_promotions_to_period(promo_daily, period)
    promo_clim = build_promo_climatology(promo_agg, period)

    # attach exog to history
    df_exog = attach_exog_to_history(df, weather_hist_agg, promo_agg)

    cfg = GRAIN_CFG[period]
    lags = cfg["lags"]
    rolls = cfg["rolls"]
    min_train = cfg["min_train_points"]

    # train global baseline + feature models (stacked series)
    frames_base = []
    frames_feat = []
    for _, g in df_exog.groupby(KEY_COLS):
        g = g.sort_values("StartDate_dt")
        if len(g) < max(lags + rolls) + 10:
            continue
        if len(g) < min_train:
            continue
        frames_base.append(build_supervised_frame(g, period, lags, rolls, use_exog=False))
        frames_feat.append(build_supervised_frame(g, period, lags, rolls, use_exog=True))

    if not frames_base or not frames_feat:
        return {"ok": False, "tag": tag, "reason": "not enough history to train global models"}

    train_base = pd.concat(frames_base, ignore_index=True)
    train_feat = pd.concat(frames_feat, ignore_index=True)

    model_base, cols_base = train_model(train_base)
    model_feat, cols_feat = train_model(train_feat)

    # forecast all series
    out_base = []
    out_feat = []
    for _, g in df_exog.groupby(KEY_COLS):
        g = g.sort_values("StartDate_dt").copy()
        if len(g) < min_train:
            continue

        out_base.append(
            forecast_one_series(
                hist=g, period=period, horizon=horizon, lags=lags, rolls=rolls,
                model=model_base, feature_cols=cols_base, use_exog=False,
                weather_hist_agg=weather_hist_agg, clim_map=clim_map,
                promo_agg=promo_agg, promo_clim=promo_clim,
            )
        )
        out_feat.append(
            forecast_one_series(
                hist=g, period=period, horizon=horizon, lags=lags, rolls=rolls,
                model=model_feat, feature_cols=cols_feat, use_exog=True,
                weather_hist_agg=weather_hist_agg, clim_map=clim_map,
                promo_agg=promo_agg, promo_clim=promo_clim,
            )
        )

    fc_base = pd.concat(out_base, ignore_index=True) if out_base else pd.DataFrame()
    fc_feat = pd.concat(out_feat, ignore_index=True) if out_feat else pd.DataFrame()

    # DB write (recommended)
    db_results = {}
    if write_to_db:
        if db_engine is None or not db_schema or not level:
            raise ValueError("write_to_db=True requires db_engine, db_schema, and level.")
        t_base = _forecast_table_name(level, period, baseline=True)
        t_feat = _forecast_table_name(level, period, baseline=False)
        db_results["baseline"] = write_forecast_df_to_db(db_engine, db_schema, t_base, fc_base)
        db_results["feat"] = write_forecast_df_to_db(db_engine, db_schema, t_feat, fc_feat)

    # backtest summary (in-memory)
    bt_h = BACKTEST_CFG[period]["bt_horizon"]
    bt_base = rolling_backtest(
        df_exog, period, bt_h, use_exog=False,
        weather_hist_agg=weather_hist_agg, clim_map=clim_map,
        promo_agg=promo_agg, promo_clim=promo_clim
    )
    bt_feat = rolling_backtest(
        df_exog, period, bt_h, use_exog=True,
        weather_hist_agg=weather_hist_agg, clim_map=clim_map,
        promo_agg=promo_agg, promo_clim=promo_clim
    )

    summary = pd.DataFrame()
    if (not bt_base.empty) and (not bt_feat.empty):
        bsum = bt_base.groupby(KEY_COLS, as_index=False)["WMAPE"].mean().rename(columns={"WMAPE": "WMAPE_base"})
        fsum = bt_feat.groupby(KEY_COLS, as_index=False)["WMAPE"].mean().rename(columns={"WMAPE": "WMAPE_feat"})
        summary = bsum.merge(fsum, on=KEY_COLS, how="inner")
        summary["Series"] = summary["ProductID"] + " | " + summary["ChannelID"] + " | " + summary["LocationID"]
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
