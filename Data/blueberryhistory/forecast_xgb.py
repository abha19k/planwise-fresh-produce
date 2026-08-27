from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# If you don't have xgboost:
#   pip install xgboost
from xgboost import XGBRegressor


# -------------------------
# CONFIG
# -------------------------
JOBS = [
    # file, period, horizon, output
    ("History_111_Daily.csv",   "Daily",   28, "Forecast_111_Daily.csv"),
    ("History_121_Daily.csv",   "Daily",   28, "Forecast_121_Daily.csv"),
    ("History_221_Daily.csv",   "Daily",   28, "Forecast_221_Daily.csv"),

    ("History_111_Weekly.csv",  "Weekly",  13, "Forecast_111_Weekly.csv"),
    ("History_121_Weekly.csv",  "Weekly",  13, "Forecast_121_Weekly.csv"),
    ("History_221_Weekly.csv",  "Weekly",  13, "Forecast_221_Weekly.csv"),

    ("History_111_Monthly.csv", "Monthly", 12, "Forecast_111_Monthly.csv"),
    ("History_121_Monthly.csv", "Monthly", 12, "Forecast_121_Monthly.csv"),
    ("History_221_Monthly.csv", "Monthly", 12, "Forecast_221_Monthly.csv"),
]

KEY_COLS = ["ProductID", "ChannelID", "LocationID"]
METHOD = "xgboost_qty"

OUT_COLS = [
    "ProductID","ChannelID","LocationID",
    "StartDate","EndDate","Period",
    "Qty","UOM","NetPrice","ListPrice",
    "ForecastQty","Method"
]

# Feature settings by grain
GRAIN_CFG = {
    "Daily":   {"freq": "D", "lags": [1, 7, 14, 28], "rolls": [7, 14, 28]},
    "Weekly":  {"freq": "W-MON", "lags": [1, 4, 13, 52], "rolls": [4, 13]},
    "Monthly": {"freq": "MS", "lags": [1, 3, 6, 12], "rolls": [3, 6, 12]},
}

# XGB parameters (reasonable defaults)
XGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    objective="reg:squarederror",
    n_jobs=8,
)


# -------------------------
# Helpers
# -------------------------
def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def parse_date(s: pd.Series) -> pd.Series:
    # history uses ISO YYYY-MM-DD in your generated files
    return pd.to_datetime(s, errors="coerce")

def make_end_date(period: str, start: pd.Timestamp) -> pd.Timestamp:
    if period == "Daily":
        return start
    if period == "Weekly":
        return start + pd.Timedelta(days=6)
    if period == "Monthly":
        return (start + pd.offsets.MonthEnd(0))
    raise ValueError(period)

def next_start(period: str, start: pd.Timestamp) -> pd.Timestamp:
    if period == "Daily":
        return start + pd.Timedelta(days=1)
    if period == "Weekly":
        return start + pd.Timedelta(weeks=1)
    if period == "Monthly":
        return start + pd.offsets.MonthBegin(1)
    raise ValueError(period)

def calendar_features(period: str, dt: pd.DatetimeIndex) -> pd.DataFrame:
    # Keep it simple and stable across grains
    out = pd.DataFrame(index=dt)
    out["year"] = dt.year
    out["month"] = dt.month
    if period == "Daily":
        out["dow"] = dt.dayofweek
        out["doy"] = dt.dayofyear
    elif period == "Weekly":
        iso = dt.isocalendar()
        out["weekofyear"] = iso.week.astype(int).values
    elif period == "Monthly":
        out["month"] = dt.month
    return out.reset_index(drop=True)

def build_supervised_frame(
    df: pd.DataFrame,
    period: str,
    lags: List[int],
    rolls: List[int],
) -> pd.DataFrame:
    """
    Input df must have:
      - StartDate_dt (datetime)
      - Qty (numeric)
      - NetPrice, ListPrice (numeric)
    Returns one row per time point with engineered features + target y=Qty.
    """
    df = df.sort_values("StartDate_dt").copy()

    # Core time index
    idx = pd.DatetimeIndex(df["StartDate_dt"].values)

    # Calendar features
    cal = calendar_features(period, idx)

    # Price features
    netp = df["NetPrice"].astype(float).values
    listp = df["ListPrice"].astype(float).values
    disc = np.where(listp > 0, 1.0 - (netp / listp), 0.0)

    feats = pd.DataFrame({
        "NetPrice": netp,
        "ListPrice": listp,
        "DiscountRate": disc,
    })

    # Lag features from Qty
    y = df["Qty"].astype(float)
    for L in lags:
        feats[f"lag_{L}"] = y.shift(L)

    # Rolling stats
    for R in rolls:
        feats[f"roll_mean_{R}"] = y.shift(1).rolling(R).mean()
        feats[f"roll_std_{R}"]  = y.shift(1).rolling(R).std()
        feats[f"roll_min_{R}"]  = y.shift(1).rolling(R).min()
        feats[f"roll_max_{R}"]  = y.shift(1).rolling(R).max()

    # Join calendar
    feats = pd.concat([feats.reset_index(drop=True), cal.reset_index(drop=True)], axis=1)

    # Attach meta
    feats["StartDate_dt"] = df["StartDate_dt"].values
    feats["y"] = y.values

    # Drop rows where lags/rolls not available
    feats = feats.dropna().reset_index(drop=True)
    return feats


def train_one_model(train_frame: pd.DataFrame) -> Tuple[XGBRegressor, List[str]]:
    feature_cols = [c for c in train_frame.columns if c not in ("StartDate_dt", "y")]
    X = train_frame[feature_cols].to_numpy()
    y = train_frame["y"].to_numpy()

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)
    return model, feature_cols


def forecast_series_recursive(
    hist: pd.DataFrame,
    period: str,
    horizon: int,
    lags: List[int],
    rolls: List[int],
    model: XGBRegressor,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Forecast one (ProductID, ChannelID, LocationID) series recursively.
    Prices are carried forward from last known row.
    """
    hist = hist.sort_values("StartDate_dt").copy()
    last_row = hist.iloc[-1]

    # Carry forward prices and UOM
    last_net = float(last_row["NetPrice"]) if pd.notna(last_row["NetPrice"]) else np.nan
    last_list = float(last_row["ListPrice"]) if pd.notna(last_row["ListPrice"]) else np.nan
    uom = last_row.get("UOM", "KG")

    # Build a working time series list of qty values (history + forecasts)
    qty_hist = hist["Qty"].astype(float).tolist()

    # Build future dates starting next time bucket after last StartDate
    start0 = next_start(period, pd.Timestamp(last_row["StartDate_dt"]))
    future_starts = [start0]
    for _ in range(1, horizon):
        future_starts.append(next_start(period, future_starts[-1]))
    future_ends = [make_end_date(period, s) for s in future_starts]

    # For feature building, we will compute features step-by-step using current qty_hist
    out = []

    for h, (s, e) in enumerate(zip(future_starts, future_ends), start=1):
        # We need to construct one feature row for time s
        # Build lag/rolling from qty_hist (which includes previous forecasts)
        feats = {}

        # Price features (carried forward)
        feats["NetPrice"] = last_net
        feats["ListPrice"] = last_list
        feats["DiscountRate"] = (0.0 if (not np.isfinite(last_list) or last_list <= 0 or not np.isfinite(last_net))
                                 else float(1.0 - (last_net / last_list)))

        # Lags
        for L in lags:
            feats[f"lag_{L}"] = qty_hist[-L] if len(qty_hist) >= L else np.nan

        # Rolls based on history up to t-1
        arr = np.array(qty_hist, dtype=float)
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

        # Calendar feats
        idx = pd.DatetimeIndex([s])
        cal = calendar_features(period, idx).iloc[0].to_dict()
        feats.update(cal)

        # Prepare model row
        Xrow = pd.DataFrame([feats])

        # Align columns exactly
        for c in feature_cols:
            if c not in Xrow.columns:
                Xrow[c] = np.nan
        Xrow = Xrow[feature_cols]

        # If still NaNs (early horizon) fallback to last observed qty (safe)
        if Xrow.isna().any(axis=1).iloc[0]:
            yhat = float(qty_hist[-1])
        else:
            yhat = float(model.predict(Xrow.to_numpy())[0])

        yhat = max(0.0, yhat)
        qty_hist.append(yhat)

        out.append((s.date().isoformat(), e.date().isoformat(), yhat, uom, last_net, last_list))

    return pd.DataFrame(out, columns=["StartDate","EndDate","ForecastQty","UOM","NetPrice","ListPrice"])


def run_job(input_file: str, period: str, horizon: int, output_file: str):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing: {input_file}")

    cfg = GRAIN_CFG[period]
    lags = cfg["lags"]
    rolls = cfg["rolls"]

    df = pd.read_csv(input_file, dtype=str).fillna("")
    # Normalize
    for c in KEY_COLS:
        df[c] = df[c].astype(str).str.strip()

    df["Qty"] = to_num(df["Qty"]).fillna(0.0)
    df["NetPrice"] = to_num(df.get("NetPrice", np.nan))
    df["ListPrice"] = to_num(df.get("ListPrice", np.nan))
    df["StartDate_dt"] = parse_date(df["StartDate"])
    df["EndDate_dt"] = parse_date(df["EndDate"])  # can be blank for Daily, ok
    df["Period"] = df["Period"].astype(str).str.strip()

    # Keep only matching period + valid StartDate
    df = df[(df["Period"] == period) & df["StartDate_dt"].notna()].copy()
    if df.empty:
        pd.DataFrame(columns=OUT_COLS).to_csv(output_file, index=False)
        print(f"[WARN] {input_file}: empty -> wrote {output_file}")
        return

    # Fill missing prices if any (carry forward within series)
    df = df.sort_values(KEY_COLS + ["StartDate_dt"])
    df["NetPrice"] = df.groupby(KEY_COLS)["NetPrice"].ffill().bfill()
    df["ListPrice"] = df.groupby(KEY_COLS)["ListPrice"].ffill().bfill()

    # Train a global model using all series stacked
    frames = []
    for _, g in df.groupby(KEY_COLS):
        # Need enough points for lags/rolls
        if len(g) < max(lags + rolls) + 5:
            continue
        fr = build_supervised_frame(g[["StartDate_dt","Qty","NetPrice","ListPrice"]].copy(), period, lags, rolls)
        frames.append(fr)

    if not frames:
        print(f"[WARN] {input_file}: not enough history for training. Falling back to naive carry-forward.")
        # fallback: just replicate last qty
        out_rows = []
        for keys, g in df.groupby(KEY_COLS):
            pid, ch, loc = keys
            g = g.sort_values("StartDate_dt")
            last = g.iloc[-1]
            start0 = next_start(period, pd.Timestamp(last["StartDate_dt"]))
            cur = start0
            for _ in range(horizon):
                end = make_end_date(period, cur)
                out_rows.append({
                    "ProductID": pid,
                    "ChannelID": ch,
                    "LocationID": loc,
                    "StartDate": cur.date().isoformat(),
                    "EndDate": end.date().isoformat(),
                    "Period": period,
                    "Qty": np.nan,
                    "UOM": last.get("UOM","KG") or "KG",
                    "NetPrice": float(last["NetPrice"]) if last["NetPrice"] != "" else np.nan,
                    "ListPrice": float(last["ListPrice"]) if last["ListPrice"] != "" else np.nan,
                    "ForecastQty": float(last["Qty"]),
                    "Method": METHOD
                })
                cur = next_start(period, cur)

        out_df = pd.DataFrame(out_rows, columns=OUT_COLS)
        out_df.to_csv(output_file, index=False)
        print(f"[OK] {output_file} rows={len(out_df):,} (naive fallback)")
        return

    train_frame = pd.concat(frames, ignore_index=True)
    model, feature_cols = train_one_model(train_frame)

    # Forecast each series recursively
    out_rows = []
    for keys, g in df.groupby(KEY_COLS):
        pid, ch, loc = keys
        g = g.sort_values("StartDate_dt").copy()
        if len(g) < max(lags + rolls) + 5:
            # too short: naive
            last = g.iloc[-1]
            start0 = next_start(period, pd.Timestamp(last["StartDate_dt"]))
            cur = start0
            for _ in range(horizon):
                end = make_end_date(period, cur)
                out_rows.append({
                    "ProductID": pid,
                    "ChannelID": ch,
                    "LocationID": loc,
                    "StartDate": cur.date().isoformat(),
                    "EndDate": end.date().isoformat(),
                    "Period": period,
                    "Qty": np.nan,
                    "UOM": last.get("UOM","KG") or "KG",
                    "NetPrice": float(last["NetPrice"]) if pd.notna(last["NetPrice"]) else np.nan,
                    "ListPrice": float(last["ListPrice"]) if pd.notna(last["ListPrice"]) else np.nan,
                    "ForecastQty": float(last["Qty"]),
                    "Method": METHOD
                })
                cur = next_start(period, cur)
            continue

        one = forecast_series_recursive(
            hist=g[["StartDate_dt","Qty","NetPrice","ListPrice","UOM"]].copy(),
            period=period,
            horizon=horizon,
            lags=lags,
            rolls=rolls,
            model=model,
            feature_cols=feature_cols
        )

        for _, r in one.iterrows():
            out_rows.append({
                "ProductID": pid,
                "ChannelID": ch,
                "LocationID": loc,
                "StartDate": r["StartDate"],
                "EndDate": r["EndDate"],
                "Period": period,
                "Qty": np.nan,
                "UOM": r["UOM"],
                "NetPrice": r["NetPrice"],
                "ListPrice": r["ListPrice"],
                "ForecastQty": float(r["ForecastQty"]),
                "Method": METHOD
            })

    out_df = pd.DataFrame(out_rows, columns=OUT_COLS)
    out_df.to_csv(output_file, index=False)
    print(f"[OK] wrote {output_file} | rows={len(out_df):,}")


def main():
    for inp, period, horizon, out in JOBS:
        print(f"[RUN] {inp} ({period}) -> {out}")
        run_job(inp, period, horizon, out)


if __name__ == "__main__":
    main()
