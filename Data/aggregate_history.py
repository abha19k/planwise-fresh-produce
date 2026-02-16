from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(".")

INPUTS = [
    ("History_111_Daily.csv", "Daily"),
    ("History_111_Weekly.csv", "Weekly"),
    ("History_111_Monthly.csv", "Monthly"),
]

OUT_COLS = ["ProductID","ChannelID","LocationID","Qty","Period","StartDate","EndDate","UOM","NetPrice","ListPrice"]


def channel_to_l2(cid: str) -> str:
    c = str(cid).strip().upper()
    if c in {"AH", "JUMBO", "PLUS"}:
        return "RETAIL"
    if c in {"SLIGRO", "HOLLANDFOODSERVICE", "GREENACRES", "MELEDI"}:
        return "WHOLESALE"
    if c in {"PICNIC", "HELLOFRESH", "FLINK"}:
        return "ONLINE"
    return "OTHER"


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def weighted_avg(df: pd.DataFrame, value_col: str) -> pd.Series:
    """
    Returns weighted average of value_col by Qty per group.
    """
    w = df["Qty"].to_numpy(float)
    x = df[value_col].to_numpy(float)
    denom = w.sum()
    if denom <= 0:
        return np.nan
    return (w * x).sum() / denom


def main():
    prod = pd.read_csv(BASE_DIR / "Product.csv", dtype=str).fillna("")
    prod["Level"] = pd.to_numeric(prod["Level"], errors="coerce")

    # Map L1 product -> family name
    p1 = prod[prod["Level"] == 1][["ProductID", "ProductFamily"]].copy()

    # Build family -> L2 ProductID mapping if L2 rows exist, else synthetic
    fam_key = {}
    fam_rows = prod[prod["Level"] == 2][["ProductID", "ProductFamily"]].copy()
    if len(fam_rows) > 0:
        for _, r in fam_rows.iterrows():
            fam_key[r["ProductFamily"]] = r["ProductID"]
    else:
        for fam in prod["ProductFamily"].unique():
            fam = str(fam).strip()
            if fam:
                fam_key[fam] = f"FAM_{fam.upper()}"

    p1["ProductL2"] = p1["ProductFamily"].map(fam_key).fillna("FAM_UNKNOWN")
    p1_to_l2 = p1.set_index("ProductID")["ProductL2"].to_dict()

    for in_file, period_label in INPUTS:
        df = pd.read_csv(BASE_DIR / in_file, dtype=str).fillna("")

        # types
        df["Qty"] = safe_num(df["Qty"]).fillna(0.0)
        df["NetPrice"] = safe_num(df["NetPrice"])
        df["ListPrice"] = safe_num(df["ListPrice"])

        # ----------------------------
        # 121 = Product L1, Channel L2, Location L1
        # ----------------------------
        d121 = df.copy()
        d121["ChannelID"] = d121["ChannelID"].apply(channel_to_l2)

        keys121 = ["ProductID","ChannelID","LocationID","Period","StartDate","EndDate","UOM"]
        g121 = d121.groupby(keys121, as_index=False)

        out121_qty = g121["Qty"].sum()

        # Weighted price via sum(Qty*Price)/sum(Qty)
        d121["NetPrice_x_Qty"] = d121["NetPrice"] * d121["Qty"]
        d121["ListPrice_x_Qty"] = d121["ListPrice"] * d121["Qty"]

        num_np = d121.groupby(keys121, as_index=False)["NetPrice_x_Qty"].sum()
        num_lp = d121.groupby(keys121, as_index=False)["ListPrice_x_Qty"].sum()

        out121 = out121_qty.merge(num_np, on=keys121, how="left").merge(num_lp, on=keys121, how="left")
        out121["NetPrice"] = np.where(out121["Qty"] > 0, out121["NetPrice_x_Qty"] / out121["Qty"], np.nan)
        out121["ListPrice"] = np.where(out121["Qty"] > 0, out121["ListPrice_x_Qty"] / out121["Qty"], np.nan)
        out121["NetPrice"] = out121["NetPrice"].round(3)
        out121["ListPrice"] = out121["ListPrice"].round(3)

        out121 = out121.drop(columns=["NetPrice_x_Qty","ListPrice_x_Qty"])
        out121 = out121[OUT_COLS]
        out121.to_csv(BASE_DIR / f"History_121_{period_label}.csv", index=False)
        print(f"[DONE] History_121_{period_label}.csv rows={len(out121):,}")

        # ----------------------------
        # 221 = Product L2, Channel L2, Location L1
        # ----------------------------
        d221 = d121.copy()
        d221["ProductID"] = d221["ProductID"].map(p1_to_l2).fillna("FAM_UNKNOWN")

        keys221 = ["ProductID","ChannelID","LocationID","Period","StartDate","EndDate","UOM"]
        g221 = d221.groupby(keys221, as_index=False)

        out221_qty = g221["Qty"].sum()

        d221["NetPrice_x_Qty"] = d221["NetPrice"] * d221["Qty"]
        d221["ListPrice_x_Qty"] = d221["ListPrice"] * d221["Qty"]

        num_np2 = d221.groupby(keys221, as_index=False)["NetPrice_x_Qty"].sum()
        num_lp2 = d221.groupby(keys221, as_index=False)["ListPrice_x_Qty"].sum()

        out221 = out221_qty.merge(num_np2, on=keys221, how="left").merge(num_lp2, on=keys221, how="left")
        out221["NetPrice"] = np.where(out221["Qty"] > 0, out221["NetPrice_x_Qty"] / out221["Qty"], np.nan)
        out221["ListPrice"] = np.where(out221["Qty"] > 0, out221["ListPrice_x_Qty"] / out221["Qty"], np.nan)
        out221["NetPrice"] = out221["NetPrice"].round(3)
        out221["ListPrice"] = out221["ListPrice"].round(3)

        out221 = out221.drop(columns=["NetPrice_x_Qty","ListPrice_x_Qty"])
        out221 = out221[OUT_COLS]
        out221.to_csv(BASE_DIR / f"History_221_{period_label}.csv", index=False)
        print(f"[DONE] History_221_{period_label}.csv rows={len(out221):,}")


if __name__ == "__main__":
    main()

