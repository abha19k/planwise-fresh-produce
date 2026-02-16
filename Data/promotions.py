from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(".")
OUT_PROMOS = "Promotions.csv"

START_YEAR = 2023
END_YEAR = 2025
PROMOS_PER_YEAR = 18  # set to 12 if you want fewer

RANDOM_SEED = 42

# Seasonal windows: most promos in summer + winter
SEASON_WINDOWS = [
    ("WINTER", (12, 1), (12, 31), 0.18),  # Dec
    ("WINTER", (1, 1), (2, 15), 0.17),    # Jan-Feb
    ("SUMMER", (6, 1), (8, 31), 0.40),    # Jun-Aug
    ("SHOULDER", (3, 1), (5, 31), 0.15),  # Mar-May
    ("SHOULDER2", (9, 1), (11, 30), 0.10) # Sep-Nov
]

# Channel-group bias (your Level=2 IDs)
GROUPS = ["RETAIL", "WHOLESALE", "ONLINE"]
GROUP_PROBS = [0.78, 0.12, 0.10]

# Customer frequency bias (used only for PromoLevel=111)
CUSTOMER_WEIGHTS = {
    # Retail
    "AH": 0.45,
    "JUMBO": 0.32,
    "PLUS": 0.23,

    # Wholesale
    "SLIGRO": 0.40,
    "HOLLANDFOODSERVICE": 0.22,
    "GREENACRES": 0.20,
    "MELEDI": 0.18,

    # Online
    "PICNIC": 0.40,
    "HELLOFRESH": 0.35,
    "FLINK": 0.25,
}

# Discount distribution + durations
DISCOUNT_OPTIONS = [5, 8, 10, 12, 15, 20, 25]
DISCOUNT_WEIGHTS = [0.12, 0.10, 0.18, 0.16, 0.20, 0.16, 0.08]

DUR_OPTIONS = [7, 10, 14]
DUR_WEIGHTS = [0.55, 0.25, 0.20]

# PromoLevel mix
PROMOLEVEL_OPTIONS = ["111", "221"]
PROMOLEVEL_WEIGHTS = [0.70, 0.30]


def pick_start_date(rng: np.random.Generator, year: int) -> tuple[date, str]:
    """Pick start date with seasonal bias."""
    probs = np.array([w[3] for w in SEASON_WINDOWS], dtype=float)
    probs = probs / probs.sum()
    idx = int(rng.choice(len(SEASON_WINDOWS), p=probs))
    tag, (m1, d1), (m2, d2), _ = SEASON_WINDOWS[idx]

    start = date(year, m1, d1)
    end = date(year, m2, d2)
    delta = (end - start).days
    start_day = start + timedelta(days=int(rng.integers(0, max(delta, 1) + 1)))
    return start_day, tag


def uplift_from_discount(rng: np.random.Generator, discount: int, group: str) -> int:
    """Uplift is derived from discount, with group sensitivity + noise + caps."""
    if group == "RETAIL":
        slope, base = 1.6, 6
    elif group == "ONLINE":
        slope, base = 1.3, 5
    else:  # WHOLESALE
        slope, base = 1.0, 4

    noise = rng.normal(0, 3.0)  # ±3% random variation
    uplift = base + slope * discount + noise
    uplift = max(5, min(55, uplift))  # cap
    return int(round(uplift))


def choose_customer(rng: np.random.Generator, group: str) -> str:
    """Pick a Level-1 customer within the chosen group, using weights."""
    if group == "RETAIL":
        choices = ["AH", "JUMBO", "PLUS"]
    elif group == "WHOLESALE":
        choices = ["SLIGRO", "HOLLANDFOODSERVICE", "GREENACRES", "MELEDI"]
    else:
        choices = ["PICNIC", "HELLOFRESH", "FLINK"]

    w = np.array([CUSTOMER_WEIGHTS.get(c, 0.1) for c in choices], dtype=float)
    w = w / w.sum()
    return str(rng.choice(choices, p=w))


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    products = pd.read_csv(BASE_DIR / "Product.csv", dtype=str).fillna("")
    channels = pd.read_csv(BASE_DIR / "Channel.csv", dtype=str).fillna("")
    locations = pd.read_csv(BASE_DIR / "Location.csv", dtype=str).fillna("")

    products["Level"] = pd.to_numeric(products["Level"], errors="coerce")
    channels["Level"] = pd.to_numeric(channels["Level"], errors="coerce")
    locations["Level"] = pd.to_numeric(locations["Level"], errors="coerce")

    # Product pools
    p1 = products.loc[products["Level"] == 1, "ProductID"].dropna().unique().tolist()
    p2 = products.loc[products["Level"] == 2, "ProductID"].dropna().unique().tolist()  # family level IDs, if present

    # Location pool (regions)
    l1 = locations.loc[locations["Level"] == 1, "LocationID"].dropna().unique().tolist()

    if not p1 or not l1:
        raise ValueError("Need Level-1 products and Level-1 locations in Product.csv/Location.csv.")

    # Verify that your Level-2 channel IDs exist (RETAIL/WHOLESALE/ONLINE)
    c2 = set(channels.loc[channels["Level"] == 2, "ChannelID"].dropna().str.strip())
    missing = [g for g in GROUPS if g not in c2]
    if missing:
        raise ValueError(f"Channel.csv is missing Level-2 groups: {missing}")

    promos = []
    promo_id = 1

    for year in range(START_YEAR, END_YEAR + 1):
        for _ in range(PROMOS_PER_YEAR):
            group = str(rng.choice(GROUPS, p=np.array(GROUP_PROBS)/sum(GROUP_PROBS)))

            start_day, season_tag = pick_start_date(rng, year)
            dur = int(rng.choice(DUR_OPTIONS, p=np.array(DUR_WEIGHTS)/sum(DUR_WEIGHTS)))
            end_day = start_day + timedelta(days=dur - 1)

            discount = int(rng.choice(DISCOUNT_OPTIONS, p=np.array(DISCOUNT_WEIGHTS)/sum(DISCOUNT_WEIGHTS)))
            uplift = uplift_from_discount(rng, discount, group)

            promo_level = str(rng.choice(PROMOLEVEL_OPTIONS, p=np.array(PROMOLEVEL_WEIGHTS)/sum(PROMOLEVEL_WEIGHTS)))
            loc_id = str(rng.choice(l1))

            if promo_level == "221" and p2:
                prod_id = str(rng.choice(p2))   # family-level product ID
                chan_id = group                 # group-level channel ID (RETAIL/WHOLESALE/ONLINE)
            else:
                promo_level = "111"
                prod_id = str(rng.choice(p1))   # variety-level product
                chan_id = choose_customer(rng, group)  # customer-level channel

            promos.append([
                f"P{promo_id:04d}",
                f"{season_tag} {group} Promo {promo_id:04d}",
                start_day.isoformat(),
                end_day.isoformat(),
                prod_id,
                chan_id,
                loc_id,
                promo_level,
                discount,
                uplift,
                f"season={season_tag}; group={group}; uplift=f(discount)+noise"
            ])
            promo_id += 1

    df = pd.DataFrame(promos, columns=[
        "PromoID", "PromoName", "StartDate", "EndDate",
        "ProductID", "ChannelID", "LocationID", "PromoLevel",
        "DiscountPct", "UpliftPct", "Notes"
    ])

    df.to_csv(OUT_PROMOS, index=False)
    print(f"[DONE] Wrote {OUT_PROMOS} with {len(df)} rows")
    print(df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()

