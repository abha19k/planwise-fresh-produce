import pandas as pd
from itertools import product

PRODUCT_FILE = "Product.csv"
CHANNEL_FILE = "Channel.csv"
LOCATION_FILE = "Location.csv"
OUTPUT_FILE = "ForecastElement.csv"

# Load
products = pd.read_csv(PRODUCT_FILE)
channels = pd.read_csv(CHANNEL_FILE)
locations = pd.read_csv(LOCATION_FILE)

# Normalize
products["Level"] = pd.to_numeric(products["Level"], errors="coerce")
channels["Level"] = pd.to_numeric(channels["Level"], errors="coerce")
locations["Level"] = pd.to_numeric(locations["Level"], errors="coerce")

products["ProductID"] = products["ProductID"].str.strip()
channels["ChannelID"] = channels["ChannelID"].str.strip()
locations["LocationID"] = locations["LocationID"].str.strip()

# Split levels
P1 = products.loc[products["Level"] == 1, "ProductID"].unique()
P2 = products.loc[products["Level"] == 2, "ProductID"].unique()

C1 = channels.loc[channels["Level"] == 1, "ChannelID"].unique()
C2 = channels.loc[channels["Level"] == 2, "ChannelID"].unique()

L1 = locations.loc[locations["Level"] == 1, "LocationID"].unique()

rows = []

# 221
rows += [[p, c, l, "221", True] for p, c, l in product(P2, C2, L1)]

# 121
rows += [[p, c, l, "121", True] for p, c, l in product(P1, C2, L1)]

# 111
rows += [[p, c, l, "111", True] for p, c, l in product(P1, C1, L1)]

df = pd.DataFrame(
    rows,
    columns=["ProductID", "ChannelID", "LocationID", "Level", "IsActive"]
).drop_duplicates()

df.to_csv(OUTPUT_FILE, index=False)

print(df.groupby("Level").size())
print("Total rows:", len(df))
