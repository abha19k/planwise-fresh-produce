from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import psycopg2
from psycopg2 import sql

# =========================
# CONFIG
# =========================
DB_URI = "postgresql://abha:planwise123@localhost:5432/planwise"
SCHEMA = "planwise_fresh_produce"

CSV_TABLES: Dict[str, str] = {
    # master data
    "product": "Product.csv",
    "channel": "Channel.csv",
    "location": "Location.csv",
    "forecastelement": "ForecastElement.csv",

    # history
    "history_111_daily": "History_111_Daily.csv",
    "history_111_weekly": "History_111_Weekly.csv",
    "history_111_monthly": "History_111_Monthly.csv",
    "history_121_daily": "History_121_Daily.csv",
    "history_121_weekly": "History_121_Weekly.csv",
    "history_121_monthly": "History_121_Monthly.csv",
    "history_221_daily": "History_221_Daily.csv",
    "history_221_weekly": "History_221_Weekly.csv",
    "history_221_monthly": "History_221_Monthly.csv",

    # exogenous
    "promotions": "Promotions.csv",
    "weather_daily": "weather_daily.csv",

    # feature forecasts
    "forecast_111_daily": "Forecast_111_Daily_feat.csv",
    "forecast_121_daily": "Forecast_121_Daily_feat.csv",
    "forecast_221_daily": "Forecast_221_Daily_feat.csv",
    "forecast_111_weekly": "Forecast_111_Weekly_feat.csv",
    "forecast_121_weekly": "Forecast_121_Weekly_feat.csv",
    "forecast_221_weekly": "Forecast_221_Weekly_feat.csv",
    "forecast_111_monthly": "Forecast_111_Monthly_feat.csv",
    "forecast_121_monthly": "Forecast_121_Monthly_feat.csv",
    "forecast_221_monthly": "Forecast_221_Monthly_feat.csv",

    # baseline forecasts
    "forecast_111_daily_baseline": "Forecast_111_Daily_baseline.csv",
    "forecast_121_daily_baseline": "Forecast_121_Daily_baseline.csv",
    "forecast_221_daily_baseline": "Forecast_221_Daily_baseline.csv",
    "forecast_111_weekly_baseline": "Forecast_111_Weekly_baseline.csv",
    "forecast_121_weekly_baseline": "Forecast_121_Weekly_baseline.csv",
    "forecast_221_weekly_baseline": "Forecast_221_Weekly_baseline.csv",
    "forecast_111_monthly_baseline": "Forecast_111_Monthly_baseline.csv",
    "forecast_121_monthly_baseline": "Forecast_121_Monthly_baseline.csv",
    "forecast_221_monthly_baseline": "Forecast_221_Monthly_baseline.csv",
}

# =========================
# HELPERS: type inference
# =========================
def infer_pg_type(series: pd.Series) -> str:
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return "text"

    uniq = set(s.str.lower().unique().tolist())
    if uniq.issubset({"true", "false", "t", "f", "0", "1", "yes", "no"}):
        return "boolean"

    as_num = pd.to_numeric(s, errors="coerce")
    if as_num.notna().all():
        if (as_num.dropna() % 1 == 0).all():
            if as_num.max() > 2_147_483_647 or as_num.min() < -2_147_483_648:
                return "bigint"
            return "integer"
        return "double precision"

    # accept YYYY-MM-DD (most of your files)
    as_dt = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d")
    if as_dt.notna().all():
        return "date"

    return "text"


def read_csv_all_str(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


# =========================
# SQL builders
# =========================
def create_schema_sql() -> sql.SQL:
    return sql.SQL("CREATE SCHEMA IF NOT EXISTS {};").format(sql.Identifier(SCHEMA))


def table_exists(cur, table: str) -> bool:
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        );
        """,
        (SCHEMA, table),
    )
    return bool(cur.fetchone()[0])


def make_create_table_sql(table: str, df: pd.DataFrame) -> sql.SQL:
    cols = []
    for col in df.columns:
        pg_type = infer_pg_type(df[col])
        cols.append(sql.SQL("{} {}").format(sql.Identifier(col), sql.SQL(pg_type)))

    return sql.SQL("CREATE TABLE {}.{} ({});").format(
        sql.Identifier(SCHEMA),
        sql.Identifier(table),
        sql.SQL(", ").join(cols),
    )


def drop_table_sql(table: str) -> sql.SQL:
    return sql.SQL("DROP TABLE IF EXISTS {}.{};").format(
        sql.Identifier(SCHEMA),
        sql.Identifier(table),
    )


def truncate_table_sql(table: str) -> sql.SQL:
    return sql.SQL("TRUNCATE TABLE {}.{};").format(
        sql.Identifier(SCHEMA),
        sql.Identifier(table),
    )


def copy_into_table(cur, table: str, csv_path: str, columns: List[str]) -> None:
    with open(csv_path, "r", encoding="utf-8") as f:
        q = sql.SQL(
            "COPY {}.{} ({}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)"
        ).format(
            sql.Identifier(SCHEMA),
            sql.Identifier(table),
            sql.SQL(", ").join(map(sql.Identifier, columns)),
        )
        cur.copy_expert(q.as_string(cur), f)


def create_indexes(cur, table: str, cols: List[str]) -> None:
    # keys index
    if all(c in cols for c in ["ProductID", "ChannelID", "LocationID"]):
        cur.execute(
            sql.SQL('CREATE INDEX IF NOT EXISTS {} ON {}.{} ("ProductID","ChannelID","LocationID");').format(
                sql.Identifier(f"ix_{table}_pcl"),
                sql.Identifier(SCHEMA),
                sql.Identifier(table),
            )
        )

    # time index
    if "StartDate" in cols:
        cur.execute(
            sql.SQL('CREATE INDEX IF NOT EXISTS {} ON {}.{} ("StartDate");').format(
                sql.Identifier(f"ix_{table}_startdate"),
                sql.Identifier(SCHEMA),
                sql.Identifier(table),
            )
        )
    if "Date" in cols:
        cur.execute(
            sql.SQL('CREATE INDEX IF NOT EXISTS {} ON {}.{} ("Date");').format(
                sql.Identifier(f"ix_{table}_date"),
                sql.Identifier(SCHEMA),
                sql.Identifier(table),
            )
        )


def rowcount(cur, table: str) -> int:
    cur.execute(
        sql.SQL("SELECT COUNT(*) FROM {}.{};").format(sql.Identifier(SCHEMA), sql.Identifier(table))
    )
    return int(cur.fetchone()[0])


# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser(description="Load PlanWise Fresh Produce CSVs into Postgres.")
    ap.add_argument(
        "--mode",
        choices=["truncate", "recreate"],
        default="truncate",
        help="truncate: TRUNCATE existing tables and reload; recreate: DROP+CREATE+load",
    )
    args = ap.parse_args()

    # Validate files exist
    for t, p in CSV_TABLES.items():
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing file for table '{t}': {p}")

    conn = psycopg2.connect(DB_URI)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            cur.execute(create_schema_sql())

            for table, csv_path in CSV_TABLES.items():
                print(f"\n[LOAD] {table} <- {csv_path}")
                df = read_csv_all_str(csv_path)

                exists = table_exists(cur, table)

                if args.mode == "recreate":
                    cur.execute(drop_table_sql(table))
                    cur.execute(make_create_table_sql(table, df))
                else:
                    # truncate mode
                    if not exists:
                        cur.execute(make_create_table_sql(table, df))
                    else:
                        cur.execute(truncate_table_sql(table))

                # load
                copy_into_table(cur, table, csv_path, list(df.columns))

                # indexes
                create_indexes(cur, table, list(df.columns))

                # verify
                n = rowcount(cur, table)
                print(f"[OK] {SCHEMA}.{table} rows={n:,}")

            conn.commit()
            print("\n[DONE] Truncate+load completed successfully.")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
