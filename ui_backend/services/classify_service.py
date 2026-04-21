from __future__ import annotations

from typing import Dict

from sqlalchemy import text

from core.db import ENGINE, get_engine, _qident, _qualified
from core.config import CLASSIFIED_FE_TABLE

_LAST_CLASSIFY_COMPUTED: Dict[str, str] = {}


def _ensure_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()


def _ensure_classified_fe_table(db_schema: str):
    _ensure_engine()

    qual = _qualified(db_schema, CLASSIFIED_FE_TABLE)

    with ENGINE.begin() as conn:
        conn.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {qual} (
          "scenario_id" INTEGER NOT NULL DEFAULT 1,
          "ProductID"  text NOT NULL,
          "ChannelID"  text NOT NULL,
          "LocationID" text NOT NULL,
          "Period"     text NOT NULL,
          "ADI"        double precision NULL,
          "CV2"        double precision NULL,
          "Category"   text NOT NULL,
          "Algorithm"  text NOT NULL,
          "CreatedAt"  timestamp with time zone NOT NULL DEFAULT now(),
          "UpdatedAt"  timestamp with time zone NOT NULL DEFAULT now()
        );
        """))

        conn.execute(text(f'''
            ALTER TABLE {qual}
            ADD COLUMN IF NOT EXISTS "scenario_id" INTEGER NOT NULL DEFAULT 1;
        '''))

        try:
            conn.execute(text(f'''
                ALTER TABLE {qual}
                DROP CONSTRAINT IF EXISTS {CLASSIFIED_FE_TABLE}_pkey;
            '''))
        except Exception:
            pass

        try:
            conn.execute(text(f'''
                ALTER TABLE {qual}
                DROP CONSTRAINT IF EXISTS ux_classified_forecast_elements_key;
            '''))
        except Exception:
            pass

        conn.execute(text(f'''
            DROP INDEX IF EXISTS {_qident(db_schema)}.{_qident("ux_classified_forecast_elements_key")};
        '''))

        conn.execute(text(f'''
            DROP INDEX IF EXISTS {_qident(db_schema)}.{_qident("classified_fe_uq_idx")};
        '''))

        conn.execute(text(f'''
            DROP INDEX IF EXISTS {_qident(db_schema)}.{_qident("classified_fe_lookup_idx")};
        '''))

        conn.execute(text(f'''
            CREATE UNIQUE INDEX IF NOT EXISTS classified_fe_uq_idx
            ON {qual}
            ("scenario_id","ProductID","ChannelID","LocationID","Period");
        '''))

        conn.execute(text(f'''
            CREATE INDEX IF NOT EXISTS classified_fe_lookup_idx
            ON {qual}
            ("scenario_id","Period","ProductID","ChannelID","LocationID");
        '''))