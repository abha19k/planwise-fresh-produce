from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import create_engine

from core.config import DATABASE_URL

ENGINE = None


def get_engine():
    global ENGINE
    if ENGINE is None:
        if not DATABASE_URL:
            raise HTTPException(
                status_code=500,
                detail="DATABASE_URL is not set. Example: postgresql+psycopg2://user:pass@host:5432/dbname",
            )
        ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True)
    return ENGINE


def _qident(s: str) -> str:
    s = str(s).replace('"', '""')
    return f'"{s}"'


def _qualified(schema: str, table: str) -> str:
    return f"{_qident(schema)}.{_qident(table)}"