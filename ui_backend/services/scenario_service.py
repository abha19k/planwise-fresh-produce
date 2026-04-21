from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy import text

from core.db import ENGINE, get_engine, _qident, _qualified
from core.config import (
    SCENARIO_TABLE,
    SCENARIO_OVERRIDE_TABLE,
)


def _ensure_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()


def _ensure_scenario_tables(db_schema: str):
    _ensure_engine()

    ddl = f"""
    CREATE TABLE IF NOT EXISTS {_qident(db_schema)}.scenario (
      scenario_id BIGSERIAL PRIMARY KEY,
      name TEXT NOT NULL,
      parent_scenario_id BIGINT NULL REFERENCES {_qident(db_schema)}.scenario(scenario_id),
      is_base BOOLEAN NOT NULL DEFAULT FALSE,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      created_by TEXT NULL,
      status TEXT NOT NULL DEFAULT 'active'
    );

    CREATE TABLE IF NOT EXISTS {_qident(db_schema)}.scenario_override (
      scenario_id BIGINT NOT NULL,
      table_name TEXT NOT NULL,
      pk JSONB NOT NULL,
      row JSONB NULL,
      is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      updated_by TEXT NULL,
      PRIMARY KEY (scenario_id, table_name, pk)
    );
    """

    with ENGINE.begin() as conn:
        for stmt in ddl.split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s + ";"))


def _get_base_scenario_id(db_schema: str) -> int:
    _ensure_engine()
    _ensure_scenario_tables(db_schema)

    sql = text(
        f'SELECT scenario_id FROM {_qident(db_schema)}.scenario WHERE is_base=true LIMIT 1'
    )

    with ENGINE.begin() as conn:
        sid = conn.execute(sql).scalar()

    if sid:
        return int(sid)

    ins = text(f"""
        INSERT INTO {_qident(db_schema)}.scenario
        (name, is_base)
        VALUES ('Base', true)
        RETURNING scenario_id
    """)

    with ENGINE.begin() as conn:
        sid = conn.execute(ins).scalar_one()

    return int(sid)


def _get_scenario_chain(db_schema: str, scenario_id: int) -> List[int]:
    _ensure_engine()

    if not scenario_id:
        return [_get_base_scenario_id(db_schema)]

    chain = []
    seen = set()
    cur = int(scenario_id)

    while cur and cur not in seen:
        seen.add(cur)
        chain.append(cur)

        sql = text(f"""
            SELECT parent_scenario_id
            FROM {_qident(db_schema)}.scenario
            WHERE scenario_id=:sid
        """)

        with ENGINE.begin() as conn:
            parent = conn.execute(sql, {"sid": cur}).scalar()

        if parent is None:
            break

        cur = int(parent)

    base_id = _get_base_scenario_id(db_schema)

    if base_id not in chain:
        chain.append(base_id)

    return chain


def _json_key(d: Dict[str, Any]) -> str:
    def norm(v):
        if isinstance(v, (date, datetime)):
            return v.isoformat()
        if isinstance(v, dict):
            return {k: norm(val) for k, val in v.items()}
        if isinstance(v, list):
            return [norm(x) for x in v]
        return v

    return json.dumps(norm(d), sort_keys=True, ensure_ascii=False)


def _read_overrides_for_chain(
    db_schema: str,
    table_name: str,
    chain: List[int],
):
    _ensure_engine()

    if not chain:
        return {}

    qual = _qualified(db_schema, SCENARIO_OVERRIDE_TABLE)

    sql = text(f"""
      SELECT *
      FROM {qual}
      WHERE table_name=:tname
        AND scenario_id = ANY(:chain)
    """)

    with ENGINE.begin() as conn:
        rows = conn.execute(
            sql,
            {"tname": table_name, "chain": chain}
        ).mappings().all()

    rank = {sid: i for i, sid in enumerate(chain)}

    best = {}
    best_rank = {}

    for r in rows:
        pk = dict(r["pk"] or {})
        k = _json_key(pk)
        rr = rank[int(r["scenario_id"])]

        if k not in best_rank or rr < best_rank[k]:
            best[k] = dict(r)
            best_rank[k] = rr

    return best