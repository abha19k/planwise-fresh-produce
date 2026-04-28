from __future__ import annotations

import json
from typing import Any, Dict, Optional

from sqlalchemy import text

from core.db import ENGINE, get_engine, _qualified
from core.config import DEFAULT_SCHEMA


def _ensure_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()


def write_audit_log(
    action: str,
    entity: str,
    user_id: Optional[str] = None,
    entity_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    db_schema: str = DEFAULT_SCHEMA,
):
    _ensure_engine()

    sql = text(f"""
    INSERT INTO {_qualified(db_schema, "audit_logs")}
      (user_id, action, entity, entity_id, details)
    VALUES
      (
        CASE
          WHEN :user_id IS NULL OR :user_id = ''
          THEN NULL
          ELSE CAST(:user_id AS uuid)
        END,
        :action,
        :entity,
        :entity_id,
        CAST(:details AS jsonb)
      )
    """)

    with ENGINE.begin() as conn:
        conn.execute(
            sql,
            {
                "user_id": user_id,
                "action": action,
                "entity": entity,
                "entity_id": entity_id,
                "details": json.dumps(details or {}, ensure_ascii=False),
            },
        )