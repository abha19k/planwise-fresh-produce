from __future__ import annotations

from fastapi import APIRouter, Query, Depends, HTTPException
from sqlalchemy import text

from core.config import DEFAULT_SCHEMA
from core.db import ENGINE, get_engine, _qualified
from services.auth_service import require_roles, hash_password
from services.audit_service import write_audit_log

router = APIRouter()


def _ensure_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()


@router.get("/api/admin/audit-logs")
def api_admin_audit_logs(
    db_schema: str = Query(DEFAULT_SCHEMA),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    action: str = Query("", description="optional action filter"),
    entity: str = Query("", description="optional entity filter"),
    current_user=Depends(require_roles("admin")),
):
    _ensure_engine()

    where = ["1=1"]
    params = {"limit": limit, "offset": offset}

    if action.strip():
        where.append("a.action ILIKE :action")
        params["action"] = f"%{action.strip()}%"

    if entity.strip():
        where.append("a.entity ILIKE :entity")
        params["entity"] = f"%{entity.strip()}%"

    where_sql = " AND ".join(where)

    sql = text(f"""
        SELECT
            a.id,
            a.user_id::text AS user_id,
            u.email AS user_email,
            a.action,
            a.entity,
            a.entity_id,
            a.details,
            a.created_at::text AS created_at
        FROM {_qualified(db_schema, "audit_logs")} a
        LEFT JOIN {_qualified(db_schema, "users")} u
          ON u.id = a.user_id
        WHERE {where_sql}
        ORDER BY a.created_at DESC
        LIMIT :limit OFFSET :offset
    """)

    with ENGINE.begin() as conn:
        rows = conn.execute(sql, params).mappings().all()

    return [dict(r) for r in rows]


@router.get("/api/admin/users")
def api_admin_users(
    db_schema: str = Query(DEFAULT_SCHEMA),
    current_user=Depends(require_roles("admin")),
):
    _ensure_engine()

    sql = text(f"""
        SELECT
            u.id::text AS id,
            u.email,
            u.full_name,
            u.is_active,
            u.failed_login_attempts,
            u.locked_until::text AS locked_until,
            u.last_login_at::text AS last_login_at,
            u.created_at::text AS created_at,
            COALESCE(array_agg(r.name) FILTER (WHERE r.name IS NOT NULL), '{{}}') AS roles
        FROM {_qualified(db_schema, "users")} u
        LEFT JOIN {_qualified(db_schema, "user_roles")} ur
          ON ur.user_id = u.id
        LEFT JOIN {_qualified(db_schema, "roles")} r
          ON r.id = ur.role_id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    """)

    with ENGINE.begin() as conn:
        rows = conn.execute(sql).mappings().all()

    return [dict(r) for r in rows]


@router.post("/api/admin/users/{user_id}/disable")
def api_admin_disable_user(
    user_id: str,
    db_schema: str = Query(DEFAULT_SCHEMA),
    current_user=Depends(require_roles("admin")),
):
    _ensure_engine()

    sql = text(f"""
        UPDATE {_qualified(db_schema, "users")}
        SET is_active = FALSE
        WHERE id = CAST(:user_id AS uuid)
        RETURNING id::text AS id, email
    """)

    with ENGINE.begin() as conn:
        row = conn.execute(sql, {"user_id": user_id}).mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="User not found")

    write_audit_log(
        action="user.disable",
        entity="user",
        user_id=str(current_user["id"]),
        entity_id=user_id,
        details={"email": row["email"]},
        db_schema=db_schema,
    )

    return {"ok": True, "user": dict(row)}


@router.post("/api/admin/users/{user_id}/enable")
def api_admin_enable_user(
    user_id: str,
    db_schema: str = Query(DEFAULT_SCHEMA),
    current_user=Depends(require_roles("admin")),
):
    _ensure_engine()

    sql = text(f"""
        UPDATE {_qualified(db_schema, "users")}
        SET is_active = TRUE,
            failed_login_attempts = 0,
            locked_until = NULL
        WHERE id = CAST(:user_id AS uuid)
        RETURNING id::text AS id, email
    """)

    with ENGINE.begin() as conn:
        row = conn.execute(sql, {"user_id": user_id}).mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="User not found")

    write_audit_log(
        action="user.enable",
        entity="user",
        user_id=str(current_user["id"]),
        entity_id=user_id,
        details={"email": row["email"]},
        db_schema=db_schema,
    )

    return {"ok": True, "user": dict(row)}


@router.post("/api/admin/users/{user_id}/reset-password")
def api_admin_reset_password(
    user_id: str,
    new_password: str = Query(..., min_length=8),
    db_schema: str = Query(DEFAULT_SCHEMA),
    current_user=Depends(require_roles("admin")),
):
    _ensure_engine()

    password_hash = hash_password(new_password)

    sql = text(f"""
        UPDATE {_qualified(db_schema, "users")}
        SET password_hash = :password_hash,
            failed_login_attempts = 0,
            locked_until = NULL
        WHERE id = CAST(:user_id AS uuid)
        RETURNING id::text AS id, email
    """)

    with ENGINE.begin() as conn:
        row = conn.execute(
            sql,
            {
                "user_id": user_id,
                "password_hash": password_hash,
            },
        ).mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="User not found")

    write_audit_log(
        action="user.reset_password",
        entity="user",
        user_id=str(current_user["id"]),
        entity_id=user_id,
        details={"email": row["email"]},
        db_schema=db_schema,
    )

    return {"ok": True}