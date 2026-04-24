from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy import text
from services.auth_service import require_roles
from core.db import ENGINE, get_engine, _qualified
from core.config import DEFAULT_SCHEMA

from models.auth import (
    LoginRequest,
    LoginResponse,
    TokenResponse,
    UserOut,
    UserCreate,
)

from services.auth_service import (
    hash_password,
    authenticate_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    require_roles,
)

router = APIRouter()


def _ensure_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()


@router.post("/api/login", response_model=LoginResponse)
def api_login(
    body: LoginRequest,
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    _ensure_engine()

    user = authenticate_user(body.email, body.password, db_schema)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token = create_access_token({"sub": user["email"]})
    _ = create_refresh_token({"sub": user["email"]})

    return {
        "user": {
            "id": str(user["id"]),
            "email": user["email"],
            "full_name": user.get("full_name"),
            "role": user["role"],
            "roles": user.get("roles", []),
            "is_active": user["is_active"],
        },
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 30 * 60,
    }


@router.post("/api/refresh", response_model=TokenResponse)
def api_refresh(body: dict):
    refresh_token = body.get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=400, detail="refresh_token is required")

    payload = decode_token(refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    access_token = create_access_token({"sub": email})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 30 * 60,
    }


@router.post("/api/logout")
def api_logout():
    return {"ok": True}


@router.get("/api/me", response_model=UserOut)
def api_me(user=Depends(get_current_user)):
    return {
        "id": str(user["id"]),
        "email": user["email"],
        "full_name": user.get("full_name"),
        "role": user["role"],
        "roles": user.get("roles", []),
        "is_active": user["is_active"],
    }


@router.post("/api/users/create")
def api_create_user(
    body: UserCreate,
    db_schema: str = Query(DEFAULT_SCHEMA),
    current_user=Depends(require_roles("admin")),
):
    _ensure_engine()

    existing_sql = text(f"""
        SELECT 1
        FROM {_qualified(db_schema, "users")}
        WHERE LOWER(email) = LOWER(:email)
        LIMIT 1
    """)

    with ENGINE.begin() as conn:
        exists = conn.execute(existing_sql, {"email": body.email}).scalar()

    if exists:
        raise HTTPException(status_code=400, detail="Email already exists")

    password_hash = hash_password(body.password)

    insert_user_sql = text(f"""
        INSERT INTO {_qualified(db_schema, "users")}
        (email, password_hash, full_name, is_active)
        VALUES
        (:email, :password_hash, :full_name, TRUE)
        RETURNING id, email, full_name, is_active
    """)

    role_sql = text(f"""
        INSERT INTO {_qualified(db_schema, "user_roles")} (user_id, role_id)
        SELECT CAST(:user_id AS uuid), r.id
        FROM {_qualified(db_schema, "roles")} r
        WHERE r.name = :role
    """)

    with ENGINE.begin() as conn:
        user_row = conn.execute(
            insert_user_sql,
            {
                "email": body.email,
                "password_hash": password_hash,
                "full_name": body.full_name,
            },
        ).mappings().first()

        conn.execute(
            role_sql,
            {
                "user_id": str(user_row["id"]),
                "role": body.role,
            },
        )

    return {
        "ok": True,
        "user": {
            "id": str(user_row["id"]),
            "email": user_row["email"],
            "full_name": user_row.get("full_name"),
            "role": body.role,
            "roles": [body.role],
            "is_active": user_row["is_active"],
        },
    }