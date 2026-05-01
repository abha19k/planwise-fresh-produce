from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Depends, UploadFile, File
from sqlalchemy import text
from core.db import ENGINE, get_engine, _qualified
from core.config import DEFAULT_SCHEMA
from services.audit_service import write_audit_log
import os

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
    store_refresh_token,
    is_refresh_token_valid,
    revoke_refresh_token,
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
        write_audit_log(
            action="login.failed",
            entity="auth",
            user_id=None,
            entity_id=body.email,
            details={"email": body.email},
            db_schema=db_schema,
        )
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    access_token = create_access_token({"sub": user["email"]})
    refresh_token = create_refresh_token({"sub": user["email"]})

    store_refresh_token(
    user_id=str(user["id"]),
    refresh_token=refresh_token,
    db_schema=db_schema,
    )

    write_audit_log(
    action="login.success",
    entity="auth",
    user_id=str(user["id"]),
    entity_id=str(user["id"]),
    details={"email": user["email"]},
    db_schema=db_schema,
    )

    return {
        "user": {
            "id": str(user["id"]),
            "email": user["email"],
            "full_name": user.get("full_name"),
            "role": user["role"],
            "roles": user.get("roles", []),
            "is_active": user["is_active"],
            "profile_image_url": user.get("profile_image_url"),
        },
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": 30 * 60,
    }


@router.post("/api/refresh", response_model=TokenResponse)
def api_refresh(
    body: dict,
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    refresh_token = body.get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=400, detail="refresh_token is required")

    if not is_refresh_token_valid(refresh_token, db_schema):
        raise HTTPException(status_code=401, detail="Refresh token revoked or expired")

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
def api_logout(
    body: dict,
    db_schema: str = Query(DEFAULT_SCHEMA),
):
    refresh_token = body.get("refresh_token")
    if refresh_token:
        revoke_refresh_token(refresh_token, db_schema)

    write_audit_log(
        action="logout",
        entity="auth",
        user_id=None,
        entity_id=None,
        details={"refresh_token_revoked": bool(refresh_token)},
        db_schema=db_schema,
    )

    return {"ok": True}

@router.get("/api/me", response_model=UserOut)
def api_me(user=Depends(get_current_user)):
    return {
        "id": str(user["id"]),
        "email": user["email"],
        "full_name": user.get("full_name"),
        "role": user["role"],
        "roles": user.get("roles", []),
        "profile_image_url": user.get("profile_image_url"),
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

    write_audit_log(
        action="user.create",
        entity="user",
        user_id=str(current_user["id"]),
        entity_id=str(user_row["id"]),
        details={
            "email": user_row["email"],
            "role": body.role,
        },
        db_schema=db_schema,
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
            "profile_image_url": None,
        },
    }

@router.post("/api/users/{user_id}/upload-avatar")
def upload_avatar(
    user_id: str,
    file: UploadFile = File(...),
    db_schema: str = Query(DEFAULT_SCHEMA),
    current_user=Depends(require_roles("admin", "planner")),
):
    _ensure_engine()

    folder = "uploads/avatars"
    os.makedirs(folder, exist_ok=True)

    filename = f"{user_id}_{file.filename}"
    filepath = os.path.join(folder, filename)

    with open(filepath, "wb") as f:
        f.write(file.file.read())

    url = f"/uploads/avatars/{filename}"

    sql = text(f"""
        UPDATE {_qualified(db_schema, "users")}
        SET profile_image_url = :url
        WHERE id = :user_id
    """)

    with ENGINE.begin() as conn:
        conn.execute(sql, {"url": url, "user_id": user_id})

    return {"ok": True, "url": url}

