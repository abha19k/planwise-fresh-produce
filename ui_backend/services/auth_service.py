from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List

import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import text

from core.db import ENGINE, get_engine, _qualified
from core.config import DEFAULT_SCHEMA, JWT_SECRET_KEY
import hashlib
from uuid import uuid4


SECRET_KEY = JWT_SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = 30
REFRESH_TOKEN_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer()


def _ensure_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_MINUTES)
    to_encode.update(
        {
            "exp": expire,
            "type": "access",
        }
    )
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_DAYS)
    to_encode.update(
        {
            "exp": expire,
            "type": "refresh",
        }
    )
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )


def get_user_by_email(
    email: str,
    db_schema: str = DEFAULT_SCHEMA,
) -> Optional[Dict[str, Any]]:
    _ensure_engine()

    sql = text(f"""
    SELECT
      id,
      email,
      password_hash,
      full_name,
      profile_image_url,
      is_active,
      created_at,
      failed_login_attempts,
      locked_until,
      last_login_at
    FROM {_qualified(db_schema, "users")}
    WHERE LOWER(email) = LOWER(:email)
    LIMIT 1
    """)

    with ENGINE.begin() as conn:
        row = conn.execute(sql, {"email": email}).mappings().first()

    return dict(row) if row is not None else None


def get_user_roles(
    user_id: str,
    db_schema: str = DEFAULT_SCHEMA,
) -> List[str]:
    _ensure_engine()

    sql = text(f"""
    SELECT r.name
    FROM {_qualified(db_schema, "user_roles")} ur
    JOIN {_qualified(db_schema, "roles")} r
      ON r.id = ur.role_id
    WHERE ur.user_id = CAST(:user_id AS uuid)
    ORDER BY r.name
    """)

    with ENGINE.begin() as conn:
        rows = conn.execute(sql, {"user_id": user_id}).scalars().all()

    return list(rows)


def get_primary_role(
    user_id: str,
    db_schema: str = DEFAULT_SCHEMA,
) -> str:
    roles = get_user_roles(user_id, db_schema)
    if "admin" in roles:
        return "admin"
    if "planner" in roles:
        return "planner"
    if "viewer" in roles:
        return "viewer"
    return "viewer"


def authenticate_user(
    email: str,
    password: str,
    db_schema: str = DEFAULT_SCHEMA,
):
    user = get_user_by_email(email, db_schema)

    # Do not reveal whether email exists
    if not user:
        return None

    if not user["is_active"]:
        return None

    if is_user_locked(user):
        raise HTTPException(
            status_code=423,
            detail="Account temporarily locked due to too many failed login attempts. Try again later.",
        )

    if not verify_password(password, user["password_hash"]):
        record_failed_login(email, db_schema)
        return None

    record_successful_login(email, db_schema)

    roles = get_user_roles(str(user["id"]), db_schema)

    user["roles"] = roles
    user["role"] = get_primary_role(str(user["id"]), db_schema)

    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db_schema: str = DEFAULT_SCHEMA,
):
    token = credentials.credentials
    payload = decode_token(token)

    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = get_user_by_email(email, db_schema)

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="Inactive user")

    user["roles"] = get_user_roles(str(user["id"]), db_schema)
    user["role"] = get_primary_role(str(user["id"]), db_schema)

    return user


def require_roles(*roles):
    def checker(user=Depends(get_current_user)):
        user_roles = set(user.get("roles", []))
        if not any(role in user_roles for role in roles):
            raise HTTPException(
                status_code=403,
                detail="Permission denied",
            )
        return user

    return checker


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def store_refresh_token(
    user_id: str,
    refresh_token: str,
    db_schema: str = DEFAULT_SCHEMA,
):
    _ensure_engine()

    token_hash = hash_token(refresh_token)

    sql = text(f"""
    INSERT INTO {_qualified(db_schema, "refresh_tokens")}
      (user_id, token_hash, expires_at, revoked, created_at)
    VALUES
      (
        CAST(:user_id AS uuid),
        :token_hash,
        NOW() + INTERVAL '{REFRESH_TOKEN_DAYS} days',
        FALSE,
        NOW()
      )
    """)

    with ENGINE.begin() as conn:
        conn.execute(sql, {
            "user_id": user_id,
            "token_hash": token_hash,
        })


def is_refresh_token_valid(
    refresh_token: str,
    db_schema: str = DEFAULT_SCHEMA,
) -> bool:
    _ensure_engine()

    token_hash = hash_token(refresh_token)

    sql = text(f"""
    SELECT 1
    FROM {_qualified(db_schema, "refresh_tokens")}
    WHERE token_hash = :token_hash
      AND revoked = FALSE
      AND expires_at > NOW()
    LIMIT 1
    """)

    with ENGINE.begin() as conn:
        row = conn.execute(sql, {"token_hash": token_hash}).scalar()

    return bool(row)


def revoke_refresh_token(
    refresh_token: str,
    db_schema: str = DEFAULT_SCHEMA,
):
    _ensure_engine()

    token_hash = hash_token(refresh_token)

    sql = text(f"""
    UPDATE {_qualified(db_schema, "refresh_tokens")}
    SET revoked = TRUE
    WHERE token_hash = :token_hash
      AND revoked = FALSE
    """)

    with ENGINE.begin() as conn:
        conn.execute(sql, {"token_hash": token_hash})

def is_user_locked(user: Dict[str, Any]) -> bool:
    locked_until = user.get("locked_until")
    if not locked_until:
        return False

    now = datetime.now(timezone.utc)

    # PostgreSQL may return timezone-aware datetime already
    if locked_until.tzinfo is None:
        locked_until = locked_until.replace(tzinfo=timezone.utc)

    return locked_until > now


def record_failed_login(
    email: str,
    db_schema: str = DEFAULT_SCHEMA,
):
    _ensure_engine()

    sql = text(f"""
    UPDATE {_qualified(db_schema, "users")}
    SET
      failed_login_attempts = failed_login_attempts + 1,
      locked_until = CASE
        WHEN failed_login_attempts + 1 >= 5
        THEN NOW() + INTERVAL '15 minutes'
        ELSE locked_until
      END
    WHERE LOWER(email) = LOWER(:email)
    """)

    with ENGINE.begin() as conn:
        conn.execute(sql, {"email": email})


def record_successful_login(
    email: str,
    db_schema: str = DEFAULT_SCHEMA,
):
    _ensure_engine()

    sql = text(f"""
    UPDATE {_qualified(db_schema, "users")}
    SET
      failed_login_attempts = 0,
      locked_until = NULL,
      last_login_at = NOW()
    WHERE LOWER(email) = LOWER(:email)
    """)

    with ENGINE.begin() as conn:
        conn.execute(sql, {"email": email})