import secrets
import time
import re
from datetime import date, datetime
from decimal import Decimal

from fastapi import HTTPException, Request, Response
from pydantic import BaseModel

from core.config import (
    CAM_TOKEN,
    GBOX_ALLOWED_DOMAIN,
    GOOGLE_OAUTH_CLIENT_ID,
    STAFF_REGISTRATION_CODE,
    STAFF_REGISTRATION_ENABLED,
    SESSION_COOKIE_NAME,
    SESSION_COOKIE_SAMESITE,
    SESSION_COOKIE_SECURE,
    SESSION_TTL_SECONDS,
    STUDENT_SESSION_TTL_SECONDS,
)
from database.database_handler import close_db_resources, get_db_pool
from services import cache_service


_sessions: dict[str, tuple[str, float]] = {}


class LoginBody(BaseModel):
    username: str
    password: str


class RegisterBody(BaseModel):
    username: str
    password: str
    full_name: str | None = None
    registration_code: str | None = None


USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


def _is_pool_exhausted(exc: Exception) -> bool:
    return exc.__class__.__name__ == "PoolExhausted"


def _is_operational_error(exc: Exception) -> bool:
    return exc.__class__.__name__ == "OperationalError"


def get_session_token(request: Request) -> str | None:
    return (
        request.cookies.get(SESSION_COOKIE_NAME)
        or request.headers.get("X-Session-Token")
    )


def _session_cache_key(token: str) -> str:
    return f"session:{token}"


def _store_session(token: str, username: str) -> None:
    expires_at = time.time() + max(60, SESSION_TTL_SECONDS)
    _sessions[token] = (username, expires_at)
    cache_service.set_json(
        _session_cache_key(token),
        {"username": username},
        SESSION_TTL_SECONDS,
    )


def _lookup_session(token: str | None) -> str | None:
    if not token:
        return None

    record = _sessions.get(token)
    if record:
        username, expires_at = record
        if expires_at > time.time():
            return username
        _sessions.pop(token, None)

    cached = cache_service.get_json(_session_cache_key(token))
    if isinstance(cached, dict) and cached.get("username"):
        username = str(cached["username"])
        _sessions[token] = (
            username,
            time.time() + max(60, SESSION_TTL_SECONDS),
        )
        return username

    return None


def is_authenticated(request: Request) -> bool:
    return _lookup_session(get_session_token(request)) is not None


def require_staff(request: Request) -> str:
    token = get_session_token(request)
    username = _lookup_session(token)
    if not username:
        raise HTTPException(status_code=401, detail="Login required")
    return username


def get_session_username(request: Request, default: str = "") -> str:
    return _lookup_session(get_session_token(request)) or default


def create_session(username: str) -> str:
    token = secrets.token_hex(32)
    _store_session(token, username)
    return token


def clear_session(token: str | None) -> str | None:
    if not token:
        return None
    record = _sessions.pop(token, None)
    cache_service.delete(_session_cache_key(token))
    if record:
        return record[0]
    return None


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=SESSION_COOKIE_SECURE,
        samesite=SESSION_COOKIE_SAMESITE,
    )


def delete_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        secure=SESSION_COOKIE_SECURE,
        httponly=True,
        samesite=SESSION_COOKIE_SAMESITE,
    )


# STUDENT SESSIONS (mobile app)
#
# Deliberately a separate token namespace and cache-key prefix from staff
# sessions above — a student token must never be usable against a
# require_staff() endpoint, or vice versa. Mobile has no cookie jar, so
# students authenticate via a bearer token only (Authorization header or
# X-Student-Session-Token), never a cookie.

_student_sessions: dict[str, tuple[int, float]] = {}


def get_student_session_token(request: Request) -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip() or None
    return request.headers.get("X-Student-Session-Token")


def _student_session_cache_key(token: str) -> str:
    return f"student_session:{token}"


def _store_student_session(token: str, student_id: int) -> None:
    expires_at = time.time() + max(60, STUDENT_SESSION_TTL_SECONDS)
    _student_sessions[token] = (student_id, expires_at)
    cache_service.set_json(
        _student_session_cache_key(token),
        {"student_id": student_id},
        STUDENT_SESSION_TTL_SECONDS,
    )


def _lookup_student_session(token: str | None) -> int | None:
    if not token:
        return None

    record = _student_sessions.get(token)
    if record:
        student_id, expires_at = record
        if expires_at > time.time():
            return student_id
        _student_sessions.pop(token, None)

    cached = cache_service.get_json(_student_session_cache_key(token))
    if isinstance(cached, dict) and cached.get("student_id") is not None:
        student_id = int(cached["student_id"])
        _student_sessions[token] = (
            student_id,
            time.time() + max(60, STUDENT_SESSION_TTL_SECONDS),
        )
        return student_id

    return None


def create_student_session(student_id: int) -> str:
    token = secrets.token_hex(32)
    _store_student_session(token, student_id)
    return token


def require_student(request: Request) -> int:
    token = get_student_session_token(request)
    student_id = _lookup_student_session(token)
    if student_id is None:
        raise HTTPException(status_code=401, detail="Login required")
    return student_id


def clear_student_session(token: str | None) -> int | None:
    if not token:
        return None
    record = _student_sessions.pop(token, None)
    cache_service.delete(_student_session_cache_key(token))
    if record:
        return record[0]
    return None


class GoogleIdentity(BaseModel):
    sub: str
    email: str
    name: str | None = None


def verify_google_id_token(id_token_str: str) -> GoogleIdentity:
    """Verify a Google ID token from the Flutter app's Sign-in-with-Google
    flow: valid signature, correct audience (our OAuth client), and email
    on the NCF Gbox domain. Raises HTTPException on any failure — this is
    the sole credential check for student accounts, so it fails closed.
    """
    if not GOOGLE_OAUTH_CLIENT_ID:
        raise HTTPException(
            status_code=503,
            detail="Google sign-in is not configured on this server.",
        )

    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token as google_id_token

        claims = google_id_token.verify_oauth2_token(
            id_token_str, google_requests.Request(), GOOGLE_OAUTH_CLIENT_ID,
        )
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid Google sign-in token: {exc}")

    if not claims.get("email_verified", False):
        raise HTTPException(status_code=401, detail="Google account email is not verified.")

    email = str(claims.get("email", "")).strip().lower()
    if GBOX_ALLOWED_DOMAIN and not email.endswith(f"@{GBOX_ALLOWED_DOMAIN.lower()}"):
        raise HTTPException(
            status_code=403,
            detail=f"Sign in with your @{GBOX_ALLOWED_DOMAIN} Gbox account.",
        )

    return GoogleIdentity(sub=str(claims["sub"]), email=email, name=claims.get("name"))


def verify_cam_token(request: Request) -> None:
    if request.headers.get("X-CAM-TOKEN", "") != CAM_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid CAM_TOKEN")


def verify_password(stored: str, entered: str) -> bool:
    if stored == entered:
        return True
    try:
        from werkzeug.security import check_password_hash

        return check_password_hash(stored, entered)
    except Exception:
        return False


def hash_password(password: str) -> str:
    from werkzeug.security import generate_password_hash

    return generate_password_hash(password)


def registration_is_enabled() -> bool:
    return STAFF_REGISTRATION_ENABLED


def registration_code_is_required() -> bool:
    return bool(STAFF_REGISTRATION_CODE)


def _validate_registration_input(
    username: str,
    password: str,
    registration_code: str | None = None,
) -> str:
    if not STAFF_REGISTRATION_ENABLED:
        raise HTTPException(status_code=403, detail="Staff registration is disabled.")

    username = (username or "").strip()
    if not USERNAME_PATTERN.match(username):
        raise HTTPException(
            status_code=400,
            detail="Username must be 3-32 characters and may use letters, numbers, underscore, dot, or dash.",
        )
    if len(password or "") < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters.",
        )
    if STAFF_REGISTRATION_CODE and registration_code != STAFF_REGISTRATION_CODE:
        raise HTTPException(status_code=403, detail="Invalid staff registration code.")

    return username


def _users_table_columns(cursor) -> set[str]:
    cursor.execute("SHOW COLUMNS FROM users")
    columns = set()
    for row in cursor.fetchall():
        if isinstance(row, dict):
            columns.add(str(row.get("Field", "")).lower())
        elif row:
            columns.add(str(row[0]).lower())
    return columns


def register_staff_user(
    username: str,
    password: str,
    full_name: str | None = None,
    registration_code: str | None = None,
) -> dict:
    username = _validate_registration_input(username, password, registration_code)
    full_name = (full_name or "").strip()

    db_pool = get_db_pool()
    if db_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable. Please try again later.",
        )

    conn = cursor = None
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT username FROM users WHERE username=%s LIMIT 1",
            (username,),
        )
        if cursor.fetchone():
            raise HTTPException(status_code=409, detail="Username is already registered.")

        columns = _users_table_columns(cursor)
        values = {
            "username": username,
            "password": hash_password(password),
        }
        expressions = {}

        if "is_active" in columns:
            values["is_active"] = 1
        if full_name and "full_name" in columns:
            values["full_name"] = full_name
        elif full_name and "name" in columns:
            values["name"] = full_name
        if "role" in columns:
            values["role"] = "staff"
        if "created_at" in columns:
            expressions["created_at"] = "NOW()"
        if "updated_at" in columns:
            expressions["updated_at"] = "NOW()"

        missing = [field for field in ("username", "password") if field not in columns]
        if missing:
            raise HTTPException(
                status_code=500,
                detail=f"users table is missing required column(s): {', '.join(missing)}",
            )

        fields = list(values.keys()) + list(expressions.keys())
        placeholders = ["%s"] * len(values) + list(expressions.values())
        sql = (
            f"INSERT INTO users ({', '.join(fields)}) "
            f"VALUES ({', '.join(placeholders)})"
        )
        cursor.execute(sql, tuple(values.values()))
        conn.commit()
        print(f"[Register] staff user '{username}' created")
        return {"username": username, "full_name": full_name or None}
    except HTTPException:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        raise
    except Exception as exc:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        if _is_pool_exhausted(exc):
            raise HTTPException(
                status_code=503,
                detail="Server is busy. Please try again shortly.",
            )
        if exc.__class__.__name__ == "IntegrityError":
            raise HTTPException(status_code=409, detail="Username is already registered.")
        print(f"[Register] DB error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Unable to register staff account. Check the users table structure.",
        )
    finally:
        close_db_resources(cursor, conn)


def authenticate_user(username: str, password: str) -> dict:
    username = (username or "").strip()
    if not username or not password:
        raise HTTPException(
            status_code=400,
            detail="Username and password are required.",
        )

    db_pool = get_db_pool()
    if db_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable. Please try again later.",
        )

    conn = cursor = row = None
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM users WHERE username=%s LIMIT 1",
            (username,),
        )
        row = cursor.fetchone()
    except Exception as exc:
        if _is_pool_exhausted(exc):
            raise HTTPException(
                status_code=503,
                detail="Server is busy. Please try again shortly.",
            )
        if _is_operational_error(exc):
            print(f"[Auth] DB operational error: {exc}")
            raise HTTPException(
                status_code=503,
                detail="Database connection failed. Please try again.",
            )
        print(f"[Auth] DB error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again.",
        )
    finally:
        close_db_resources(cursor, conn)

    if row is None or not verify_password(row["password"], password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not row.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account is disabled")

    return row


def _clean_user_profile(row: dict) -> dict:
    profile = {}
    for key, value in row.items():
        lowered = key.lower()
        if "password" in lowered:
            continue
        if isinstance(value, (datetime, date)):
            profile[key] = value.isoformat()
        elif isinstance(value, Decimal):
            profile[key] = float(value)
        else:
            profile[key] = value
    return profile


def get_user_profile(username: str) -> dict:
    db_pool = get_db_pool()
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    conn = cursor = row = None
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM users WHERE username=%s LIMIT 1",
            (username,),
        )
        row = cursor.fetchone()
    except Exception as exc:
        if _is_pool_exhausted(exc):
            raise HTTPException(
                status_code=503,
                detail="Server is busy. Please try again shortly.",
            )
        print(f"[Profile] DB error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Unable to load profile.",
        )
    finally:
        close_db_resources(cursor, conn)

    if row is None:
        raise HTTPException(status_code=404, detail="User profile not found")

    profile = _clean_user_profile(row)
    profile.setdefault("username", username)
    return profile


def touch_last_login(username: str) -> None:
    db_pool = get_db_pool()
    if db_pool is None:
        return

    conn = cursor = None
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET last_login=NOW() WHERE username=%s",
            (username,),
        )
        conn.commit()
    except Exception:
        pass
    finally:
        close_db_resources(cursor, conn)
