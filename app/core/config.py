import os
import secrets
from pathlib import Path

from dotenv import load_dotenv


APP_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_ROOT.parent

load_dotenv(PROJECT_ROOT / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_list(name: str, default: list[str] | None = None) -> list[str]:
    raw = os.getenv(name, "")
    if not raw.strip():
        return list(default or [])
    return [item.strip() for item in raw.split(",") if item.strip()]


APP_ENV = os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "development")).strip().lower()
IS_PRODUCTION = APP_ENV in {"prod", "production"}

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = env_int("PORT", env_int("API_PORT", 5000))
PORTAL_BASE_URL = os.getenv("PORTAL_BASE_URL", "http://localhost:5000").strip().rstrip("/")

CORS_ORIGINS = env_list(
    "CORS_ORIGINS",
    ["http://localhost:3000", "http://localhost:5000", PORTAL_BASE_URL],
)
TRUSTED_HOSTS = env_list("TRUSTED_HOSTS", ["*"])

SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "session_token").strip() or "session_token"
SESSION_COOKIE_SECURE = env_bool("SESSION_COOKIE_SECURE", IS_PRODUCTION)
SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "lax").strip().lower() or "lax"
if SESSION_COOKIE_SAMESITE not in {"lax", "strict", "none"}:
    SESSION_COOKIE_SAMESITE = "lax"
SESSION_TTL_SECONDS = env_int("SESSION_TTL_SECONDS", 60 * 60 * 8)

STAFF_REGISTRATION_ENABLED = env_bool(
    "STAFF_REGISTRATION_ENABLED",
    not IS_PRODUCTION,
)
STAFF_REGISTRATION_CODE = os.getenv("STAFF_REGISTRATION_CODE", "").strip()

HISTORY_LEN = 60

API_HIGH_CONF = env_float("API_HIGH_CONF", 0.55)
LOW_CONF_BOOST = env_int("LOW_CONF_BOOST", 2)
API_MIN_BBOX_AREA = env_int(
    "API_MIN_BBOX_AREA",
    env_int("MIN_BBOX_AREA", 3000),
)

QUEUE_MIN_MOTION_PIXELS = env_int("QUEUE_MIN_MOTION_PIXELS", 8)
QUEUE_STATIC_STDEV_THRESHOLD = env_float("QUEUE_STATIC_STDEV_THRESHOLD", 1.5)
QUEUE_STATIC_CONF_BYPASS = env_float("QUEUE_STATIC_CONF_BYPASS", 0.45)
QUEUE_MIN_PORTRAIT_ASPECT = env_float("QUEUE_MIN_PORTRAIT_ASPECT", 0.60)
QUEUE_MIN_CONFIRM_FRAMES = env_int("QUEUE_MIN_CONFIRM_FRAMES", 14)
QUEUE_MAX_MISSING_FRAMES = env_int("QUEUE_MAX_MISSING_FRAMES", 240)
QUEUE_NOSHOW_WINDOW_SECONDS = env_int("QUEUE_NOSHOW_WINDOW_SECONDS", 300)
QUEUE_AUTO_NOSHOW_ENABLED = env_bool("QUEUE_AUTO_NOSHOW_ENABLED", False)
QUEUE_RECENCY_SINGLE_MATCH_SECONDS = env_int(
    "QUEUE_RECENCY_SINGLE_MATCH_SECONDS",
    60,
)
QUEUE_DEDUP_IOU_THRESH = env_float("QUEUE_DEDUP_IOU_THRESH", 0.10)
QUEUE_DEDUP_CENTRE_FRAC = env_float("QUEUE_DEDUP_CENTRE_FRAC", 0.50)
QUEUE_REMAP_IOU_THRESH = env_float("QUEUE_REMAP_IOU_THRESH", 0.10)
QUEUE_REMAP_DIST_THRESH = env_float("QUEUE_REMAP_DIST_THRESH", 180)
QUEUE_REMAP_ABSENT_FRAMES = env_int("QUEUE_REMAP_ABSENT_FRAMES", 45)

QUEUE_CONFIG = {
    "avg_service_time": env_float("AVG_SERVICE_TIME", 3.0),
    "num_counters": env_int("NUM_COUNTERS", 3),
}

CAM_TOKEN = os.getenv("CAM_TOKEN", "detector-secret-token")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "").strip() or secrets.token_hex(32)
TICKETS_OUTPUT_DIR = Path(
    os.getenv("TICKETS_OUTPUT_DIR", str(APP_ROOT / "tickets"))
).expanduser()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = env_int("DB_PORT", 3306)
DB_NAME = os.getenv("DB_NAME", "Crowd_Detection")
DB_USERNAME = os.getenv("DB_USERNAME", "crowd_monitoring_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password123")
DB_SSL_MODE = os.getenv("DB_SSL_MODE", "DISABLED").strip().upper()
DB_SSL_CA = os.getenv("DB_SSL_CA", "").strip()

REDIS_URL = os.getenv("REDIS_URL", "").strip()
CACHE_KEY_PREFIX = os.getenv("CACHE_KEY_PREFIX", "queueflow").strip() or "queueflow"
CACHE_STATE_TTL_SECONDS = env_int("CACHE_STATE_TTL_SECONDS", 30)
CACHE_SNAPSHOT_TTL_SECONDS = env_int("CACHE_SNAPSHOT_TTL_SECONDS", 10)
CACHE_HISTORY_TTL_SECONDS = env_int("CACHE_HISTORY_TTL_SECONDS", 3600)
CACHE_SNAPSHOT_MIN_INTERVAL_SECONDS = env_float(
    "CACHE_SNAPSHOT_MIN_INTERVAL_SECONDS",
    0.5,
)
REDIS_SOCKET_TIMEOUT = env_float("REDIS_SOCKET_TIMEOUT", 0.25)
REDIS_CONNECT_TIMEOUT = env_float("REDIS_CONNECT_TIMEOUT", 0.25)

OBJECT_STORAGE_ENABLED = env_bool("OBJECT_STORAGE_ENABLED", False)
OBJECT_STORAGE_ENDPOINT_URL = os.getenv("OBJECT_STORAGE_ENDPOINT_URL", "").strip()
OBJECT_STORAGE_BUCKET = os.getenv("OBJECT_STORAGE_BUCKET", "").strip()
OBJECT_STORAGE_REGION = os.getenv("OBJECT_STORAGE_REGION", "").strip() or None
OBJECT_STORAGE_ACCESS_KEY_ID = os.getenv("OBJECT_STORAGE_ACCESS_KEY_ID", "").strip() or None
OBJECT_STORAGE_SECRET_ACCESS_KEY = os.getenv("OBJECT_STORAGE_SECRET_ACCESS_KEY", "").strip() or None
OBJECT_STORAGE_PREFIX = os.getenv("OBJECT_STORAGE_PREFIX", "tickets").strip().strip("/")
OBJECT_STORAGE_PUBLIC_BASE_URL = os.getenv("OBJECT_STORAGE_PUBLIC_BASE_URL", "").strip().rstrip("/")
OBJECT_STORAGE_ADDRESSING_STYLE = os.getenv("OBJECT_STORAGE_ADDRESSING_STYLE", "auto").strip()

TEMPLATES_DIR = APP_ROOT / "templates"

MJPEG_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "X-Accel-Buffering": "no",
}


def validate_cloud_config() -> None:
    if not IS_PRODUCTION:
        return
    if not os.getenv("JWT_SECRET_KEY", "").strip():
        raise RuntimeError("JWT_SECRET_KEY must be set when APP_ENV=production")
    if CAM_TOKEN == "detector-secret-token":
        raise RuntimeError("CAM_TOKEN must be changed when APP_ENV=production")
    if "localhost" in PORTAL_BASE_URL or "127.0.0.1" in PORTAL_BASE_URL:
        raise RuntimeError("PORTAL_BASE_URL must be your public cloud URL when APP_ENV=production")
    if STAFF_REGISTRATION_ENABLED and not STAFF_REGISTRATION_CODE:
        raise RuntimeError("STAFF_REGISTRATION_CODE must be set when staff registration is enabled in production")
    if OBJECT_STORAGE_ENABLED and not OBJECT_STORAGE_BUCKET:
        raise RuntimeError("OBJECT_STORAGE_BUCKET must be set when object storage is enabled")
    if SESSION_COOKIE_SAMESITE == "none" and not SESSION_COOKIE_SECURE:
        raise RuntimeError("SESSION_COOKIE_SECURE=1 is required when SESSION_COOKIE_SAMESITE=none")
