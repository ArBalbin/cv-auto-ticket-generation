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


def _detect_lan_ip() -> str | None:
    """
    The LAN address a phone on the same network can actually reach.

    This matters because PORTAL_BASE_URL is baked into every printed ticket's
    QR code. Left at "localhost", the QR resolves to the STUDENT'S OWN phone,
    not the server, so every scanned ticket fails — and the failure only shows
    up when someone scans a printed ticket, which is exactly when it is too
    late to notice.

    Opening a UDP socket toward a public address sends no packet; it just asks
    the OS routing table which interface it would use. That correctly picks the
    real Wi-Fi/Ethernet address and skips virtual adapters (VirtualBox's
    192.168.56.x, WSL, VPN tunnels) that a simple hostname lookup returns and
    no phone can reach.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.settimeout(0.5)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        return ip if ip and not ip.startswith("127.") else None
    except Exception:
        return None
    finally:
        sock.close()


_portal_env = os.getenv("PORTAL_BASE_URL", "").strip().rstrip("/")
_portal_autodetected = False

if _portal_env and "localhost" not in _portal_env and "127.0.0.1" not in _portal_env:
    # An explicit, non-loopback value always wins — a deployed URL, a domain,
    # or an IP the operator pinned on purpose.
    PORTAL_BASE_URL = _portal_env
else:
    # Unset or loopback. In development, fall back to the machine's LAN
    # address so tickets printed on a laptop are scannable from a phone on
    # the same Wi-Fi without anyone remembering to edit .env after every
    # network change. Production keeps loopback and fails the guard below,
    # which is correct: a cloud deployment must state its real public URL.
    _lan_ip = None if IS_PRODUCTION else _detect_lan_ip()
    if _lan_ip:
        PORTAL_BASE_URL = f"http://{_lan_ip}:{API_PORT}"
        _portal_autodetected = True
    else:
        PORTAL_BASE_URL = _portal_env or "http://localhost:5000"

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

# Student self-registration (mobile app) — Sign-in-with-Google against the
# student's NCF Gbox account. No student password is ever stored; Google is
# the credential authority. GOOGLE_OAUTH_CLIENT_ID is the OAuth client ID
# registered for the Flutter app in Google Cloud Console — required to
# verify the "aud" claim on every Google ID token the app sends us.
GBOX_ALLOWED_DOMAIN = os.getenv("GBOX_ALLOWED_DOMAIN", "gbox.ncf.edu.ph").strip()
GOOGLE_OAUTH_CLIENT_ID = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "").strip()
STUDENT_SESSION_TTL_SECONDS = env_int("STUDENT_SESSION_TTL_SECONDS", 60 * 60 * 24 * 30)

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
QUEUE_DEDUP_IOU_THRESH = env_float("QUEUE_DEDUP_IOU_THRESH", 0.10)
QUEUE_DEDUP_CENTRE_FRAC = env_float("QUEUE_DEDUP_CENTRE_FRAC", 0.50)
QUEUE_REMAP_IOU_THRESH = env_float("QUEUE_REMAP_IOU_THRESH", 0.10)
QUEUE_REMAP_DIST_THRESH = env_float("QUEUE_REMAP_DIST_THRESH", 180)
QUEUE_REMAP_ABSENT_FRAMES = env_int("QUEUE_REMAP_ABSENT_FRAMES", 45)

QUEUE_CONFIG = {
    "avg_service_time": env_float("AVG_SERVICE_TIME", 3.0),
    "num_counters": env_int("NUM_COUNTERS", 3),
}

# Face-recognition identity validation (students only) — see app/services/face_service.py
# Thresholds calibrated 2026-08 via ML/calibrate_face_recognition.py against 16
# real people / 43 verification trials (buffalo_s): 100% correct identification,
# 0 wrong-identity accepts at this operating point once one contaminated (AI
# face-filter) photo was excluded from the calibration set. Re-run that script
# and update these if the deployment camera/lighting differs meaningfully from
# the calibration photos.
FACE_MODEL_PACK = os.getenv("FACE_MODEL_PACK", "buffalo_s").strip() or "buffalo_s"
FACE_MATCH_THRESHOLD = env_float("FACE_MATCH_THRESHOLD", 0.30)
# 0.15, raised from 0.10 after ML/evaluate_face_accuracy.py found the old
# value sat inside the impostor range. In the open-set test (a non-enrolled
# person at the camera) one stranger was accepted as an enrolled student with
# margin 0.110 — clearing 0.10 by 0.01. Measured separation on 17 identities:
# strangers reach at most 0.110, genuine students sit at 0.222 and above
# (excluding one probe already refused at any setting). 0.15 lands between
# them with headroom on both sides and costs zero genuine links: still 44/45
# linked automatically, now with 0 false identifications instead of 1.
FACE_MARGIN_THRESHOLD = env_float("FACE_MARGIN_THRESHOLD", 0.15)
FACE_MIN_DETECT_CONF = env_float("FACE_MIN_DETECT_CONF", 0.60)

# Per SOP #2 (reconfirmed by the panel after the pre-oral revision meeting):
# a registered student recognized at the queue-zone camera gets a queue
# number minted directly by the system — no kiosk button press needed. This
# is an ADDITIONAL path for registered students only; it never touches the
# kiosk's own walk-in flow. Numbers start well above any realistic kiosk
# range so a face-only number can never collide with a printed kiosk ticket.
FACE_ONLY_QUEUE_NUMBER_START = env_int("FACE_ONLY_QUEUE_NUMBER_START", 5000)

# How long a person can sit in the zone, presence-confirmed but not yet linked
# to a printed queue number, before staff are alerted to resolve it manually.
PENDING_LINK_TIMEOUT_SECONDS = env_int("PENDING_LINK_TIMEOUT_SECONDS", 45)

# How long a student's "I want a ticket" intent stays valid after they tap
# it in the app. Recognition only issues a number to a student who has armed
# themselves, so that merely walking past the camera never produces a ticket
# nobody asked for. The intent lapses on its own so a tap made and forgotten
# cannot surprise them with a number on a later visit.
JOIN_INTENT_TIMEOUT_MINUTES = env_int("JOIN_INTENT_TIMEOUT_MINUTES", 60)

CAM_TOKEN = os.getenv("CAM_TOKEN", "detector-secret-token")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "").strip() or secrets.token_hex(32)
_tickets_output_raw = os.getenv("TICKETS_OUTPUT_DIR", str(APP_ROOT / "tickets")).strip()
TICKETS_OUTPUT_DIR = Path(_tickets_output_raw).expanduser()
if not TICKETS_OUTPUT_DIR.is_absolute():
    TICKETS_OUTPUT_DIR = PROJECT_ROOT / TICKETS_OUTPUT_DIR

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
    if not GOOGLE_OAUTH_CLIENT_ID:
        raise RuntimeError("GOOGLE_OAUTH_CLIENT_ID must be set when APP_ENV=production")
