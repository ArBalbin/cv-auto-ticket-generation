import json
import threading
import time
from typing import Any

from core.config import (
    CACHE_KEY_PREFIX,
    REDIS_CONNECT_TIMEOUT,
    REDIS_SOCKET_TIMEOUT,
    REDIS_URL,
)

try:
    import redis
    from redis.exceptions import RedisError
except ModuleNotFoundError:
    redis = None

    class RedisError(Exception):
        pass


_client = None
_client_lock = threading.Lock()
_warned_unavailable = False
_retry_after = 0.0


def cache_key(name: str) -> str:
    return f"{CACHE_KEY_PREFIX}:{name}"


def is_configured() -> bool:
    return bool(REDIS_URL)


def client():
    global _client, _retry_after, _warned_unavailable

    if not REDIS_URL:
        return None
    if redis is None:
        if not _warned_unavailable:
            print("[Cache] Redis package is not installed; cache disabled")
            _warned_unavailable = True
        return None

    if _client is not None:
        return _client
    if time.time() < _retry_after:
        return None

    with _client_lock:
        if _client is not None:
            return _client
        if time.time() < _retry_after:
            return None
        try:
            _client = redis.from_url(
                REDIS_URL,
                socket_timeout=REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
                retry_on_timeout=False,
                decode_responses=False,
                health_check_interval=30,
            )
            _client.ping()
            print("[Cache] Redis connected")
        except RedisError as exc:
            _client = None
            _retry_after = time.time() + 5.0
            if not _warned_unavailable:
                print(f"[Cache] Redis unavailable; using process memory only: {exc}")
                _warned_unavailable = True
        return _client


def is_available() -> bool:
    c = client()
    if c is None:
        return False
    try:
        c.ping()
        return True
    except RedisError:
        return False


def set_bytes(name: str, value: bytes, ttl_seconds: int) -> None:
    c = client()
    if c is None:
        return
    try:
        c.setex(cache_key(name), max(1, ttl_seconds), value)
    except RedisError:
        pass


def get_bytes(name: str) -> bytes | None:
    c = client()
    if c is None:
        return None
    try:
        value = c.get(cache_key(name))
    except RedisError:
        return None
    return value if isinstance(value, bytes) else None


def set_json(name: str, value: Any, ttl_seconds: int) -> None:
    payload = json.dumps(value, separators=(",", ":"), default=str).encode("utf-8")
    set_bytes(name, payload, ttl_seconds)


def get_json(name: str, fallback: Any = None) -> Any:
    payload = get_bytes(name)
    if payload is None:
        return fallback
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return fallback


def delete(name: str) -> None:
    c = client()
    if c is None:
        return
    try:
        c.delete(cache_key(name))
    except RedisError:
        pass
