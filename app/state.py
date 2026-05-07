import threading
import time
from collections import deque

from core.config import (
    CACHE_HISTORY_TTL_SECONDS,
    CACHE_SNAPSHOT_MIN_INTERVAL_SECONDS,
    CACHE_SNAPSHOT_TTL_SECONDS,
    CACHE_STATE_TTL_SECONDS,
    HISTORY_LEN,
    QUEUE_CONFIG,
)
from services import cache_service


# Written by POST /yolo/push-frame; read by GET /api/* endpoints.
state_lock = threading.Lock()
latest_state = {
    "count": 0,
    "avg_density": 0.0,
    "max_density": 0.0,
    "timestamp": time.time(),
    "queue_length": 0,
    "active_counters": QUEUE_CONFIG["num_counters"],
    "estimated_wait_time": 0.0,
    "arrival_rate": 0.0,
    "system_utilization": 0.0,
    "predicted_wait_5min": 0.0,
    "predicted_wait_15min": 0.0,
    "predicted_wait_30min": 0.0,
    "queue_state": {},
}

latest_snapshot: bytes | None = None
latest_snapshot_seq = 0
latest_snapshot_cache_write_at = 0.0
_snapshot_cache_lock = threading.Lock()
_snapshot_cache_pending: tuple[bytes, int] | None = None
_snapshot_cache_writer_running = False
snapshot_cond = threading.Condition()
history: deque = deque(maxlen=HISTORY_LEN)

_STATE_CACHE_KEY = "state:latest"
_HISTORY_CACHE_KEY = "history:recent"
_SNAPSHOT_CACHE_KEY = "snapshot:latest"
_SNAPSHOT_SEQ_CACHE_KEY = "snapshot:seq"
_cache_write_lock = threading.Lock()
_cache_write_pending: dict[str, tuple[object, int]] = {}
_cache_write_worker_running = False


def _schedule_json_cache_write(name: str, value: object, ttl_seconds: int) -> None:
    global _cache_write_worker_running

    with _cache_write_lock:
        _cache_write_pending[name] = (value, ttl_seconds)
        if _cache_write_worker_running:
            return
        _cache_write_worker_running = True

    threading.Thread(
        target=_json_cache_writer,
        name="StateCacheWriter",
        daemon=True,
    ).start()


def _json_cache_writer() -> None:
    global _cache_write_worker_running

    while True:
        with _cache_write_lock:
            pending = dict(_cache_write_pending)
            _cache_write_pending.clear()

        if not pending:
            with _cache_write_lock:
                if not _cache_write_pending:
                    _cache_write_worker_running = False
                    return
                continue

        for name, (value, ttl_seconds) in pending.items():
            cache_service.set_json(name, value, ttl_seconds)


def update_from_detector_payload(body: dict) -> None:
    global latest_state

    with state_lock:
        latest_state.update({
            "count": body.get("count", 0),
            "avg_density": body.get("avg_density", 0.0),
            "max_density": body.get("max_density", 0.0),
            "queue_state": body.get("queue_state", {}),
            "timestamp": body.get("timestamp", time.time()),
            "queue_length": body.get("queue_length", 0),
            "active_counters": QUEUE_CONFIG["num_counters"],
            "estimated_wait_time": body.get("estimated_wait_time", 0.0),
            "arrival_rate": body.get("arrival_rate", 0.0),
            "system_utilization": body.get("system_utilization", 0.0),
            "predicted_wait_5min": body.get("predicted_wait_5min", 0.0),
            "predicted_wait_15min": body.get("predicted_wait_15min", 0.0),
            "predicted_wait_30min": body.get("predicted_wait_30min", 0.0),
        })
        history.append({
            "count": body.get("count", 0),
            "timestamp": body.get("timestamp", time.time()),
        })
        state_snapshot = dict(latest_state)
        history_snapshot = list(history)

    _schedule_json_cache_write(
        _STATE_CACHE_KEY,
        state_snapshot,
        CACHE_STATE_TTL_SECONDS,
    )
    _schedule_json_cache_write(
        _HISTORY_CACHE_KEY,
        history_snapshot,
        CACHE_HISTORY_TTL_SECONDS,
    )


def set_active_counters(counters: int) -> None:
    with state_lock:
        latest_state["active_counters"] = counters
        state_snapshot = dict(latest_state)

    _schedule_json_cache_write(
        _STATE_CACHE_KEY,
        state_snapshot,
        CACHE_STATE_TTL_SECONDS,
    )


def set_snapshot(snapshot: bytes) -> None:
    global latest_snapshot, latest_snapshot_cache_write_at, latest_snapshot_seq
    with snapshot_cond:
        latest_snapshot = snapshot
        latest_snapshot_seq += 1
        seq = latest_snapshot_seq
        snapshot_cond.notify_all()
    now = time.time()
    if now - latest_snapshot_cache_write_at < CACHE_SNAPSHOT_MIN_INTERVAL_SECONDS:
        return
    latest_snapshot_cache_write_at = now
    _schedule_snapshot_cache_write(snapshot, seq)


def _schedule_snapshot_cache_write(snapshot: bytes, seq: int) -> None:
    global _snapshot_cache_pending, _snapshot_cache_writer_running

    with _snapshot_cache_lock:
        _snapshot_cache_pending = (snapshot, seq)
        if _snapshot_cache_writer_running:
            return
        _snapshot_cache_writer_running = True

    threading.Thread(
        target=_snapshot_cache_writer,
        name="SnapshotCacheWriter",
        daemon=True,
    ).start()


def _snapshot_cache_writer() -> None:
    global _snapshot_cache_pending, _snapshot_cache_writer_running

    while True:
        with _snapshot_cache_lock:
            pending = _snapshot_cache_pending
            _snapshot_cache_pending = None

        if pending is None:
            with _snapshot_cache_lock:
                if _snapshot_cache_pending is None:
                    _snapshot_cache_writer_running = False
                    return
                continue

        snapshot, seq = pending
        _write_snapshot_cache(snapshot, seq)


def _write_snapshot_cache(snapshot: bytes, seq: int) -> None:
    cache_service.set_bytes(
        _SNAPSHOT_CACHE_KEY,
        snapshot,
        CACHE_SNAPSHOT_TTL_SECONDS,
    )
    cache_service.set_json(
        _SNAPSHOT_SEQ_CACHE_KEY,
        seq,
        CACHE_SNAPSHOT_TTL_SECONDS,
    )


def get_snapshot() -> bytes | None:
    if latest_snapshot is not None:
        return latest_snapshot
    cached = cache_service.get_bytes(_SNAPSHOT_CACHE_KEY)
    if cached is not None:
        return cached
    return None


def get_snapshot_seq() -> int:
    if latest_snapshot_seq:
        return latest_snapshot_seq
    cached = cache_service.get_json(_SNAPSHOT_SEQ_CACHE_KEY)
    if isinstance(cached, int):
        return cached
    return 0


def has_snapshot() -> bool:
    return get_snapshot() is not None


def wait_for_snapshot(last_seq: int, timeout: float = 2.0) -> tuple[bytes | None, int]:
    deadline = time.time() + timeout

    while True:
        with snapshot_cond:
            current_seq = latest_snapshot_seq
            if latest_snapshot is not None and current_seq != last_seq:
                return latest_snapshot, current_seq

            remaining = max(0.0, min(0.25, deadline - time.time()))
            if remaining > 0:
                snapshot_cond.wait(timeout=remaining)

        cached_seq = get_snapshot_seq()
        cached_snapshot = get_snapshot()
        if cached_snapshot is not None and cached_seq != last_seq:
            return cached_snapshot, cached_seq

        if time.time() >= deadline:
            return None, last_seq


def crowd_stats() -> dict:
    data = _latest_state_data()
    return {
        "count": data["count"],
        "avg_density": data["avg_density"],
        "max_density": data["max_density"],
        "timestamp": data["timestamp"],
    }


def full_data() -> dict:
    return _latest_state_data()


def history_data() -> dict:
    with state_lock:
        history_snapshot = list(history)
    if history_snapshot:
        return {"history": history_snapshot}
    cached = cache_service.get_json(_HISTORY_CACHE_KEY)
    if isinstance(cached, list):
        return {"history": cached}
    return {"history": history_snapshot}


def _latest_state_data() -> dict:
    with state_lock:
        data = dict(latest_state)
    if time.time() - float(data.get("timestamp") or 0) <= CACHE_STATE_TTL_SECONDS:
        return data
    cached = cache_service.get_json(_STATE_CACHE_KEY)
    if isinstance(cached, dict):
        data.update(cached)
    return data


def crowd_prediction_fields() -> dict:
    data = _latest_state_data()
    return {
        "count": data["count"],
        "avg_density": data["avg_density"],
        "max_density": data["max_density"],
        "timestamp": data["timestamp"],
        "queue_length": data["queue_length"],
        "active_counters": data["active_counters"],
        "estimated_wait_time": data["estimated_wait_time"],
        "arrival_rate": data["arrival_rate"],
        "system_utilization": data["system_utilization"],
        "predicted_wait_5min": data["predicted_wait_5min"],
        "predicted_wait_15min": data["predicted_wait_15min"],
        "predicted_wait_30min": data["predicted_wait_30min"],
    }
