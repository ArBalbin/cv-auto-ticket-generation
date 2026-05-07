import threading
import time
from datetime import datetime, timedelta

import numpy as np

import state
from core.config import (
    API_HIGH_CONF,
    API_MIN_BBOX_AREA,
    LOW_CONF_BOOST,
    QUEUE_CONFIG,
    QUEUE_MIN_MOTION_PIXELS,
    QUEUE_MIN_PORTRAIT_ASPECT,
    QUEUE_STATIC_CONF_BYPASS,
    QUEUE_STATIC_STDEV_THRESHOLD,
)
from database.database_handler import update_queue_status
from services.queue_tracker import QueueTracker, QueueZone
from services import ticket_service
from services.ticket_printer import delete_all_tickets


queue_zone = QueueZone(x1=10, y1=10, x2=1910, y2=1070)
queue_tracker = QueueTracker(zone=queue_zone)

_REMAP_IOU_THRESH = 0.15
_REMAP_DIST_THRESH = 120
_MAX_REMAP_ABSENT_FRAMES = 18
_config_lock = threading.Lock()


def _on_new_person(
    queue_number: int,
    wait_time_str: str,
    joined_at_str: str,
    access_token: str,
) -> None:
    """Drop a ticket job into the background worker immediately."""
    position = queue_tracker.get_position(queue_number)
    try:
        ticket_service.enqueue_ticket(
            queue_number=queue_number,
            position=position,
            est_wait_min=0,
        )
        print(f"[QueueService] Q{queue_number:03d} queued for ticket generation")
    except Exception:
        print(f"[QueueService] Ticket queue full - Q{queue_number:03d} skipped")


def _on_noshow(queue_number: int) -> None:
    threading.Thread(
        target=update_queue_status,
        args=(queue_number, "no_show"),
        daemon=True,
        name=f"DBUpdate-Q{queue_number:03d}-noshow",
    ).start()


def wire_callbacks() -> None:
    queue_tracker.on_new_person = _on_new_person
    queue_tracker.on_noshow = _on_noshow
    queue_tracker.MAX_MISSING_FRAMES = 150
    queue_tracker.MIN_CONFIRM_FRAMES = 8
    queue_tracker.MIN_MOTION_PIXELS = QUEUE_MIN_MOTION_PIXELS
    queue_tracker.STATIC_STDEV_THRESHOLD = QUEUE_STATIC_STDEV_THRESHOLD
    queue_tracker.STATIC_CONF_BYPASS_THRESHOLD = QUEUE_STATIC_CONF_BYPASS
    queue_tracker.MIN_PORTRAIT_ASPECT = QUEUE_MIN_PORTRAIT_ASPECT
    queue_tracker.APPEARANCE_TIEBREAK_THRESHOLD = 0.20
    queue_tracker.DONE_BLACKLIST_THRESH = 0.55
    queue_tracker.NOSHOW_WINDOW_SECONDS = 180
    queue_tracker.RECENCY_SINGLE_MATCH_SECONDS = 20


def _bbox_iou(a: tuple, b: tuple) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


def _bbox_centroid_dist(a: tuple, b: tuple) -> float:
    return (
        ((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2) ** 2
        + ((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2) ** 2
    ) ** 0.5


def remap_track_ids(tracked: list, tracker) -> list:
    if not tracked:
        return tracked

    incoming_ids = {p["track_id"] for p in tracked}
    known_ids = set(tracker.active_queue.keys())

    unknown = [p for p in tracked if p["track_id"] not in known_ids]
    if not unknown:
        return tracked

    candidates: list[tuple[int, tuple]] = []
    for tid, person in tracker.active_queue.items():
        if person.status == "done_pending":
            continue
        if tid in incoming_ids:
            continue

        absent_frames = getattr(person, "missing_frames", 0)
        if absent_frames > _MAX_REMAP_ABSENT_FRAMES:
            continue

        bbox = getattr(person, "bbox", None)
        if bbox and len(bbox) == 4:
            candidates.append((tid, tuple(bbox)))

    if not candidates:
        return tracked

    remapped = []
    used_cands: set[int] = set()

    for p in tracked:
        tid = p["track_id"]
        bbox = tuple(p["bbox"])

        if tid in known_ids:
            remapped.append(p)
            continue

        best_tid = None
        best_score = -1.0

        for c_tid, c_bbox in candidates:
            if c_tid in used_cands:
                continue
            iou = _bbox_iou(bbox, c_bbox)
            dist = _bbox_centroid_dist(bbox, c_bbox)

            if iou >= _REMAP_IOU_THRESH:
                score = iou + 1.0
            elif dist <= _REMAP_DIST_THRESH:
                score = 1.0 - dist / _REMAP_DIST_THRESH
            else:
                continue

            if score > best_score:
                best_score = score
                best_tid = c_tid

        if best_tid is not None:
            used_cands.add(best_tid)
            print(
                f"[QueueService] Remap track_id {tid} -> {best_tid} "
                f"(score={best_score:.2f})"
            )
            remapped_p = dict(p)
            remapped_p["track_id"] = best_tid
            remapped_p["bbox"] = list(bbox)
            remapped.append(remapped_p)
        else:
            remapped.append(p)

    return remapped


def inject_appearances(raw_tracked: list, tracker) -> None:
    for p in raw_tracked:
        app_list = p.get("appearance")
        if not app_list:
            continue

        tid = p["track_id"]
        person = tracker.active_queue.get(tid)
        if person is None:
            continue

        try:
            new_sig = np.array(app_list, dtype=np.float32)
            if person.appearance_signature is None:
                person.appearance_signature = new_sig
            else:
                person.appearance_signature = (
                    0.6 * person.appearance_signature + 0.4 * new_sig
                )
            history = person.appearance_history
            history.append(new_sig)
            if len(history) > 5:
                history.pop(0)
        except Exception as exc:
            print(f"[QueueService] appearance injection error tid={tid}: {exc}")


def process_tracked_persons(raw_tracked: list, yolo_frame_idx: int = 0) -> dict:
    queue_state: dict = {}

    if raw_tracked:
        high_conf_ids = set()
        filtered = []
        for p in raw_tracked:
            b = p["bbox"]
            area = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
            if area < API_MIN_BBOX_AREA:
                continue
            conf = p.get("conf", 0.5)
            if conf >= API_HIGH_CONF:
                high_conf_ids.add(p["track_id"])
            filtered.append(p)

        tracked_filtered = []
        conf_boost = max(1, LOW_CONF_BOOST)
        for p in filtered:
            if p["track_id"] in high_conf_ids:
                tracked_filtered.append(p)
            elif yolo_frame_idx % conf_boost == 0:
                tracked_filtered.append(p)

        tracked = [
            {
                "track_id": p["track_id"],
                "bbox": tuple(p["bbox"]),
                "conf": p.get("conf", 0.0),
            }
            for p in tracked_filtered
        ]
        tracked = remap_track_ids(tracked, queue_tracker)
        try:
            queue_state = queue_tracker.process_frame(tracked, frame=None)
        except Exception as exc:
            print(f"[QueueService] process_frame error: {exc}")

        inject_appearances(filtered, queue_tracker)
    else:
        try:
            queue_state = queue_tracker.process_frame([], frame=None)
        except Exception as exc:
            print(f"[QueueService] process_frame empty error: {exc}")

    return queue_state


def done_pending_people() -> list:
    return [
        p.to_dict()
        for p in queue_tracker.active_queue.values()
        if p.status == "done_pending"
    ]


def reset_queue() -> None:
    global queue_tracker
    deleted = delete_all_tickets()
    print(f"[Reset] Deleted {deleted} ticket PDF(s)")
    queue_tracker = QueueTracker(zone=queue_zone)
    wire_callbacks()


def mark_done(queue_number: int) -> dict | None:
    if not queue_tracker.mark_transaction_done(queue_number):
        return None

    threading.Thread(
        target=update_queue_status,
        args=(queue_number, "served"),
        daemon=True,
        name=f"DBUpdate-Q{queue_number:03d}-served",
    ).start()
    return queue_tracker.get_state()


def zone_dict() -> dict:
    z = queue_zone
    return {"x1": z.x1, "y1": z.y1, "x2": z.x2, "y2": z.y2}


def set_zone(x1: int, y1: int, x2: int, y2: int) -> dict:
    queue_zone.set_zone(x1, y1, x2, y2)
    return zone_dict()


def as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def runtime_config() -> dict:
    with _config_lock:
        counters = max(1, as_int(QUEUE_CONFIG.get("num_counters"), 3))
        avg_service_time = max(
            0.1,
            as_float(QUEUE_CONFIG.get("avg_service_time"), 3.0),
        )
    return {
        "active_counters": counters,
        "avg_service_time": avg_service_time,
    }


def set_active_counters(counters: int) -> dict:
    counters = max(1, as_int(counters, QUEUE_CONFIG["num_counters"]))
    with _config_lock:
        QUEUE_CONFIG["num_counters"] = counters
        avg_service_time = max(
            0.1,
            as_float(QUEUE_CONFIG.get("avg_service_time"), 3.0),
        )

    state.set_active_counters(counters)
    return {
        "active_counters": counters,
        "avg_service_time": avg_service_time,
    }


def format_minutes(minutes: float) -> str:
    minutes = max(0.0, float(minutes))
    if minutes < 1:
        return "less than 1 min"
    if minutes < 60:
        whole = int(round(minutes))
        return f"{whole} min" if whole == 1 else f"{whole} mins"
    hours = int(minutes // 60)
    mins = int(round(minutes % 60))
    if mins == 0:
        return f"{hours} hr" if hours == 1 else f"{hours} hrs"
    return f"{hours} hr {mins} min" if hours == 1 else f"{hours} hrs {mins} min"


def eta_iso(wait_minutes: float) -> str:
    return (datetime.now() + timedelta(minutes=max(0.0, wait_minutes))).isoformat()


def estimate_wait_for_position(
    position: int,
    avg_service_time: float,
    counters: int,
) -> float:
    if position <= 0:
        return 0.0
    counters = max(1, counters)
    avg_service_time = max(0.1, avg_service_time)
    batches_before = max(0, (position - 1) // counters)
    return round(batches_before * avg_service_time, 1)


def prediction_for_position(
    position: int,
    avg_service_time: float,
    counters: int,
) -> dict:
    wait_min = estimate_wait_for_position(position, avg_service_time, counters)
    service_min = round(max(0.1, avg_service_time), 1)
    return {
        "position": position,
        "estimated_wait_time": wait_min,
        "estimated_wait_time_minutes": wait_min,
        "estimated_wait_time_label": format_minutes(wait_min),
        "estimated_service_start_at": eta_iso(wait_min),
        "estimated_service_finish_at": eta_iso(wait_min + service_min),
        "estimated_service_time_min": service_min,
    }


def build_queue_prediction() -> dict:
    with state.state_lock:
        metrics = {
            "timestamp": state.latest_state["timestamp"],
            "active_counters": state.latest_state["active_counters"],
            "estimated_wait_time": state.latest_state["estimated_wait_time"],
            "arrival_rate": state.latest_state["arrival_rate"],
            "system_utilization": state.latest_state["system_utilization"],
            "predicted_wait_5min": state.latest_state["predicted_wait_5min"],
            "predicted_wait_15min": state.latest_state["predicted_wait_15min"],
            "predicted_wait_30min": state.latest_state["predicted_wait_30min"],
        }

    queue_state = queue_tracker.get_state()
    active_queue = queue_state.get("active_queue", [])
    queue_count = len(active_queue)
    config = runtime_config()
    counters = config["active_counters"]
    avg_service_time = config["avg_service_time"]
    data_age_seconds = max(0.0, time.time() - as_float(metrics["timestamp"], time.time()))
    current_wait = as_float(metrics["estimated_wait_time"], 0.0)
    fallback_new_wait = estimate_wait_for_position(
        queue_count + 1,
        avg_service_time,
        counters,
    )
    new_arrival_wait = round(
        current_wait if current_wait > 0 else fallback_new_wait,
        1,
    )

    people = []
    for person in active_queue:
        position = as_int(person.get("position_in_line"), 0)
        people.append({
            "queue_number": person.get("queue_number"),
            "queue_label": person.get("queue_label"),
            "status": person.get("status"),
            **prediction_for_position(position, avg_service_time, counters),
        })

    return {
        "queue_length": queue_count,
        "active_counters": counters,
        "avg_service_time_min": round(avg_service_time, 1),
        "arrival_rate_per_min": as_float(metrics["arrival_rate"], 0.0),
        "system_utilization": as_float(metrics["system_utilization"], 0.0),
        "data_age_seconds": round(data_age_seconds, 1),
        "data_status": "stale" if data_age_seconds > 15 else "live",
        "new_arrival": {
            "position": queue_count + 1,
            "estimated_wait_time": new_arrival_wait,
            "estimated_wait_time_minutes": new_arrival_wait,
            "estimated_wait_time_label": format_minutes(new_arrival_wait),
            "estimated_service_start_at": eta_iso(new_arrival_wait),
        },
        "forecast": {
            "now": {
                "horizon_minutes": 0,
                "estimated_wait_time": round(new_arrival_wait, 1),
                "estimated_wait_time_minutes": round(new_arrival_wait, 1),
                "estimated_wait_time_label": format_minutes(new_arrival_wait),
            },
            "in_5min": {
                "horizon_minutes": 5,
                "estimated_wait_time": as_float(
                    metrics["predicted_wait_5min"],
                    new_arrival_wait,
                ),
                "estimated_wait_time_minutes": as_float(
                    metrics["predicted_wait_5min"],
                    new_arrival_wait,
                ),
                "estimated_wait_time_label": format_minutes(
                    as_float(metrics["predicted_wait_5min"], new_arrival_wait)
                ),
            },
            "in_15min": {
                "horizon_minutes": 15,
                "estimated_wait_time": as_float(
                    metrics["predicted_wait_15min"],
                    new_arrival_wait,
                ),
                "estimated_wait_time_minutes": as_float(
                    metrics["predicted_wait_15min"],
                    new_arrival_wait,
                ),
                "estimated_wait_time_label": format_minutes(
                    as_float(metrics["predicted_wait_15min"], new_arrival_wait)
                ),
            },
            "in_30min": {
                "horizon_minutes": 30,
                "estimated_wait_time": as_float(
                    metrics["predicted_wait_30min"],
                    new_arrival_wait,
                ),
                "estimated_wait_time_minutes": as_float(
                    metrics["predicted_wait_30min"],
                    new_arrival_wait,
                ),
                "estimated_wait_time_label": format_minutes(
                    as_float(metrics["predicted_wait_30min"], new_arrival_wait)
                ),
            },
        },
        "active_queue": people,
    }


def _clean_queue_record(record: dict) -> dict:
    hidden = {"access_token", "jwt_token", "short_code", "pdf_path"}
    return {key: value for key, value in record.items() if key not in hidden}


def _average_wait(wait_values: list[int]) -> float:
    if not wait_values:
        return 0.0
    return round(sum(wait_values) / len(wait_values), 1)


def _wait_band_counts(active_queue: list[dict]) -> list[dict]:
    bands = [
        ("0-2 min", 0, 120),
        ("2-5 min", 120, 300),
        ("5-10 min", 300, 600),
        ("10+ min", 600, None),
    ]
    counts = []
    for label, lower, upper in bands:
        count = 0
        for person in active_queue:
            seconds = as_int(person.get("wait_time_seconds"), 0)
            if upper is None:
                in_band = seconds >= lower
            else:
                in_band = lower <= seconds < upper
            if in_band:
                count += 1
        counts.append({"label": label, "count": count})
    return counts


def _analytics_recommendation(
    data_status: str,
    utilization: float,
    queue_length: int,
    missing_count: int,
    noshow_alert_count: int,
) -> str:
    if data_status == "stale":
        return "Detector updates are stale. Check the camera process before trusting the trend."
    if noshow_alert_count:
        return "A no-show countdown is active. Watch the first position before marking the next person."
    if missing_count:
        return "Some queued people are missing from the zone. Confirm the camera view and queue zone."
    if utilization >= 0.9 and queue_length:
        return "The queue is close to full counter capacity. Opening another counter may reduce wait time."
    if queue_length == 0:
        return "Queue is clear. Keep the detector running for fresh arrivals."
    return "Queue is moving normally."


def build_queue_analytics() -> dict:
    prediction = build_queue_prediction()
    queue_state = queue_tracker.get_state()
    active_queue = queue_state.get("active_queue", [])
    completed_all = list(queue_tracker.completed_queue)
    recent_completed = completed_all[-10:]

    waiting_count = sum(1 for person in active_queue if person.get("status") == "waiting")
    missing_count = sum(1 for person in active_queue if person.get("status") == "missing")
    queue_length = len(active_queue)

    active_waits = [
        as_int(person.get("wait_time_seconds"), 0)
        for person in active_queue
    ]
    completed_waits = [
        as_int(record.get("wait_time_seconds"), 0)
        for record in completed_all
    ]

    total_completed = len(completed_all)
    total_no_show = sum(
        1 for record in completed_all
        if record.get("bump_reason") == "no_show"
    )
    total_served = sum(
        1 for record in completed_all
        if record.get("bump_reason") == "served"
    )
    total_assigned = max(
        0,
        as_int(queue_state.get("next_number"), 1) - 1,
    )

    data_status = prediction.get("data_status", "live")
    utilization = as_float(prediction.get("system_utilization"), 0.0)
    noshow_alerts = queue_state.get("noshow_alerts", [])

    active_by_number = {
        person.get("queue_number"): _clean_queue_record(person)
        for person in active_queue
    }
    active_people = []
    for person in prediction.get("active_queue", []):
        queue_number = person.get("queue_number")
        active_people.append({
            **active_by_number.get(queue_number, {}),
            **person,
        })

    forecast = prediction.get("forecast", {})
    forecast_wait = [
        {
            "label": "Now",
            "horizon_minutes": 0,
            "wait_minutes": as_float(
                forecast.get("now", {}).get("estimated_wait_time_minutes"),
                0.0,
            ),
        },
        {
            "label": "5 min",
            "horizon_minutes": 5,
            "wait_minutes": as_float(
                forecast.get("in_5min", {}).get("estimated_wait_time_minutes"),
                0.0,
            ),
        },
        {
            "label": "15 min",
            "horizon_minutes": 15,
            "wait_minutes": as_float(
                forecast.get("in_15min", {}).get("estimated_wait_time_minutes"),
                0.0,
            ),
        },
        {
            "label": "30 min",
            "horizon_minutes": 30,
            "wait_minutes": as_float(
                forecast.get("in_30min", {}).get("estimated_wait_time_minutes"),
                0.0,
            ),
        },
    ]

    active_avg_wait = _average_wait(active_waits)
    active_max_wait = max(active_waits) if active_waits else 0
    completed_avg_wait = _average_wait(completed_waits)

    return {
        "generated_at": datetime.now().isoformat(),
        "overview": {
            "queue_length": queue_length,
            "waiting": waiting_count,
            "missing": missing_count,
            "active_counters": prediction.get("active_counters", 0),
            "avg_service_time_min": prediction.get("avg_service_time_min", 0),
            "next_number": queue_state.get("next_number"),
            "total_assigned": total_assigned,
            "total_completed": total_completed,
            "total_served": total_served,
            "total_no_show": total_no_show,
            "completion_rate_percent": round(
                (total_completed / total_assigned) * 100,
                1,
            ) if total_assigned else 0.0,
            "no_show_rate_percent": round(
                (total_no_show / total_completed) * 100,
                1,
            ) if total_completed else 0.0,
            "data_status": data_status,
            "data_age_seconds": prediction.get("data_age_seconds", 0),
        },
        "wait_times": {
            "active_average_wait_seconds": active_avg_wait,
            "active_average_wait_label": format_minutes(active_avg_wait / 60),
            "active_max_wait_seconds": active_max_wait,
            "active_max_wait_label": format_minutes(active_max_wait / 60),
            "completed_average_wait_seconds": completed_avg_wait,
            "completed_average_wait_label": format_minutes(completed_avg_wait / 60),
        },
        "throughput": {
            "recent_completed_count": len(recent_completed),
            "recent_served_count": sum(
                1 for record in recent_completed
                if record.get("bump_reason") == "served"
            ),
            "recent_no_show_count": sum(
                1 for record in recent_completed
                if record.get("bump_reason") == "no_show"
            ),
        },
        "live_crowd": state.crowd_prediction_fields(),
        "new_arrival": prediction.get("new_arrival", {}),
        "forecast": forecast,
        "charts": {
            "forecast_wait": forecast_wait,
            "status_breakdown": [
                {"label": "Waiting", "count": waiting_count},
                {"label": "Missing", "count": missing_count},
                {"label": "Served", "count": total_served},
                {"label": "No-show", "count": total_no_show},
            ],
            "wait_bands": _wait_band_counts(active_queue),
        },
        "active_queue": active_people,
        "recent_completed": [
            _clean_queue_record(record)
            for record in reversed(recent_completed)
        ],
        "noshow_alerts": noshow_alerts,
        "appearance_rejections": queue_state.get("appearance_rejections", []),
        "zone": zone_dict(),
        "recommendation": _analytics_recommendation(
            data_status,
            utilization,
            queue_length,
            missing_count,
            len(noshow_alerts),
        ),
    }


wire_callbacks()
