import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np

import state
from core.config import (
    API_HIGH_CONF,
    API_MIN_BBOX_AREA,
    FACE_MARGIN_THRESHOLD,
    FACE_MATCH_THRESHOLD,
    LOW_CONF_BOOST,
    PENDING_LINK_TIMEOUT_SECONDS,
    QUEUE_CONFIG,
    QUEUE_DEDUP_CENTRE_FRAC,
    QUEUE_DEDUP_IOU_THRESH,
    QUEUE_MAX_MISSING_FRAMES,
    QUEUE_MIN_CONFIRM_FRAMES,
    QUEUE_MIN_MOTION_PIXELS,
    QUEUE_MIN_PORTRAIT_ASPECT,
    QUEUE_REMAP_ABSENT_FRAMES,
    QUEUE_REMAP_DIST_THRESH,
    QUEUE_REMAP_IOU_THRESH,
    QUEUE_STATIC_CONF_BYPASS,
    QUEUE_STATIC_STDEV_THRESHOLD,
    SCHOOL_UTC_OFFSET_HOURS,
    WAIT_ESTIMATOR,
)
from database.database_handler import (
    fetch_waiting_queue_records,
    get_student_by_id,
    measure_department_service_time,
    get_all_active_embeddings,
    measure_avg_service_time,
    record_counter_config_change,
    record_face_match_event,
    record_recognition_metric,
    record_queue_reset,
    update_queue_status,
)
from services.queue_tracker import QueueTracker, QueueZone
from services import face_service, queue_engine, ticket_service, wait_model
from services import ets_prediction
from services.embedding_index import EmbeddingIndex
from services.ticket_printer import delete_all_tickets, invalidate_ticket_cache

queue_zone = QueueZone(x1=10, y1=10, x2=1910, y2=1070)
queue_tracker = QueueTracker(zone=queue_zone)

# One shared index of enrolled faces for every 1:N match (queue camera, face
# login, duplicate-enrolment check). The loader looks the database function up
# at call time so tests can substitute it.
embedding_index = EmbeddingIndex(loader=lambda: get_all_active_embeddings())

_REMAP_IOU_THRESH = QUEUE_REMAP_IOU_THRESH
_REMAP_DIST_THRESH = QUEUE_REMAP_DIST_THRESH
_MAX_REMAP_ABSENT_FRAMES = QUEUE_REMAP_ABSENT_FRAMES
# Longest gap, in frames, over which a track is remapped without a face to
# confirm it is the same person (about half a second at 15 fps).
_REMAP_FACELESS_MAX_FRAMES = 8
_config_lock = threading.Lock()
_department_average_cache: dict[str, tuple[float, float]] = {}

USE_QUEUE_ENGINE = WAIT_ESTIMATOR != "regression"
SCHOOL_TZ = timezone(timedelta(hours=SCHOOL_UTC_OFFSET_HOURS))
# Kept here as well as on the tracker so a queue reset does not reopen
# counters that staff closed.
_counters_closed_until: datetime | None = None

def _on_number_linked(
    queue_number: int,
    wait_time_str: str,
    joined_at_str: str,
    access_token: str,
    student_id: int | None = None,
    linked_via: str = "manual",
    is_walkin: bool = True,
) -> None:
    """A printed kiosk number was just linked to a present person (student
    face+OCR or face-only match, or staff manual entry) — drop a ticket job
    into the background worker immediately. Ticket generation itself is
    unchanged; only the trigger event moved from "CV auto-detected someone"
    to "a real kiosk number was linked.\""""
    position = queue_tracker.get_position(queue_number)
    counters = runtime_config()["active_counters"]
    student = get_student_by_id(student_id) if student_id is not None else None
    department = (student or {}).get("department")
    initial_prediction = prediction_for_position(
        position, runtime_config()["avg_service_time"], counters, department
    )
    try:
        ticket_service.enqueue_ticket(
            queue_number=queue_number,
            position=position,
            est_wait_min=initial_prediction["estimated_wait_time_minutes"],
            student_id=student_id,
            linked_via=linked_via,
            is_walkin=is_walkin,
            # Logged with the ticket, so the estimate the student saw can
            # later be checked against the wait they actually had.
            people_ahead=max(0, position - 1),
            active_counters=counters,
            estimator_version=f"{initial_prediction['prediction_method']}:{initial_prediction.get('model_version') or 'fallback'}",
            predicted_service_time_seconds=initial_prediction["estimated_service_time_seconds"],
        )
        print(f"[QueueService] Q{queue_number:03d} ({linked_via}) queued for ticket generation")
    except Exception:
        print(f"[QueueService] Ticket queue full - Q{queue_number:03d} skipped")

def _service_time_refresh_loop() -> None:
    """Background thread: re-measure avg_service_time from DB every 5 minutes."""
    while True:
        time.sleep(300)
        try:
            num_counters = max(1, int(QUEUE_CONFIG.get("num_counters", 3)))
            measured = measure_avg_service_time(num_counters)
            if measured is not None:
                with _config_lock:
                    old = QUEUE_CONFIG.get("avg_service_time", 3.0)
                    # Blend: 70% measured, 30% previous to avoid sudden jumps
                    blended = round(0.7 * measured + 0.3 * old, 2)
                    QUEUE_CONFIG["avg_service_time"] = blended
                print(f"[QueueService] avg_service_time updated: "
                      f"{old:.2f} → {blended:.2f} min (measured={measured:.2f})")
        except Exception as exc:
            print(f"[QueueService] service time refresh error: {exc}")

def wire_callbacks() -> None:
    queue_tracker.on_number_linked = _on_number_linked
    queue_tracker.MAX_MISSING_FRAMES = QUEUE_MAX_MISSING_FRAMES
    queue_tracker.MIN_CONFIRM_FRAMES = QUEUE_MIN_CONFIRM_FRAMES
    queue_tracker.MIN_MOTION_PIXELS = QUEUE_MIN_MOTION_PIXELS
    queue_tracker.STATIC_STDEV_THRESHOLD = QUEUE_STATIC_STDEV_THRESHOLD
    queue_tracker.STATIC_CONF_BYPASS_THRESHOLD = QUEUE_STATIC_CONF_BYPASS
    queue_tracker.MIN_PORTRAIT_ASPECT = QUEUE_MIN_PORTRAIT_ASPECT
    queue_tracker.FACE_MATCH_THRESHOLD = FACE_MATCH_THRESHOLD
    queue_tracker.FACE_MARGIN_THRESHOLD = FACE_MARGIN_THRESHOLD
    queue_tracker.PENDING_LINK_TIMEOUT_SECONDS = PENDING_LINK_TIMEOUT_SECONDS
    queue_tracker.DEDUP_IOU_THRESH = 0.40
    queue_tracker.DEDUP_CENTRE_FRAC = 0.15

    queue_tracker._num_counters = max(1, int(QUEUE_CONFIG.get('num_counters', 3)))
    queue_tracker.counters_closed_until = _counters_closed_until

    # Store the .env default so build_queue_prediction can label the source.
    QUEUE_CONFIG.setdefault("_default_avg_service_time",
                            QUEUE_CONFIG.get("avg_service_time", 3.0))

    threading.Thread(
        target=_service_time_refresh_loop,
        daemon=True,
        name="ServiceTimeRefresh",
    ).start()

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

    candidates: list[tuple[int, tuple, object, int]] = []
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
            candidates.append((tid, tuple(bbox), person.face_embedding, absent_frames))

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
        incoming_face = p.get("face_embedding")

        for c_tid, c_bbox, c_face, c_absent in candidates:
            if c_tid in used_cands:
                continue
            # Position alone cannot tell a lost track from a different person
            # stepping into the spot someone just left; remapping them handed
            # the first person's queue number to the second. When both faces
            # are known they must match; without a face, only a near-instant
            # track switch (the tracker losing the same person) is remapped.
            if c_face is not None and incoming_face:
                sim = face_service.cosine_similarity(
                    np.asarray(incoming_face, dtype=np.float32), c_face)
                if sim < tracker.FACE_MATCH_THRESHOLD:
                    continue
            elif c_absent > _REMAP_FACELESS_MAX_FRAMES:
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

def inject_face_embeddings(raw_tracked: list, tracker) -> None:
    """Refresh a tracked person's stored face embedding from the latest
    detector payload. Unlike the retired HSV signature (which was EMA-
    blended to smooth pixel noise), ArcFace embeddings for the same person
    are already stable frame-to-frame, so the freshest confident read is
    kept as-is rather than blended across possibly different angles."""
    for p in raw_tracked:
        embedding = p.get("face_embedding")
        if not embedding:
            continue

        tid = p["track_id"]
        person = tracker.active_queue.get(tid)
        if person is None:
            continue

        try:
            person.face_embedding = np.array(embedding, dtype=np.float32)
        except Exception as exc:
            print(f"[QueueService] face embedding injection error tid={tid}: {exc}")

def _try_link_students(raw_tracked: list) -> None:
    """For each presence-confirmed-but-unlinked person whose track carries a
    face embedding this frame, resolve their identity against enrolled
    students (Algorithm 4).

    Per SOP #2 (reconfirmed by the panel after the pre-oral revision
    meeting): a registered student recognized here gets a queue number
    minted directly by the system — no kiosk ticket needed. Walk-ins never
    reach a linked outcome here — an unenrolled face never produces an
    accepted match, so they're linked via the staff-assisted manual
    force_new_person() path instead, and the kiosk's own numbering for them
    is completely untouched by any of this.

    An unmatched (or not-yet-confident) face just stays pending — the same
    "don't guess, escalate to staff" path already used for anything else
    ambiguous (see get_pending_link_alerts()).

    A student only ever holds one active entry at a time: if they're
    recognized while already having a pending/waiting ticket, no new one is
    minted or linked — staff must mark the existing entry done (served or
    no-show) first.
    """
    for p in raw_tracked:
        track_id = p["track_id"]
        person = queue_tracker.active_queue.get(track_id)
        if person is None or person.identity_status != "pending_link":
            continue

        embedding = p.get("face_embedding")
        if not embedding:
            continue

        try:
            live_embedding = np.array(embedding, dtype=np.float32)
            match = embedding_index.match(
                live_embedding,
                match_threshold=FACE_MATCH_THRESHOLD,
                margin_threshold=FACE_MARGIN_THRESHOLD,
            )
            record_face_match_event(
                match.student_id, track_id, match.score, match.margin, match.accepted,
            )
            if not match.accepted:
                continue

            # Recognition is not consent. A registered student only gets a
            # number if they asked for one in the app first — otherwise
            # anyone merely walking past the camera (accompanying a friend,
            # passing through, working nearby) would be issued a ticket they
            # never wanted and silently added to the queue.
            #
            # Mark them so they are skipped cheaply on later frames instead
            # of re-running the match every frame, and so they never appear
            # as a pending entry or raise a staff alert. Everything that
            # surfaces pending people filters on 'pending_link'.
            if not queue_tracker.has_join_intent(match.student_id):
                # Remember who this is, so arm_join_intent() can find and
                # revive them the instant they do tap join — otherwise the
                # bystander mark is a one-way door and someone recognized
                # *before* tapping join could never get a number for as long
                # as the camera held that track.
                person.student_id = match.student_id
                person.identity_status = "bystander"
                continue

            # One active entry per student at a time — if they already have a
            # pending/waiting ticket (from an earlier recognition this
            # session), don't mint or link another. Staff must mark the
            # existing one done (served/no-show) before this student can be
            # queued again.
            #
            # A done_pending entry does not count. It is a finished
            # transaction that lingers in active_queue only until the cleanup
            # pass removes it, and that pass waits for the person to go
            # missing for several frames. Treating it as blocking meant a
            # student who had just been served could not be served again
            # without physically leaving the camera's view first, because
            # standing still kept their completed entry alive indefinitely.
            # Immediate re-entry is already governed by the spatial
            # done-cooldown, which is the guard actually designed for it.
            existing = queue_tracker.get_person_by_student_id(match.student_id)
            if existing is not None and existing.status != 'done_pending':
                continue

        except Exception as exc:
            print(f"[QueueService] student link error tid={track_id}: {exc}")
            continue

        # Read the join-intent timestamp BEFORE minting: minting consumes the
        # intent, so asking afterwards returns None and the metric silently
        # falls back to a less meaningful baseline.
        intent_at = queue_tracker.get_join_intent(match.student_id)

        linked = queue_tracker.mint_face_only_number(track_id, match.student_id)

        if linked is not None:
            _record_recognition_metric(linked, match, intent_at)

def _record_recognition_metric(person, match, intent_at=None) -> None:
    """
    Capture how long this recognition took and how many frames it cost, for
    the evaluation chapter (SOP #2). Never allowed to disturb the queue: a
    failure here is logged and dropped.

    `seconds_to_link` measures from the moment the system was first ABLE to
    issue a number, not from when the camera first saw the person. Those are
    very different once the Join-the-Queue consent gate exists: a student can
    stand in frame for minutes before opening the app, and measuring from
    first sighting charges all of that waiting to the system. A real run on
    2026-09-12 recorded 889 s that way — 14.8 minutes of a person standing in
    front of the camera during unrelated debugging — which would be a grossly
    misleading "time to recognition" if quoted.

    The baseline is therefore the later of presence-confirmation and
    join-intent, whichever happened last. The raw `first_seen_at`,
    `confirmed_at` and `linked_at` timestamps are still stored, so any other
    interpretation can be recomputed from the table afterwards.
    """
    try:
        first_seen = person.first_seen_at or person.entered_at
        confirmed = person.entered_at
        linked_at = person.linked_at or datetime.now()

        actionable_from = confirmed
        if intent_at is not None and intent_at > actionable_from:
            actionable_from = intent_at

        record_recognition_metric({
            "track_id": person.track_id,
            "student_id": person.student_id,
            "queue_number": person.queue_number,
            "linked_via": person.linked_via,
            "first_seen_at": first_seen,
            "confirmed_at": confirmed,
            "linked_at": linked_at,
            "seconds_to_confirm": (confirmed - first_seen).total_seconds(),
            "seconds_to_link": (linked_at - actionable_from).total_seconds(),
            "embed_attempts": person.embed_attempts,
            "embed_successes": person.embed_successes,
            "match_score": match.score,
            "match_margin": match.margin,
        })
    except Exception as exc:
        print(f"[QueueService] recognition metric skipped: {exc}")

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
                "face_embedding": p.get("face_embedding"),
            }
            for p in tracked_filtered
        ]
        tracked = remap_track_ids(tracked, queue_tracker)
        try:
            queue_state = queue_tracker.process_frame(tracked)
        except Exception as exc:
            print(f"[QueueService] process_frame error: {exc}")

        inject_face_embeddings(filtered, queue_tracker)
        try:
            _try_link_students(filtered)
        except Exception as exc:
            print(f"[QueueService] student linking pass error: {exc}")
    else:
        try:
            queue_state = queue_tracker.process_frame([])
        except Exception as exc:
            print(f"[QueueService] process_frame empty error: {exc}")

    return queue_state

def done_pending_people() -> list:
    return [
        p.to_dict()
        for p in queue_tracker.active_queue.values()
        if p.status == "done_pending"
    ]

def is_queue_number_active(queue_number: int) -> bool:
    return any(
        p.queue_number == queue_number and p.status in ("waiting", "missing")
        for p in queue_tracker.active_queue.values()
    )

def _update_status_then_invalidate(
    queue_number: int,
    status: str,
    actor_username: str | None,
    called_seconds_ago: int | None = None,
) -> None:
    """Persist a status change, then drop the ticket from the validation cache
    so the student's next status poll sees it at once."""
    try:
        person = next((p for p in queue_tracker.active_queue.values()
                       if p.queue_number == queue_number), None)
        cashier_window = getattr(person, "counter_number", None)
        update_queue_status(queue_number, status, actor_username,
                            called_seconds_ago=called_seconds_ago, cashier_window=cashier_window)
    finally:
        invalidate_ticket_cache(queue_number)

def _seconds_since_called(queue_number: int) -> int | None:
    """How long ago this person was called to a counter. Sent as an age, not a
    time: created_at and served_at come from the database's clock, which need
    not match this process's (the server is UTC, a laptop is not)."""
    called_at = queue_tracker.get_called_at(queue_number)
    if called_at is None:
        return None
    return max(0, int((datetime.now() - called_at).total_seconds()))


def reset_queue(actor_username: str | None = None) -> None:
    global queue_tracker
    invalidate_ticket_cache()
    deleted = delete_all_tickets()
    print(f"[Reset] Deleted {deleted} ticket PDF(s)")
    queue_tracker = QueueTracker(zone=queue_zone)
    wire_callbacks()
    threading.Thread(
        target=record_queue_reset,
        args=(actor_username,),
        daemon=True,
        name="DBAudit-queue-reset",
    ).start()

def mark_done(queue_number: int, actor_username: str | None = None) -> dict | None:
    called_seconds_ago = _seconds_since_called(queue_number)
    if not queue_tracker.mark_transaction_done(queue_number):
        return None

    invalidate_ticket_cache(queue_number)
    threading.Thread(
        target=_update_status_then_invalidate,
        args=(queue_number, "served", actor_username, called_seconds_ago),
        daemon=True,
        name=f"DBUpdate-Q{queue_number:03d}-served",
    ).start()
    return queue_tracker.get_state()

def mark_no_show(queue_number: int, actor_username: str | None = None) -> dict | None:
    """Staff action, not a detection. A member of staff who can see the queue
    area decides that the person whose turn it is has not come, and the
    system then bumps the number and moves the queue on."""
    called_seconds_ago = _seconds_since_called(queue_number)
    if not queue_tracker.mark_no_show(queue_number):
        return None

    invalidate_ticket_cache(queue_number)
    threading.Thread(
        target=_update_status_then_invalidate,
        args=(queue_number, "no_show", actor_username, called_seconds_ago),
        daemon=True,
        name=f"DBUpdate-Q{queue_number:03d}-noshow",
    ).start()
    return queue_tracker.get_state()

def link_pending_person(
    track_id: int,
    queue_number: int,
    student_id: int | None = None,
) -> dict | None:
    """Staff manual resolution for a person confirmed present but not yet
    linked (face/OCR match was ambiguous, or the digit read never firmed
    up) — the "escalate to a human, don't guess" fallback behind the
    PENDING_LINK_TIMEOUT_SECONDS alerts in get_state()."""
    person = queue_tracker.link_queue_number(
        track_id, queue_number, student_id=student_id, linked_via="manual",
    )
    return person.to_dict() if person else None

def force_new_person(queue_number: int, is_walkin: bool = True) -> dict | None:
    """Staff manual entry — link an already-printed kiosk number directly.
    This is the walk-in path (per the panel's directive, walk-ins keep their
    kiosk number as-is) and the fallback when CV/face/OCR fails for a
    student. Returns None if that number is already linked today."""
    return queue_tracker.force_new_person(queue_number, is_walkin=is_walkin)

def mark_on_the_way(queue_number: int) -> dict | None:
    return queue_tracker.mark_on_the_way(queue_number)

def record_on_the_way_signal(queue_number: int) -> dict:
    return queue_tracker.record_on_the_way_signal(queue_number)

def on_way_notification_state() -> dict:
    queue_state = queue_tracker.get_state()
    active_queue = queue_state.get("active_queue", [])
    notifications = list(queue_state.get("on_way_notifications", []))
    notification_ids = {item.get("id") for item in notifications}

    for person in active_queue:
        if person.get("on_the_way") is not True:
            continue
        queue_number = as_int(person.get("queue_number"), 0)
        queue_label = person.get("queue_label") or f"Q{queue_number:03d}"
        notification_id = (
            f"{queue_label}-{person.get('on_the_way_at') or 'active'}"
        )
        if notification_id in notification_ids:
            continue
        notifications.append({
            "id": notification_id,
            "queue_number": queue_number,
            "queue_label": queue_label,
            "created_at": person.get("on_the_way_at"),
            "created_at_display": person.get("on_the_way_at_display"),
            "message": f"{queue_label} is on the way to the queue zone.",
        })
        notification_ids.add(notification_id)

    # Age out here as well as in the tracker. This function rebuilds entries
    # from anyone still flagged on_the_way, so without the same cutoff a
    # person who kept that flag would regenerate their banner indefinitely
    # and the tracker's own pruning would never be visible to the dashboard.
    cutoff = datetime.now() - timedelta(
        seconds=queue_tracker.ON_WAY_NOTIFICATION_TTL_SECONDS
    )
    live = []
    for item in notifications:
        try:
            created = datetime.fromisoformat(str(item.get("created_at")))
        except (TypeError, ValueError):
            continue
        if created >= cutoff:
            live.append(item)

    return {
        "notifications": live[-20:],
        "active_queue": active_queue,
    }

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

def set_active_counters(
    counters: int,
    actor_username: str | None = None,
) -> dict:
    counters = max(1, as_int(counters, QUEUE_CONFIG["num_counters"]))
    with _config_lock:
        old_counters = max(1, as_int(QUEUE_CONFIG.get("num_counters"), 3))
        QUEUE_CONFIG["num_counters"] = counters
        avg_service_time = max(
            0.1,
            as_float(QUEUE_CONFIG.get("avg_service_time"), 3.0),
        )

    state.set_active_counters(counters)
    queue_tracker._num_counters = counters
    threading.Thread(
        target=record_counter_config_change,
        args=(old_counters, counters, avg_service_time, actor_username),
        daemon=True,
        name="DBAudit-counter-config",
    ).start()
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

def format_seconds(seconds: int) -> str:
    seconds = max(0, as_int(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    minutes, remaining = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {remaining}s" if remaining else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"

def eta_iso(wait_minutes: float) -> str:
    return (datetime.now() + timedelta(minutes=max(0.0, wait_minutes))).isoformat()

def estimator_version() -> str:
    """Logged with each ticket, so its estimate can be traced to its method."""
    return "queue-engine" if USE_QUEUE_ENGINE else wait_model.MODEL_VERSION

def counters_closed_until() -> datetime | None:
    until = queue_tracker.counters_closed_until
    return until if until is not None and until > datetime.now() else None

def school_time_today(clock) -> datetime:
    """A time of day on the school's clock, today, as the server's own naive
    local time (the form every datetime.now() in this code uses)."""
    school_now = datetime.now(SCHOOL_TZ)
    return datetime.combine(school_now.date(), clock, tzinfo=SCHOOL_TZ).astimezone().replace(tzinfo=None)

def school_iso(moment: datetime | None) -> str | None:
    """A server-local time as ISO 8601 with the school's offset, for clients."""
    return moment.astimezone().astimezone(SCHOOL_TZ).isoformat() if moment else None

def set_counters_closed_until(until: datetime | None) -> datetime | None:
    """Close every counter until `until` (lunch), or reopen with None."""
    global _counters_closed_until
    _counters_closed_until = until
    queue_tracker.counters_closed_until = until
    # Reopening calls the people at the front at once, not on the next frame.
    queue_tracker._recalculate_positions()
    return counters_closed_until()

def _waiting_ahead(line: list[dict], position: int) -> int | None:
    """People still waiting in front of `position`, not counting those already
    at a counter. None when the person at `position` is at a counter."""
    if any(q["position"] == position and q["counter_number"] is not None for q in line):
        return None
    known = [q for q in line if q["position"] < position]
    # Positions past the live line (the next arrival, or rows from the database
    # while the tracker is empty) count everyone before them as waiting.
    return sum(1 for q in known if q["counter_number"] is None) + max(0, position - 1 - len(known))

def _queue_engine_wait(position: int, counters: int, service_min: float) -> float:
    now = datetime.now()
    line = queue_tracker.line_snapshot()
    ahead = _waiting_ahead(line, position)
    if ahead is None:
        return 0.0
    called = {q["counter_number"]: q["called_at"] for q in line if q["counter_number"] is not None}
    closed = counters_closed_until()
    states = [
        queue_engine.CounterState(busy_since=called.get(n), closed_until=closed)
        for n in range(1, max(1, counters) + 1)
    ]
    return queue_engine.wait_minutes(now, states, ahead, service_min)

def estimate_wait_for_position(position: int, counters: int) -> float:
    # Position 0 means not in the live line, e.g. a ticket already served.
    if position <= 0:
        return 0.0
    if USE_QUEUE_ENGINE:
        service_min = runtime_config()["avg_service_time"]
        return round(_queue_engine_wait(position, counters, service_min), 1)
    return round(wait_model.predict_minutes(position - 1, counters), 1)

def wait_range_fields(wait_min: float, in_line: bool = True) -> dict:
    """The usual range around an estimate, for the app to show beside it.

    The range is the spread of the regression's errors on unseen days. The
    queue engine has no measured errors yet, so it shows no range rather than
    borrow one that was never measured for it."""
    if not in_line or USE_QUEUE_ENGINE:
        return {"estimated_wait_range_minutes": [0, 0], "estimated_wait_range_label": ""}
    low, high = wait_model.usual_range(wait_min)
    return {
        "estimated_wait_range_minutes": [low, high],
        "estimated_wait_range_label": f"usually {low}–{high} mins",
    }

def prediction_for_position(
    position: int,
    avg_service_time: float,
    counters: int,
    department: str | None = None,
) -> dict:
    def service_prediction(dep):
        fallback = max(0.0, float(avg_service_time) * 60.0)
        fallback_method = "global_average_fallback"
        if dep:
            cached = _department_average_cache.get(dep)
            if cached and time.monotonic() - cached[0] < 300:
                fallback = cached[1]
                fallback_method = "department_average_fallback"
            else:
                measured = measure_department_service_time(dep)
                if measured is not None and measured >= 0:
                    fallback = measured
                    fallback_method = "department_average_fallback"
                    _department_average_cache[dep] = (time.monotonic(), measured)
        return ets_prediction.predict_service_time(dep, fallback, fallback_method=fallback_method)

    own = service_prediction(department)
    wait_seconds = None
    method = own["method"]
    unavailable = bool(counters_closed_until())
    if unavailable:
        wait_seconds = 0.0
    if position > 0 and counters > 0 and not unavailable and not (not USE_QUEUE_ENGINE and department is None):
        try:
            from services.queue_simulator import simulate_department_wait_seconds
            active = live_or_db_active_queue()
            line = [{
                "position": as_int(item.get("position_in_line") or item.get("position"), index),
                "queue_number": item.get("queue_number"),
                "counter_number": item.get("counter_number"),
                "called_at": item.get("called_at"),
            } for index, item in enumerate(active, start=1)
                if item.get("status") in ("waiting", "missing") or item.get("counter_number") is not None]
            target = next((item for item in line if item["position"] == position), None)
            if target is None:
                raise ValueError("target queue entry is not available")
            target_number = as_int((target or {}).get("queue_number"), 0)
            records = {as_int(row.get("queue_number"), 0): row for row in _waiting_queue_records()}
            ahead = []
            busy = []
            now = datetime.now()
            for item in line:
                if item["position"] >= position:
                    continue
                record = records.get(as_int(item.get("queue_number"), 0), {})
                dep = record.get("department_at_issue")
                if item.get("counter_number") is not None:
                    duration = service_prediction(dep)["seconds"]
                    called_at = item.get("called_at")
                    elapsed = max(0.0, (now - called_at).total_seconds()) if called_at else 0.0
                    busy.append({"residual_seconds": max(0.0, duration - elapsed)})
                else:
                    ahead.append({"department": dep})
            wait_seconds = simulate_department_wait_seconds(
                ahead, counters, busy,
                lambda student: service_prediction(student.get("department"))["seconds"],
            )
            if target_number and target_number in records:
                department = department or records[target_number].get("department_at_issue")
        except Exception as exc:
            print(f"[Prediction] Department FIFO unavailable; using existing wait estimate ({type(exc).__name__})")
    if wait_seconds is None:
        wait_min = estimate_wait_for_position(position, counters) if position > 0 else 0.0
        wait_seconds = wait_min * 60.0
    wait_min = round(max(0.0, wait_seconds) / 60.0, 1)
    service_min = round(max(0.0, own["seconds"]) / 60.0, 1)
    active_windows = 0 if counters_closed_until() else counters
    return {
        "position": position,
        "estimated_wait_time": wait_min,
        "estimated_wait_time_minutes": wait_min,
        "estimated_wait_time_label": format_minutes(wait_min),
        **wait_range_fields(wait_min, in_line=position > 0),
        "estimated_service_start_at": eta_iso(wait_min),
        "estimated_service_finish_at": eta_iso(wait_min + service_min),
        "estimated_service_time_min": service_min,
        "estimated_service_time_seconds": int(round(own["seconds"])),
        "estimated_wait_time_seconds": int(round(max(0.0, wait_seconds))),
        "estimated_completion_duration_seconds": int(round(max(0.0, wait_seconds) + own["seconds"])),
        "department": department,
        "active_cashier_windows": active_windows,
        "people_ahead": max(0, position - 1),
        "prediction_method": method,
        "model_version": own["model_version"],
        "prediction_state": "paused" if unavailable else "available",
    }

def _as_datetime(value) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None

def ticket_record_status_response(record: dict) -> dict:
    queue_number = as_int(record.get("queue_number"), 0)
    status = str(record.get("status") or "expired")
    created_at = _as_datetime(record.get("created_at")) or datetime.now()
    ended_at = _as_datetime(record.get("served_at")) or datetime.now()
    wait_seconds = max(0, int((ended_at - created_at).total_seconds()))
    config = runtime_config()

    messages = {
        "served": "Your queue ticket is done. Please exit the queue area.",
        # Kept for historical records written before no-show detection was
        # removed; the system no longer produces this status.
        "no_show": "Your queue ticket is no longer active. Please contact staff if you still need service.",
        "expired": "Your queue ticket has expired. Please get a new ticket if you still need service.",
    }

    return {
        "queue_number": queue_number,
        "queue_label": f"Q{queue_number:03d}",
        "status": status,
        "position_in_line": 0,
        "wait_time": format_seconds(wait_seconds),
        "wait_time_seconds": wait_seconds,
        "joined_at": created_at.strftime("%I:%M:%S %p"),
        "joined_at_full": created_at.strftime("%b %d, %Y %I:%M:%S %p"),
        "joined_at_iso": created_at.isoformat(),
        "completed_at": ended_at.strftime("%I:%M:%S %p"),
        "completed_at_full": ended_at.strftime("%b %d, %Y %I:%M:%S %p"),
        "completed_at_iso": ended_at.isoformat(),

        "message": messages.get(status, "This queue ticket is no longer active."),
        "prediction": prediction_for_position(
            0,
            config["avg_service_time"],
            config["active_counters"],
        ),
    }

def ticket_record_waiting_response(record: dict) -> dict:
    queue_number = as_int(record.get("queue_number"), 0)
    created_at = _as_datetime(record.get("created_at")) or datetime.now()
    wait_seconds = max(0, int((datetime.now() - created_at).total_seconds()))
    config = runtime_config()

    return {
        "queue_number": queue_number,
        "queue_label": f"Q{queue_number:03d}",
        "status": "waiting",
        "position_in_line": 0,
        "wait_time": format_seconds(wait_seconds),
        "wait_time_seconds": wait_seconds,
        "joined_at": created_at.strftime("%I:%M:%S %p"),
        "joined_at_full": created_at.strftime("%b %d, %Y %I:%M:%S %p"),
        "joined_at_iso": created_at.isoformat(),

        "message": (
            "Your ticket is still active, but you are not currently visible "
            "in the live queue. Please return to the queue zone and ask staff "
            "if your number is not shown."
        ),
        "prediction": prediction_for_position(
            0,
            config["avg_service_time"],
            config["active_counters"],
        ),
    }

def _active_ticket_numbers() -> set[int]:
    return {
        p.queue_number
        for p in queue_tracker.active_queue.values()
        if p.status in ("waiting", "missing")
    }

def _fallback_person_from_ticket(record: dict, position: int) -> dict:
    queue_number = as_int(record.get("queue_number"), 0)
    created_at = _as_datetime(record.get("created_at")) or datetime.now()
    wait_seconds = max(0, int((datetime.now() - created_at).total_seconds()))
    return {
        "queue_number": queue_number,
        "queue_label": f"Q{queue_number:03d}",
        "status": "missing",
        "position_in_line": position,
        "position": position,
        "wait_time": format_seconds(wait_seconds),
        "wait_time_seconds": wait_seconds,
        "joined_at": created_at.strftime("%I:%M:%S %p"),
        "joined_at_full": created_at.strftime("%b %d, %Y %I:%M:%S %p"),
        "joined_at_iso": created_at.isoformat(),
        "on_the_way": False,
        "on_the_way_at": None,
        "on_the_way_at_display": None,
        "source": "database_waiting_ticket",
        "message": (
            "Ticket is still waiting in the database, but the person is not "
            "currently visible to the detector."
        ),
    }

# Kiosk tickets that are waiting but that the camera has not linked to a
# tracked person yet — the fallback behind the live tracker. Reading them
# costs a round trip to the managed database, and /api/queue/prediction is
# polled every five seconds by every phone as well as the staff dashboard, so
# the uncached version spent about 2.7s of the deployed instance on every
# request re-reading rows that change at human speed. Measured on the live
# database: 242ms to take a pooled connection, 498ms for the select itself.
#
# A few seconds of staleness is invisible here. These records only move when
# somebody takes a ticket at the kiosk or staff marks one served, and the
# parts of the queue that do change continuously — positions, counts, who is
# being served — come from the in-memory tracker above and are never cached.
#
# The window is longer than every poller's interval on purpose. The staff
# dashboard asks every 3s and each phone every 5s, so a TTL shorter than
# those would expire before the next caller arrived and cache nothing. At 10s
# one read serves every client in that window, whatever their number.
_WAITING_RECORDS_TTL_SECONDS = 10.0
_waiting_records_cache: list[dict] = []
_waiting_records_read_at = 0.0
_waiting_records_refreshing = False
_waiting_records_lock = threading.Lock()


def _refresh_waiting_records() -> list[dict]:
    global _waiting_records_cache, _waiting_records_read_at, _waiting_records_refreshing
    try:
        records = fetch_waiting_queue_records()
        with _waiting_records_lock:
            _waiting_records_cache = records
            _waiting_records_read_at = time.monotonic()
        return records
    finally:
        with _waiting_records_lock:
            _waiting_records_refreshing = False


def _waiting_queue_records() -> list[dict]:
    """Stale-while-revalidate. Under load, every request that found the window
    expired used to read the database at once, and each paid seconds for the
    round trip. Now a single background read refreshes the list while callers
    keep the last one; only the very first call, with nothing cached, waits."""
    global _waiting_records_refreshing

    with _waiting_records_lock:
        age = time.monotonic() - _waiting_records_read_at
        if _waiting_records_read_at and age < _WAITING_RECORDS_TTL_SECONDS:
            return _waiting_records_cache
        have_previous = bool(_waiting_records_read_at)
        start_refresh = not _waiting_records_refreshing
        if start_refresh:
            _waiting_records_refreshing = True
        previous = _waiting_records_cache

    if have_previous:
        if start_refresh:
            threading.Thread(
                target=_refresh_waiting_records, daemon=True, name="WaitingRecordsRefresh"
            ).start()
        return previous
    if start_refresh:
        return _refresh_waiting_records()
    return previous

def live_or_db_active_queue() -> list[dict]:
    queue_state = queue_tracker.get_state()
    active = list(queue_state.get("active_queue", []))
    seen_numbers = {as_int(person.get("queue_number"), 0) for person in active}

    for record in _waiting_queue_records():
        queue_number = as_int(record.get("queue_number"), 0)
        if not queue_number or queue_number in seen_numbers:
            continue
        active.append(_fallback_person_from_ticket(record, len(active) + 1))
        seen_numbers.add(queue_number)

    active.sort(key=lambda person: as_int(person.get("queue_number"), 0))
    for index, person in enumerate(active, start=1):
        if as_int(person.get("position_in_line"), 0) <= 0:
            person["position_in_line"] = index
        if as_int(person.get("position"), 0) <= 0:
            person["position"] = person["position_in_line"]
    return active

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

    active_queue = live_or_db_active_queue()
    queue_count = len(active_queue)
    config = runtime_config()
    counters = config["active_counters"]
    avg_service_time = config["avg_service_time"]
    data_age_seconds = max(0.0, time.time() - as_float(metrics["timestamp"], time.time()))
    if USE_QUEUE_ENGINE:
        new_arrival_wait = estimate_wait_for_position(queue_count + 1, counters)
        # The line ahead of a later arrival, projected with the camera's rate
        # of growth in people per minute (never negative). Not validated.
        growth = max(0.0, as_float(metrics["arrival_rate"], 0.0))
        horizon_waits = {
            h: min(120.0, estimate_wait_for_position(queue_count + 1 + round(growth * h), counters))
            for h in (5, 15, 30)
        }
    else:
        current_wait = as_float(metrics["estimated_wait_time"], 0.0)
        fallback_new_wait = estimate_wait_for_position(queue_count + 1, counters)
        new_arrival_wait = round(
            current_wait if current_wait > 0 else fallback_new_wait,
            1,
        )
        horizon_waits = {
            h: as_float(metrics[f"predicted_wait_{h}min"], new_arrival_wait) for h in (5, 15, 30)
        }
    closed_until = counters_closed_until()

    people = []
    for person in active_queue:
        position = as_int(person.get("position_in_line"), 0)
        people.append({
            "queue_number": person.get("queue_number"),
            "queue_label": person.get("queue_label"),
            "status": person.get("status"),
            **prediction_for_position(position, avg_service_time, counters),
        })

    _default_svc_time = float(QUEUE_CONFIG.get("_default_avg_service_time",
                                               QUEUE_CONFIG.get("avg_service_time", 3.0)))
    service_time_source = (
        "measured" if abs(avg_service_time - _default_svc_time) > 0.05 else "default"
    )

    return {
        "queue_length": queue_count,
        "active_counters": counters,
        "avg_service_time_min": round(avg_service_time, 2),
        "service_time_source": service_time_source,
        "arrival_rate_per_min": as_float(metrics["arrival_rate"], 0.0),
        "system_utilization": as_float(metrics["system_utilization"], 0.0),
        "data_age_seconds": round(data_age_seconds, 1),
        "data_status": "stale" if data_age_seconds > 15 else "live",
        "wait_estimator": estimator_version(),
        "counters_closed_until": school_iso(closed_until),
        "new_arrival": {
            "position": queue_count + 1,
            "estimated_wait_time": new_arrival_wait,
            "estimated_wait_time_minutes": new_arrival_wait,
            "estimated_wait_time_label": format_minutes(new_arrival_wait),
            "estimated_service_start_at": eta_iso(new_arrival_wait),
            **wait_range_fields(new_arrival_wait),
        },
        "forecast": {
            "now": {
                "horizon_minutes": 0,
                "estimated_wait_time": round(new_arrival_wait, 1),
                "estimated_wait_time_minutes": round(new_arrival_wait, 1),
                "estimated_wait_time_label": format_minutes(new_arrival_wait),
            },
            **{
                f"in_{h}min": {
                    "horizon_minutes": h,
                    "estimated_wait_time": round(minutes, 1),
                    "estimated_wait_time_minutes": round(minutes, 1),
                    "estimated_wait_time_label": format_minutes(minutes),
                }
                for h, minutes in horizon_waits.items()
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
) -> str:
    if data_status == "stale":
        return "Detector updates are stale. Check the camera process before trusting the trend."
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
    active_queue = live_or_db_active_queue()
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
    total_served = sum(
        1 for record in completed_all
        if record.get("bump_reason") == "served"
    )
    # Counted separately from served on purpose: a no-show never reached
    # the counter, so folding it into the served total would overstate
    # both completed tickets and any service rate derived from them.
    total_no_show = sum(
        1 for record in completed_all
        if record.get("bump_reason") == "no_show"
    )
    # No internal counter mints numbers anymore (they always come from the
    # kiosk) — "assigned today" is everyone currently active plus everyone
    # already completed.
    total_assigned = queue_length + total_completed

    data_status = prediction.get("data_status", "live")
    utilization = as_float(prediction.get("system_utilization"), 0.0)

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
            "pending_count": queue_state.get("pending_count", 0),
            "total_assigned": total_assigned,
            "total_completed": total_completed,
            "total_served": total_served,
            "total_no_show": total_no_show,
            "completion_rate_percent": round(
                (total_completed / total_assigned) * 100,
                1,
            ) if total_assigned else 0.0,
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
        "on_way_notifications": queue_state.get("on_way_notifications", []),
        "appearance_rejections": queue_state.get("appearance_rejections", []),
        "zone": zone_dict(),
        "recommendation": _analytics_recommendation(
            data_status,
            utilization,
            queue_length,
            missing_count,
        ),
    }

wire_callbacks()
