import threading
import time
from collections import deque
from dataclasses import dataclass
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
    QUEUE_CALL_MODE,
    QUEUE_CONFIG,
    QUEUE_DEDUP_CENTRE_FRAC,
    QUEUE_DEDUP_IOU_THRESH,
    QUEUE_MAX_MISSING_FRAMES,
    QUEUE_MIN_CONFIRM_FACE_FRAMES,
    QUEUE_MIN_CONFIRM_FRAMES,
    QUEUE_MIN_CONFIRM_SECONDS,
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
    fetch_queue_records,
    fetch_recent_service_seconds,
    get_all_active_embeddings,
    record_counter_config_change,
    record_face_match_event,
    record_recognition_metric,
    record_queue_reset,
    update_queue_status,
)
from services.queue_tracker import QueueTracker, QueueZone
from services import face_service, queue_engine, ticket_service, wait_model
from services.embedding_index import EmbeddingIndex
from services.ticket_printer import delete_all_tickets, invalidate_ticket_cache
from core.school_time import CLOCK_FULL, clock, db_clock, db_on_school_clock

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

USE_QUEUE_ENGINE = WAIT_ESTIMATOR != "regression"
SCHOOL_TZ = timezone(timedelta(hours=SCHOOL_UTC_OFFSET_HOURS))
# Kept here as well as on the tracker so a queue reset does not reopen
# counters that staff closed.
_counters_closed_until: datetime | None = None
# "staff" or "auto". Starts from QUEUE_CALL_MODE; staff switch it from the
# dashboard. Kept here, like the lunch closure, so a queue reset keeps it.
_call_mode: str = QUEUE_CALL_MODE

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
    person = queue_tracker._person_by_queue_number(queue_number)
    try:
        ticket_service.enqueue_ticket(
            queue_number=queue_number,
            position=position,
            est_wait_min=estimate_wait_for_position(position, counters),
            student_id=student_id,
            linked_via=linked_via,
            is_walkin=is_walkin,
            # Logged with the ticket, so the estimate the student saw can
            # later be checked against the wait they actually had.
            people_ahead=max(0, position - 1),
            # The windows the estimate counted, so the log shows what it
            # assumed (the configured count when calling is automatic).
            active_counters=len(serving_windows(counters)),
            estimator_version=estimator_version(),
            payment_type=getattr(person, "payment_type", None),
            payment_amount=getattr(person, "payment_amount", None),
            payment_note=getattr(person, "payment_note", None),
        )
        print(f"[QueueService] Q{queue_number:03d} ({linked_via}) queued for ticket generation")
    except Exception:
        print(f"[QueueService] Ticket queue full - Q{queue_number:03d} skipped")

# HOW LONG RECENT SERVICES TOOK
#
# Each service staff mark done is timed from the call to the window to Done,
# and kept here with when it finished. The wait estimate's current average
# service time is the last ten of these from the past two hours
# (queue_engine.service_model). Timed here, not re-read from the database, so
# the next estimate already knows; read from the database once after a
# restart, which empties this.
#
# Services shorter than MIN_TIMED_SERVICE_SECONDS are a mis-click or a test;
# longer than MAX_TIMED_SERVICE_SECONDS, a Done that was forgotten.
MIN_TIMED_SERVICE_SECONDS = 5
MAX_TIMED_SERVICE_SECONDS = 30 * 60
_recent_services: deque = deque(maxlen=50)   # (time.monotonic() at Done, seconds)
_recent_services_lock = threading.Lock()
_recent_services_seeded = False


def record_service_time(service_seconds: int | None, finished_ago_seconds: float = 0.0) -> bool:
    """Count one finished service in the current average. False when ignored."""
    if service_seconds is None:
        return False
    if not MIN_TIMED_SERVICE_SECONDS <= service_seconds <= MAX_TIMED_SERVICE_SECONDS:
        return False
    with _recent_services_lock:
        _recent_services.append((time.monotonic() - max(0.0, finished_ago_seconds), int(service_seconds)))
    return True


def recent_service_minutes() -> list[float]:
    """Services finished in the past two hours, oldest first, in minutes."""
    horizon = time.monotonic() - queue_engine.RECENT_WINDOW_MIN * 60
    with _recent_services_lock:
        return [seconds / 60 for finished, seconds in _recent_services if finished >= horizon]


def seed_recent_services() -> None:
    """At startup: the services finished in the past two hours, from the
    database, so a restart does not drop the average back to the long-run one."""
    global _recent_services_seeded
    if _recent_services_seeded:
        return
    _recent_services_seeded = True
    try:
        rows = fetch_recent_service_seconds(int(queue_engine.RECENT_WINDOW_MIN),
                                            queue_engine.RECENT_SERVICES)
        with _recent_services_lock:
            already = len(_recent_services)
        if already:
            return
        for finished_ago, seconds in rows:
            record_service_time(seconds, finished_ago)
        print(f"[QueueService] {len(rows)} recent service time(s) read for the live average")
    except Exception as exc:
        print(f"[QueueService] could not read recent service times: {exc}")

def wire_callbacks() -> None:
    queue_tracker.on_number_linked = _on_number_linked
    queue_tracker.auto_call = _call_mode == "auto"
    queue_tracker.MAX_MISSING_FRAMES = QUEUE_MAX_MISSING_FRAMES
    queue_tracker.MIN_CONFIRM_FRAMES = QUEUE_MIN_CONFIRM_FRAMES
    queue_tracker.MIN_CONFIRM_SECONDS = QUEUE_MIN_CONFIRM_SECONDS
    queue_tracker.MIN_CONFIRM_FACE_FRAMES = QUEUE_MIN_CONFIRM_FACE_FRAMES
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

            # A student who stepped out of view and came back: their face no
            # longer matched the last capture of it, so the camera confirmed
            # them as a new, unidentified person while their own number sat
            # "missing" on the staff screen. Their registration signature does
            # match, so their number goes back to them on this track. This
            # comes before the join-intent check, because the request that
            # produced their number was used up when the number was issued.
            existing = queue_tracker.get_person_by_student_id(match.student_id)
            if existing is not None and queue_tracker.return_to_track(existing, track_id, live_embedding):
                label = f"Q{existing.queue_number:03d}" if existing.queue_number is not None else "(entry)"
                print(f"↩️  {label} is back: student {match.student_id} recognised on track {track_id}")
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
        detector_ids = [p["track_id"] for p in tracked]
        tracked = remap_track_ids(tracked, queue_tracker)
        # The detector's number -> the number the tracker holds the person
        # under, for people the detector renumbered (remap_track_ids keeps the
        # order). Faces arrive under the detector's number.
        renumbered = {
            old: new["track_id"] for old, new in zip(detector_ids, tracked)
            if old != new["track_id"]
        }
        _remember_detector_numbers(renumbered)
        try:
            queue_state = queue_tracker.process_frame(tracked)
        except Exception as exc:
            print(f"[QueueService] process_frame error: {exc}")

        # A renumbered person's face must reach them under the tracker's
        # number. It used to be looked up under the detector's, which the
        # tracker does not know, so it was dropped on every frame: the person
        # could never be recognised. Two people in view make the detector
        # renumber far more often (23 Nov 2026).
        if renumbered:
            filtered = [
                {**p, "track_id": renumbered.get(p["track_id"], p["track_id"])}
                for p in filtered
            ]
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


# The detector's track number -> the tracker's, for the people the detector
# renumbered in the latest update (see process_tracked_persons).
_detector_numbers: dict = {}


def _remember_detector_numbers(renumbered: dict) -> None:
    global _detector_numbers
    _detector_numbers = dict(renumbered)


def face_resolved_track_ids() -> list:
    """Tracks whose identity is settled, so the detector need only refresh
    their faces now and then: students with a number, numbered walk-ins, and
    students recognized but not queueing (a fresh face from them is still
    used, at a slower pace, if they tap join). Everyone else gets the
    detector's face budget first.

    Given in the detector's own numbers too: for a person it renumbered,
    the tracker's number means nothing to it, and it would keep spending its
    face budget on someone already settled instead of on the person waiting."""
    resolved = [
        tid for tid, p in queue_tracker.active_queue.items()
        if p.queue_number is not None or p.identity_status in ("linked", "bystander")
    ]
    settled = set(resolved)
    resolved += [det for det, tid in _detector_numbers.items() if tid in settled]
    return resolved

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
    cashier_window: int | None = None,
) -> None:
    """Persist a status change, then drop the ticket from the validation cache
    so the student's next status poll sees it at once."""
    try:
        update_queue_status(
            queue_number, status, actor_username,
            called_seconds_ago=called_seconds_ago, cashier_window=cashier_window,
        )
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
    previous = queue_tracker
    queue_tracker = QueueTracker(zone=queue_zone)
    wire_callbacks()
    # Students who tapped Get a Queue Number and are still waiting for the
    # camera keep their request; the reset clears the line, not them.
    kept = queue_tracker.carry_join_intents_from(previous)
    if kept:
        print(f"[Reset] Kept {kept} student request(s) for a number")
    threading.Thread(
        target=record_queue_reset,
        args=(actor_username,),
        daemon=True,
        name="DBAudit-queue-reset",
    ).start()

# WINDOWS HELD BY STAFF ACCOUNTS
#
# Each signed-in staff member works one window, and one window belongs to one
# account at a time, so two people never call numbers to the same window. The
# dashboard renews its claim while it is open; a claim it stops renewing (the
# browser was closed without logging out) lapses after WINDOW_CLAIM_TTL_SECONDS.
WINDOW_CLAIM_TTL_SECONDS = 120
_window_claims: dict[int, tuple[str, float]] = {}
_window_claims_lock = threading.Lock()


class WindowTaken(Exception):
    def __init__(self, counter_number: int, holder: str):
        super().__init__(f"Window {counter_number} is being used by {holder}.")
        self.counter_number = counter_number
        self.holder = holder


def _purge_expired_claims(now: float) -> None:
    for counter, (_, renewed) in list(_window_claims.items()):
        if now - renewed > WINDOW_CLAIM_TTL_SECONDS:
            del _window_claims[counter]


def claim_window(username: str, counter_number: int) -> None:
    """Hold `counter_number` for `username`, or renew their hold on it.
    A staff member holds one window: claiming another releases the first."""
    if not 1 <= counter_number <= queue_tracker._num_counters:
        raise ValueError(f"There is no window {counter_number}.")
    now = time.monotonic()
    with _window_claims_lock:
        _purge_expired_claims(now)
        holder = _window_claims.get(counter_number)
        if holder is not None and holder[0] != username:
            raise WindowTaken(counter_number, holder[0])
        for counter, (user, _) in list(_window_claims.items()):
            if user == username and counter != counter_number:
                del _window_claims[counter]
        _window_claims[counter_number] = (username, now)


def release_window(username: str) -> None:
    with _window_claims_lock:
        for counter, (user, _) in list(_window_claims.items()):
            if user == username:
                del _window_claims[counter]


def window_overview(username: str) -> dict:
    now = time.monotonic()
    with _window_claims_lock:
        _purge_expired_claims(now)
        claims = {counter: user for counter, (user, _) in _window_claims.items()}
    count = queue_tracker._num_counters
    return {
        "call_mode": "auto" if queue_tracker.auto_call else "staff",
        "num_counters": count,
        "my_window": next((c for c, u in claims.items() if u == username), None),
        "windows": [
            {"counter_number": c, "claimed_by": claims.get(c)} for c in range(1, count + 1)
        ],
    }


def set_call_mode(mode: str, actor_username: str | None = None) -> str:
    """Switch every window between staff calling (Call next) and automatic
    calling. Lasts until changed again or the server restarts, when
    QUEUE_CALL_MODE applies again."""
    global _call_mode
    if mode not in ("staff", "auto"):
        raise ValueError('Call mode must be "staff" or "auto".')
    _call_mode = mode
    queue_tracker.auto_call = mode == "auto"
    # Switching to automatic fills the free windows at once, not on the next
    # camera update; switching to staff calling leaves current calls as they are.
    queue_tracker._recalculate_positions()
    print(f"[QueueService] call mode set to {mode} by {actor_username or 'unknown'}")
    return mode


def call_next(counter_number: int, username: str) -> tuple[str, dict | None]:
    """The staff member at `counter_number` calls the next number. Their
    hold on the window is checked (and renewed) first."""
    claim_window(username, counter_number)
    outcome, person = queue_tracker.call_next(counter_number)
    if outcome == "called" and person is not None:
        invalidate_ticket_cache(person.get("queue_number"))
    return outcome, person


def mark_done(queue_number: int, actor_username: str | None = None) -> dict | None:
    # Read before the entry is retired; afterwards the tracker no longer has it.
    called_seconds_ago = _seconds_since_called(queue_number)
    cashier_window = queue_tracker.get_counter_number(queue_number)
    if not queue_tracker.mark_transaction_done(queue_number):
        return None

    # Timed from the call to Done; the next estimate counts it.
    record_service_time(called_seconds_ago)
    invalidate_ticket_cache(queue_number)
    threading.Thread(
        target=_update_status_then_invalidate,
        args=(queue_number, "served", actor_username, called_seconds_ago, cashier_window),
        daemon=True,
        name=f"DBUpdate-Q{queue_number:03d}-served",
    ).start()
    return queue_tracker.get_state()

def mark_no_show(queue_number: int, actor_username: str | None = None) -> dict | None:
    """Staff action, not a detection. A member of staff who can see the queue
    area decides that the person whose turn it is has not come, and the
    system then bumps the number and moves the queue on."""
    called_seconds_ago = _seconds_since_called(queue_number)
    cashier_window = queue_tracker.get_counter_number(queue_number)
    if not queue_tracker.mark_no_show(queue_number):
        return None

    invalidate_ticket_cache(queue_number)
    threading.Thread(
        target=_update_status_then_invalidate,
        args=(queue_number, "no_show", actor_username, called_seconds_ago, cashier_window),
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

def format_wait_seconds(seconds: float) -> str:
    """A wait in minutes and seconds: '3 min 05 sec', '45 sec', 'now'."""
    seconds = max(0, int(round(seconds)))
    if seconds == 0:
        return "now"
    if seconds < 60:
        return f"{seconds} sec"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {rest:02d} sec"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} hr {minutes} min"

def clock_label(moment: datetime) -> str:
    """'2:41:30 PM' on the school's clock."""
    return clock(moment, "%I:%M:%S %p").lstrip("0")

# Logged with each ticket (wait_model_version), so estimates made before and
# after the real-time engine can be told apart when checked against real waits.
ENGINE_VERSION = "queue-engine-v2"

def estimator_version() -> str:
    """Logged with each ticket, so its estimate can be traced to its method."""
    return ENGINE_VERSION if USE_QUEUE_ENGINE else wait_model.MODEL_VERSION

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

def school_time_label(moment: datetime) -> str:
    """'1:00 PM' on the school's clock, for messages to staff."""
    return moment.astimezone().astimezone(SCHOOL_TZ).strftime("%I:%M %p").lstrip("0")

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

def calling_windows(counters: int, line: list[dict] | None = None) -> list[int]:
    """The windows taking people now: every open one in automatic calling;
    in staff calling, those a staff member holds or where someone is being
    served. Empty when no window is calling."""
    configured = list(range(1, max(1, counters) + 1))
    if queue_tracker.auto_call:
        return configured
    now = time.monotonic()
    with _window_claims_lock:
        _purge_expired_claims(now)
        held = {c for c in _window_claims if c in configured}
    if line is None:
        line = queue_tracker.line_snapshot()
    busy = {q["counter_number"] for q in line if q.get("counter_number") in configured}
    return sorted(held | busy)


def serving_windows(counters: int, line: list[dict] | None = None) -> list[int]:
    """The windows the wait estimate counts as serving.

    In automatic calling every open window takes the next person. In staff
    calling a window takes people only while a staff member holds it, so only
    the windows held count, together with any where someone is being served.
    Counting every configured window instead made the estimate assume windows
    nobody was working: in the user acceptance test three were set open while
    one was staffed, and every student was told there was no wait.

    With staff calling and no window held, nobody is calling yet. One window
    is assumed, which overstates the wait rather than promising none.
    """
    return calling_windows(counters, line) or [1]


@dataclass(frozen=True)
class LineState:
    """What one round of estimates is worked out from, read once per request
    so every person in the same response is timed against the same moment."""
    now: datetime
    line: list
    model: queue_engine.ServiceModel
    counters: list
    windows_calling: int = 0   # windows actually calling (0: one is assumed)

    @property
    def moving(self) -> bool:
        """Someone is at a window, so the line moves on its own. With nobody
        at one it waits for a window to call, and no estimate counts down."""
        return any(c.busy_since is not None for c in self.counters)


def line_state(counters: int) -> LineState:
    """The line now: who is at which serving window and since when, and the
    current average service time (queue_engine.service_model)."""
    now = datetime.now()
    line = queue_tracker.line_snapshot()
    called = {q["counter_number"]: q["called_at"] for q in line if q["counter_number"] is not None}
    at_window = [(now - at).total_seconds() / 60 for at in called.values() if at is not None]
    model = queue_engine.service_model(
        recent_service_minutes(), at_window, prior_min=runtime_config()["avg_service_time"],
    )
    closed = counters_closed_until()
    calling = calling_windows(counters, line)
    states = [
        queue_engine.CounterState(busy_since=called.get(n), closed_until=closed)
        for n in (calling or [1])
    ]
    return LineState(now=now, line=line, model=model, counters=states, windows_calling=len(calling))


def wait_minutes_for_position(position: int, counters: int, state: LineState | None = None) -> float:
    """Minutes until the person at `position` reaches a window, unrounded."""
    # Position 0 means not in the live line, e.g. a ticket already served.
    if position <= 0:
        return 0.0
    if not USE_QUEUE_ENGINE:
        return wait_model.predict_minutes(position - 1, len(serving_windows(counters)))
    state = state or line_state(counters)
    ahead = _waiting_ahead(state.line, position)
    if ahead is None:
        return 0.0
    return queue_engine.wait_minutes(state.now, state.counters, ahead, state.model)


def estimate_wait_for_position(position: int, counters: int) -> float:
    return round(wait_minutes_for_position(position, counters), 1)


def windows_running_long(state: LineState) -> list[dict]:
    """Windows whose transaction has run longer than the current average."""
    average = state.model.mean_min * 60
    out = []
    for q in state.line:
        if q["counter_number"] is None or q["called_at"] is None:
            continue
        elapsed = (state.now - q["called_at"]).total_seconds()
        if elapsed > average:
            out.append({"counter_number": q["counter_number"], "queue_number": q["queue_number"],
                        "serving_seconds": int(elapsed)})
    return sorted(out, key=lambda w: w["counter_number"])


def adjustment_note(state: LineState) -> str:
    """Why a waiting student's time is moving later, when it is."""
    slow = windows_running_long(state)
    if not slow:
        return ""
    w = slow[0]
    return (
        f"Window {w['counter_number']} has been serving Q{w['queue_number']:03d} for "
        f"{format_wait_seconds(w['serving_seconds'])}, longer than the average of "
        f"{format_wait_seconds(state.model.mean_min * 60)}, so your time moved later."
    )


def average_service_fields(model: queue_engine.ServiceModel) -> dict:
    """The current average service time, for students and staff to see."""
    seconds = int(round(model.mean_min * 60))
    if model.timed:
        basis = f"last {model.timed} service{'' if model.timed == 1 else 's'}"
    else:
        basis = "usual pace"
    return {
        "avg_service_seconds": seconds,
        "avg_service_label": format_wait_seconds(seconds),
        "avg_service_basis": basis,
    }

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
    counters: int,
    state: LineState | None = None,
) -> dict:
    """One person's wait in seconds, the clock time their turn is expected
    (estimated time of service), and the average service time behind it.
    Worked out again on every request, so it moves with the line: the app
    counts the seconds down between requests and takes the new figure each
    time it asks."""
    state = state or line_state(counters)
    minutes = wait_minutes_for_position(position, counters, state)
    seconds = int(round(minutes * 60))
    start = state.now + timedelta(seconds=seconds)
    service_min = state.model.mean_min
    waiting = position > 0 and seconds > 0
    ahead = _waiting_ahead(state.line, position) if position > 0 else 0
    return {
        "position": position,
        "estimated_wait_time": round(minutes, 1),
        "estimated_wait_time_minutes": round(minutes, 1),
        "estimated_wait_seconds": seconds,
        "estimated_wait_time_label": format_wait_seconds(seconds),
        **wait_range_fields(minutes, in_line=position > 0),
        "estimated_service_start_at": school_iso(start),
        "estimated_service_start_label": clock_label(start),
        "estimated_service_finish_at": school_iso(start + timedelta(minutes=service_min)),
        "estimated_service_time_min": round(service_min, 2),
        **average_service_fields(state.model),
        "computed_at": school_iso(state.now),
        "adjustment_note": adjustment_note(state) if waiting else "",
        # What the estimate was worked out from, so the app can say so.
        "people_ahead": ahead or 0,
        "windows_counted": len(state.counters),
        "windows_calling": state.windows_calling,
        "line_moving": state.moving,
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

NO_SHOW_MESSAGE = (
    "Staff marked you as not present when your number was called, so your "
    "number was skipped. Tap Get a Queue Number if you still need the cashier."
)


def ticket_record_status_response(record: dict) -> dict:
    queue_number = as_int(record.get("queue_number"), 0)
    status = str(record.get("status") or "expired")
    created_at = _as_datetime(record.get("created_at")) or datetime.now()
    ended_at = _as_datetime(record.get("served_at")) or datetime.now()
    wait_seconds = max(0, int((ended_at - created_at).total_seconds()))
    config = runtime_config()

    messages = {
        "served": "Your queue ticket is done. Please exit the queue area.",
        # Staff mark a no-show from the dashboard; nothing is decided automatically.
        "no_show": NO_SHOW_MESSAGE,
        "expired": "Your queue ticket has expired. Please get a new ticket if you still need service.",
    }

    return {
        "queue_number": queue_number,
        "queue_label": f"Q{queue_number:03d}",
        "status": status,
        "position_in_line": 0,
        "wait_time": format_seconds(wait_seconds),
        "wait_time_seconds": wait_seconds,
        "joined_at": db_clock(created_at),
        "joined_at_full": db_clock(created_at, CLOCK_FULL),
        "joined_at_iso": created_at.isoformat(),
        "completed_at": db_clock(ended_at),
        "completed_at_full": db_clock(ended_at, CLOCK_FULL),
        "completed_at_iso": ended_at.isoformat(),

        "message": messages.get(status, "This queue ticket is no longer active."),
        "prediction": prediction_for_position(0, config["active_counters"]),
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
        "joined_at": db_clock(created_at),
        "joined_at_full": db_clock(created_at, CLOCK_FULL),
        "joined_at_iso": created_at.isoformat(),

        "message": (
            "Your ticket is still active, but you are not currently visible "
            "in the live queue. Please return to the queue zone and ask staff "
            "if your number is not shown."
        ),
        "prediction": prediction_for_position(0, config["active_counters"]),
    }

def _active_ticket_numbers() -> set[int]:
    return {
        p.queue_number
        for p in queue_tracker.active_queue.values()
        if p.status in ("waiting", "missing")
    }

def live_or_db_active_queue() -> list[dict]:
    """The line as the tracker holds it: the same people staff see and call.

    Tickets still 'waiting' in the database but unknown to the tracker used to
    be added here, a fallback from when kiosk tickets were printed before the
    camera linked them. A ticket row is now written only after the tracker has
    the student, so a database-only ticket is one a restart or a queue reset
    lost. Nobody can call it or mark it done, and counting it showed students
    people who were not there ("2 in line" with the queue area empty, 20 Nov
    2026) and pushed every estimate back behind them. The student holding one
    is told their ticket is active but not in the live queue
    (ticket_record_waiting_response)."""
    queue_state = queue_tracker.get_state()
    active = list(queue_state.get("active_queue", []))

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
    # One reading of the line for everyone in this response.
    line = line_state(counters)
    model = line.model
    data_age_seconds = max(0.0, time.time() - as_float(metrics["timestamp"], time.time()))
    new_arrival = prediction_for_position(queue_count + 1, counters, line)
    if USE_QUEUE_ENGINE:
        new_arrival_wait = new_arrival["estimated_wait_time_minutes"]
        # The line ahead of a later arrival, projected with the camera's rate
        # of growth in people per minute (never negative). Not validated.
        growth = max(0.0, as_float(metrics["arrival_rate"], 0.0))
        horizon_waits = {
            h: min(120.0, round(wait_minutes_for_position(queue_count + 1 + round(growth * h), counters, line), 1))
            for h in (5, 15, 30)
        }
    else:
        current_wait = as_float(metrics["estimated_wait_time"], 0.0)
        fallback_new_wait = new_arrival["estimated_wait_time_minutes"]
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
            **prediction_for_position(position, counters, line),
        })

    # Who is at each window and for how long, for the app's "Now serving"
    # card. The same numbers the public display board shows; no ticket codes.
    def serving_entry(person: dict) -> dict:
        seconds = person.get("seconds_at_counter")
        entry = {
            "counter_number": as_int(person.get("counter_number"), 0),
            "queue_number": person.get("queue_number"),
            "queue_label": person.get("queue_label"),
            "serving_seconds": seconds,
            "expected_remaining_seconds": None,
            "running_long": False,
        }
        if seconds is not None:
            entry["expected_remaining_seconds"] = int(round(model.remaining(seconds / 60) * 60))
            entry["running_long"] = seconds > model.mean_min * 60
        return entry

    now_serving = sorted(
        (
            serving_entry(person)
            for person in active_queue
            if person.get("counter_number") is not None and person.get("queue_number") is not None
        ),
        key=lambda entry: entry["counter_number"],
    )

    return {
        "queue_length": queue_count,
        "active_counters": counters,
        "now_serving": now_serving,
        "avg_service_time_min": round(model.mean_min, 2),
        **average_service_fields(model),
        "service_time_source": "live" if model.timed else "usual pace",
        "computed_at": school_iso(line.now),
        "windows_calling": line.windows_calling,
        "line_moving": line.moving,
        "arrival_rate_per_min": as_float(metrics["arrival_rate"], 0.0),
        "system_utilization": as_float(metrics["system_utilization"], 0.0),
        "data_age_seconds": round(data_age_seconds, 1),
        "data_status": "stale" if data_age_seconds > 15 else "live",
        "wait_estimator": estimator_version(),
        "counters_closed_until": school_iso(closed_until),
        "new_arrival": {
            **new_arrival,
            "estimated_wait_time": new_arrival_wait,
            "estimated_wait_time_minutes": new_arrival_wait,
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

# QUEUE RECORDS (the staff dashboard's Queue Records page)

RECORD_STATUS_LABELS = {"served": "Served", "no_show": "Skipped", "expired": "Expired"}


def school_day_bounds_utc(day) -> tuple[datetime, datetime]:
    """A school day as the UTC times the database stamps with NOW(). The
    service_date column holds the database's own (UTC) date, so a ticket
    taken before 8 AM in Naga carries the day before; the join time does
    not."""
    start = datetime(day.year, day.month, day.day) - timedelta(hours=SCHOOL_UTC_OFFSET_HOURS)
    return start, start + timedelta(days=1)


def _number(value):
    return None if value is None else float(value)


def queue_record_row(record: dict, is_today: bool) -> dict:
    """One ticket as the Queue Records page shows it, on the school's clock."""
    created = _as_datetime(record.get("created_at"))
    called = _as_datetime(record.get("called_at"))
    finished = _as_datetime(record.get("served_at"))
    status = str(record.get("status") or "waiting")
    queue_number = as_int(record.get("queue_number"), 0)

    wait_seconds = (
        int((called - created).total_seconds())
        if called and created and called >= created else None
    )
    service_seconds = (
        int((finished - called).total_seconds())
        if status == "served" and called and finished and finished >= called else None
    )
    if status == "waiting":
        # A ticket from an earlier day still marked waiting was never called
        # or finished (most often the line was lost in a restart or reset).
        status_label = "Waiting" if is_today else "Not finished"
    else:
        status_label = RECORD_STATUS_LABELS.get(status, status.replace("_", " ").title())
    student_name = record.get("student_name")
    school_id = record.get("student_school_id")
    estimate = _number(record.get("predicted_wait_min"))
    return {
        "id": record.get("id"),
        "queue_number": queue_number,
        "queue_label": f"Q{queue_number:03d}",
        "status": status,
        "status_label": status_label,
        "joined_at": db_clock(created) if created else None,
        "joined_at_iso": db_on_school_clock(created).isoformat() if created else None,
        "called_at": db_clock(called) if called else None,
        "finished_at": db_clock(finished) if finished and status != "waiting" else None,
        "wait_seconds": wait_seconds,
        "service_seconds": service_seconds,
        "cashier_window": record.get("cashier_window"),
        "payment_type": record.get("payment_type"),
        "payment_amount": _number(record.get("payment_amount")),
        "payment_note": record.get("payment_note"),
        "student_name": student_name,
        "student_school_id": school_id,
        "walk_in": bool(record.get("is_walkin")) and school_id is None,
        "linked_via": record.get("linked_via"),
        "estimated_wait_seconds": int(round(estimate * 60)) if estimate is not None else None,
        "people_ahead_at_issue": record.get("people_ahead_at_issue"),
        "served_by": record.get("served_by"),
    }


def queue_records_page(
    day: str | None = None,
    status: str | None = None,
    search: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> dict | None:
    """A school day's tickets, newest first, a page at a time, with the day's
    totals. None when the database is unavailable."""
    today = datetime.now(SCHOOL_TZ).date()
    try:
        chosen = datetime.strptime(day, "%Y-%m-%d").date() if day else today
    except ValueError:
        chosen = today
    status = status if status in ("waiting", "served", "no_show", "expired") else None
    page = max(1, as_int(page, 1))
    page_size = max(1, min(200, as_int(page_size, 50)))

    start, end = school_day_bounds_utc(chosen)
    result = fetch_queue_records(start, end, status, search, limit=page_size,
                                 offset=(page - 1) * page_size)
    if result is None:
        return None
    is_today = chosen == today
    s = result.get("summary") or {}
    avg_wait = _number(s.get("avg_wait_seconds"))
    avg_service = _number(s.get("avg_service_seconds"))
    total = as_int(result.get("total"), 0)
    return {
        "date": chosen.isoformat(),
        "is_today": is_today,
        "status": status or "all",
        "search": (search or "").strip(),
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": max(1, -(-total // page_size)),
        "summary": {
            "tickets": as_int(s.get("tickets"), 0),
            "served": as_int(s.get("served"), 0),
            "skipped": as_int(s.get("no_show"), 0),
            "waiting": as_int(s.get("waiting"), 0),
            "avg_wait_seconds": int(round(avg_wait)) if avg_wait is not None else None,
            "avg_service_seconds": int(round(avg_service)) if avg_service is not None else None,
            "served_amount": _number(s.get("served_amount")) or 0.0,
        },
        "records": [queue_record_row(r, is_today) for r in result.get("rows") or []],
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
