import cv2
import numpy as np
import secrets
import statistics
import threading
from collections import OrderedDict
from datetime import datetime, timedelta

from services import face_service
from core.config import FACE_ONLY_QUEUE_NUMBER_START, JOIN_INTENT_TIMEOUT_MINUTES


def queue_label(queue_number: int | None) -> str:
    """Display label for a queue number. Most numbers are real, printed
    kiosk numbers (or a staff-entered walk-in number); a registered student
    recognized by face alone gets one minted directly by the system, in a
    numeric range well above any kiosk ticket so the two can never collide."""
    if queue_number is None:
        return 'PENDING'
    return f"Q{queue_number:03d}"


class QueuePerson:
    __slots__ = (
        'queue_number', 'track_id', 'bbox', 'entered_at', 'last_seen',
        'went_missing_at', 'status', 'missing_frames', 'position_in_line',
        'access_token', 'short_code', 'pdf_path',
        'face_embedding',
        'on_the_way', 'on_the_way_at', 'is_manual', 'counter_number',
        'dedup_immune_frames',
        'student_id', 'is_walkin', 'linked_via', 'identity_status', 'linked_at',
        # Evaluation instrumentation (recognition_metrics) — first_seen_at is
        # the very first frame this person appeared in, which is earlier than
        # entered_at (the moment presence was *confirmed*).
        'first_seen_at', 'embed_attempts', 'embed_successes',
    )

    def __init__(self, track_id: int, bbox: tuple, queue_number: int | None = None):
        self.queue_number         = queue_number
        self.track_id             = track_id
        self.bbox                 = bbox
        self.entered_at           = datetime.now()
        self.last_seen            = datetime.now()
        self.went_missing_at      = None
        self.status               = 'waiting'
        self.missing_frames       = 0
        self.position_in_line     = 0
        self.access_token         = secrets.token_urlsafe(8)
        self.short_code           = None
        self.pdf_path             = None
        self.face_embedding       = None
        self.on_the_way           = False
        self.on_the_way_at        = None
        self.is_manual            = False
        self.counter_number       = None
        self.dedup_immune_frames  = 0
        self.student_id           = None
        self.is_walkin            = True
        self.linked_via           = None
        self.identity_status      = 'linked' if queue_number is not None else 'pending_link'
        self.linked_at            = datetime.now() if queue_number is not None else None
        self.first_seen_at        = None
        self.embed_attempts       = 0
        self.embed_successes      = 0

    def link_queue_number(self, queue_number: int, student_id: int | None = None,
                           linked_via: str = 'manual') -> None:
        """Attach a queue number to this presence-confirmed person — either
        an already-printed kiosk number (face+OCR match, or staff manual
        entry for a walk-in), or one the system minted directly for a
        recognized registered student (face-only match, no kiosk ticket)."""
        self.queue_number    = queue_number
        self.status          = 'waiting'
        self.student_id      = student_id
        self.is_walkin       = student_id is None
        self.linked_via      = linked_via
        self.identity_status = 'linked'
        self.linked_at       = datetime.now()

    @property
    def wait_duration(self) -> timedelta:
        return datetime.now() - self.entered_at

    @property
    def wait_time_seconds(self) -> int:
        return int(self.wait_duration.total_seconds())

    @property
    def wait_time_str(self) -> str:
        t = self.wait_time_seconds
        h, m, s = t // 3600, (t % 3600) // 60, t % 60
        if h: return f"{h}h {m}m"
        if m: return f"{m}m {s}s"
        return f"{s}s"

    @property
    def seconds_missing(self) -> float:
        if self.went_missing_at is None:
            return self.missing_frames / 15.0 if self.missing_frames else 0.0
        return (datetime.now() - self.went_missing_at).total_seconds()

    @property
    def joined_at_str(self) -> str:
        return self.entered_at.strftime("%I:%M:%S %p")

    @property
    def joined_at_full(self) -> str:
        return self.entered_at.strftime("%b %d, %Y %I:%M:%S %p")

    def to_dict(self):
        return {
            'queue_number':      self.queue_number,
            'queue_label':       queue_label(self.queue_number),
            'track_id':          self.track_id,
            'status':            self.status,
            'identity_status':   self.identity_status,
            'position_in_line':  self.position_in_line,
            'wait_time':         self.wait_time_str,
            'wait_time_seconds': self.wait_time_seconds,
            'joined_at':         self.joined_at_str,
            'joined_at_full':    self.joined_at_full,
            'joined_at_iso':     self.entered_at.isoformat(),
            'bbox':              self.bbox,
            'access_token':      self.short_code if self.short_code else self.access_token,
            'on_the_way':        self.on_the_way,
            'on_the_way_at':     self.on_the_way_at.isoformat() if self.on_the_way_at else None,
            'on_the_way_at_display': self.on_the_way_at.strftime("%I:%M:%S %p") if self.on_the_way_at else None,
            'is_manual':         self.is_manual,
            'is_walkin':         self.is_walkin,
            'linked_via':        self.linked_via,
            'has_face_embedding': self.face_embedding is not None,
            'counter_number':    self.counter_number,
        }


class QueueZone:
    __slots__ = ('x1', 'y1', 'x2', 'y2')

    def __init__(self, x1=100, y1=50, x2=540, y2=430):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def set_zone(self, x1, y1, x2, y2):
        self.x1, self.y1 = min(x1, x2), min(y1, y2)
        self.x2, self.y2 = max(x1, x2), max(y1, y2)

    def is_person_inside(self, bbox: tuple) -> bool:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) >> 1, (y1 + y2) >> 1
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2


class QueueTracker:
    MAX_MISSING_FRAMES            = 300
    MIN_CONFIRM_FRAMES            = 15
    DONE_COOLDOWN_FRAMES          = 150
    NOSHOW_WINDOW_SECONDS         = 60
    # NOTE: every value in this block is overwritten at startup by
    # queue_service.wire_callbacks() from core/config.py, so config is the
    # real source of truth. They are kept in sync anyway — a stale default
    # here reads as the deployed value to anyone opening this file, and would
    # become the actual value in any path that skips wire_callbacks.
    FACE_MATCH_THRESHOLD          = 0.30  # calibrated; see FACE_RECOGNITION_CALIBRATION.md
    FACE_MARGIN_THRESHOLD         = 0.15  # raised from 0.10; see ACCURACY.md §A1
    PENDING_LINK_TIMEOUT_SECONDS  = 45
    MIN_MOTION_PIXELS             = 8
    MOTION_HISTORY_LEN            = 20
    STATIC_STDEV_THRESHOLD        = 1.5
    STATIC_CONF_BYPASS_THRESHOLD  = 0.70
    MIN_PORTRAIT_ASPECT           = 0.50
    DEDUP_IOU_THRESH              = 0.15
    DEDUP_CENTRE_FRAC             = 0.55
    BBOX_SMOOTH_ALPHA             = 0.45

    def __init__(self, zone: QueueZone = None):
        self.zone                        = zone or QueueZone()
        self.active_queue: OrderedDict[int, QueuePerson] = OrderedDict()
        self._candidates: dict           = {}
        # Kiosk numbers already linked today — guards against double-linking
        # the same printed ticket to two different people.
        self._used_numbers: set          = set()
        self._done_cooldowns: list       = []
        self.completed_queue: list       = []
        self.total_served                = 0
        self._noshow_timers: dict        = {}
        self.appearance_rejections: list = []
        self.on_way_notifications: list  = []
        # Fired once a printed kiosk number is linked to a present person —
        # via student face+ticket-OCR match, or staff manual entry.
        self.on_number_linked             = None
        # Fired when a person is confirmed present in the zone but not yet
        # linked to a kiosk number (Algorithm 2's new terminal state).
        self.on_presence_confirmed        = None
        self.on_noshow                    = None
        self._claimed_this_frame: dict    = {}
        self._lock                        = threading.RLock()
        self._num_counters                = 3
        self._announced_numbers: set      = set()
        self._newly_called: list          = []
        self._by_queue_number: dict       = {}   # queue_number -> track_id
        self._by_student_id: dict         = {}   # student_id -> track_id
        # Next number to hand out via mint_face_only_number() — a dedicated
        # range so a system-minted number never collides with a kiosk ticket.
        self._next_face_only_number       = FACE_ONLY_QUEUE_NUMBER_START
        # Students who have tapped "join the queue" in the app and are
        # therefore eligible to be issued a number on recognition
        # (student_id -> armed_at). Recognition alone is NOT consent to be
        # queued; without an entry here a recognized student is ignored.
        self._join_intents: dict          = {}


    # LOOKUP BY QUEUE NUMBER (O(1) via secondary index)

    def _person_by_queue_number(self, queue_number: int) -> QueuePerson | None:
        tid = self._by_queue_number.get(queue_number)
        if tid is None:
            return None
        return self.active_queue.get(tid)

    def get_person_by_student_id(self, student_id: int) -> QueuePerson | None:
        """Public lookup for the student-facing 'my queue status' API — finds
        a student's own active entry without them needing to know their
        queue number/token (relevant for face-only links, which have no
        printed ticket to read those off of). A stale _by_student_id entry
        pointing at an already-removed track_id safely resolves to None via
        the dict.get() below, so no separate cleanup is required here."""
        tid = self._by_student_id.get(student_id)
        if tid is None:
            return None
        return self.active_queue.get(tid)


    # JOIN INTENT — explicit consent to be issued a number on recognition

    def arm_join_intent(self, student_id: int) -> datetime:
        """Student tapped "join the queue" in the app. Tapping again just
        refreshes the timestamp.

        Also revives this student if the camera already wrote them off as a
        bystander. That happens whenever someone is recognized *before* they
        tap join — walking up to the camera first and deciding after, which
        is the natural order. Without this the bystander mark would stand
        for as long as the tracker held that track, and the number would
        never be issued no matter how many times they tapped.
        """
        now = datetime.now()
        self._join_intents[student_id] = now

        for person in self.active_queue.values():
            if (person.identity_status == 'bystander'
                    and person.student_id == student_id):
                person.identity_status = 'pending_link'
                print(f"🔓 Student {student_id} joined — re-evaluating "
                      f"track {person.track_id}")
        return now

    def clear_join_intent(self, student_id: int) -> None:
        self._join_intents.pop(student_id, None)

    def has_join_intent(self, student_id: int) -> bool:
        """True only if this student armed themselves and the intent has not
        lapsed. Expiry is checked on read rather than swept on a timer, so a
        stale entry can never authorise a ticket even if no sweep has run."""
        armed_at = self._join_intents.get(student_id)
        if armed_at is None:
            return False
        if datetime.now() - armed_at > timedelta(minutes=JOIN_INTENT_TIMEOUT_MINUTES):
            self._join_intents.pop(student_id, None)
            return False
        return True

    def get_join_intent(self, student_id: int) -> datetime | None:
        """Armed-at time if the intent is still valid, else None."""
        return self._join_intents.get(student_id) if self.has_join_intent(student_id) else None

    def set_short_code(self, queue_number: int, short_code: str) -> bool:
        p = self._person_by_queue_number(queue_number)
        if p is None:
            print(f"[QueueTracker] ⚠️  set_short_code: Q{queue_number:03d} not found in active queue")
            return False
        p.short_code = short_code
        print(f"[QueueTracker] 🔑 Q{queue_number:03d} short_code set: {short_code}")
        return True

    def set_pdf_path(self, queue_number: int, pdf_path: str) -> bool:
        import os as _os
        p = self._person_by_queue_number(queue_number)
        if p is None:
            print(f"[QueueTracker] ⚠️  set_pdf_path: Q{queue_number:03d} not found")
            return False
        p.pdf_path = pdf_path
        print(f"[QueueTracker] 📄 Q{queue_number:03d} pdf_path=" + _os.path.basename(pdf_path))
        return True

    def get_position(self, queue_number: int) -> int:
        p = self._person_by_queue_number(queue_number)
        return p.position_in_line if p else 0


    # QUEUE-NUMBER LINKING (Algorithm 2's terminal step)

    def link_queue_number(self, track_id: int, queue_number: int,
                           student_id: int | None = None,
                           linked_via: str = 'manual') -> QueuePerson | None:
        """Attach a queue number to a presence-confirmed, not-yet-linked
        person. Called with an already-printed kiosk number (student
        face+OCR match, or staff manual entry for a walk-in), or with a
        system-minted number from mint_face_only_number() below. Together
        with force_new_person() (the walk-in path), this is the only place a
        QueuePerson ever gets a queue_number."""
        person = self.active_queue.get(track_id)
        if person is None or person.identity_status == 'linked':
            return None
        if queue_number in self._used_numbers:
            print(f"⚠️  Q{queue_number:03d} already linked today — rejecting duplicate link")
            return None

        person.link_queue_number(queue_number, student_id=student_id, linked_via=linked_via)
        self._used_numbers.add(queue_number)
        self._by_queue_number[queue_number] = track_id
        if student_id is not None:
            self._by_student_id[student_id] = track_id
            # Intent is spent once it produces a number — the student must
            # opt in again for their next visit.
            self.clear_join_intent(student_id)

        self._recalculate_positions()
        if self.on_number_linked:
            try:
                self.on_number_linked(queue_number, person.wait_time_str, person.joined_at_str,
                                       person.access_token, student_id, linked_via,
                                       person.is_walkin)
            except Exception as e:
                print(f"⚠️  on_number_linked callback error: {e}")
        print(f"🔗 Q{queue_number:03d} linked (track_id={track_id}, via={linked_via})")
        return person

    def mint_face_only_number(self, track_id: int, student_id: int) -> QueuePerson | None:
        """Per SOP #2: a registered student recognized at the queue-zone
        camera gets a queue number generated directly by the system — no
        kiosk ticket, no staff involvement. This never touches the kiosk's
        own numbering; it draws from a separate, non-colliding range
        (FACE_ONLY_QUEUE_NUMBER_START and up). Walk-ins are unaffected and
        keep going through link_queue_number()/force_new_person() exactly
        as before."""
        while self._next_face_only_number in self._used_numbers:
            self._next_face_only_number += 1
        queue_number = self._next_face_only_number
        self._next_face_only_number += 1
        return self.link_queue_number(
            track_id, queue_number, student_id=student_id, linked_via='face_only',
        )


    # FACE-BASED RE-ENTRY MATCHING (Algorithm 4 — replaces the retired
    # HSV appearance/twin-guard/blacklist/spatial-fallback chain)

    def _get_missing_face_candidates(self, current_track_ids: set) -> list:
        """Missing entries (pending or already-linked) with a stored face
        embedding — eligible for face-based re-entry. Walk-ins never appear
        here: they're is_manual and never transition to missing_frames > 0."""
        claimed = self._claimed_this_frame
        return [
            (tid, p) for tid, p in self.active_queue.items()
            if tid not in current_track_ids
            and p.status != 'done_pending'
            and tid not in claimed
            and p.missing_frames > 0
            and p.face_embedding is not None
        ]

    def _find_returning_student(self, live_embedding, current_track_ids):
        """Face-based re-identification. A face embedding is discriminative
        enough on its own that the old tiered recency/spatial-fallback/twin-
        guard logic (needed to compensate for weak colour histograms) is no
        longer necessary — one threshold+margin check replaces all of it.
        Never guesses: an ambiguous match returns no match, same philosophy
        as before (see face_service.match_student)."""
        if live_embedding is None:
            return None, None, 0.0

        missing = self._get_missing_face_candidates(current_track_ids)
        if not missing:
            return None, None, 0.0

        candidates = [(tid, p.face_embedding) for tid, p in missing]
        result = face_service.match_student(
            live_embedding, candidates,
            match_threshold=self.FACE_MATCH_THRESHOLD,
            margin_threshold=self.FACE_MARGIN_THRESHOLD,
        )
        if not result.accepted:
            return None, None, 0.0

        matched_tid = result.student_id   # generic field name; here it's the track_id key we passed in
        matched_person = self.active_queue.get(matched_tid)
        return matched_tid, matched_person, result.score

    def _restore_missing_person(self, ret_person, ret_tid, track_id, bbox, live_embedding):
        ret_person.track_id            = track_id
        ret_person.bbox                = bbox
        ret_person.missing_frames      = 0
        ret_person.status              = 'waiting'
        ret_person.went_missing_at     = None
        ret_person.dedup_immune_frames = 60
        self._noshow_timers.pop(ret_person.queue_number, None)
        if live_embedding is not None:
            ret_person.face_embedding = live_embedding
        if ret_tid != track_id:
            self.active_queue[track_id] = ret_person
            del self.active_queue[ret_tid]
            if ret_person.queue_number is not None:
                self._by_queue_number[ret_person.queue_number] = track_id
            if ret_person.student_id is not None:
                self._by_student_id[ret_person.student_id] = track_id
        self._candidates.pop(track_id, None)
        self._claimed_this_frame[ret_tid] = track_id


    # NO-SHOW HANDLING

    def _check_noshow(self):
        now     = datetime.now()
        to_bump = []
        win     = self.NOSHOW_WINDOW_SECONDS

        for tid, p in self.active_queue.items():
            qn = p.queue_number
            if p.position_in_line == 1 and p.status == 'missing':
                if qn not in self._noshow_timers:
                    self._noshow_timers[qn] = now
                    print(f"⏳ Q{qn:03d} is #1 but absent — {win}s countdown")
                elif (now - self._noshow_timers[qn]).total_seconds() >= win:
                    to_bump.append((tid, p))
            else:
                self._noshow_timers.pop(qn, None)

        for tid, p in to_bump:
            qn = p.queue_number
            print(f"🚫 Q{qn:03d} NO-SHOW bumped")
            self._noshow_timers.pop(qn, None)

            if p.pdf_path:
                try:
                    from services.ticket_printer import delete_ticket
                    delete_ticket(p.pdf_path)
                except Exception as e:
                    print(f"[QueueTracker] ⚠️  PDF delete error (no-show) Q{qn:03d}: {e}")

            completed = p.to_dict()
            completed.update({
                'completed_at':      now.strftime("%I:%M:%S %p"),
                'completed_at_full': now.strftime("%b %d, %Y %I:%M:%S %p"),
                'total_wait_time':   p.wait_time_str,
                'bump_reason':       'no_show',
            })
            with self._lock:
                self.completed_queue.append(completed)
                self.total_served += 1
            p.status         = 'done_pending'
            p.missing_frames = self.MAX_MISSING_FRAMES + 1
            self._done_cooldowns.append({'bbox': p.bbox, 'frames_left': self.DONE_COOLDOWN_FRAMES})

            if self.on_noshow:
                try:
                    self.on_noshow(qn)
                except Exception as e:
                    print(f"⚠️  on_noshow callback error Q{qn:03d}: {e}")

    def get_noshow_alerts(self) -> list:
        now = datetime.now()
        alerts = []
        win = self.NOSHOW_WINDOW_SECONDS
        for p in self.active_queue.values():
            qn = p.queue_number
            if qn in self._noshow_timers:
                elapsed   = (now - self._noshow_timers[qn]).total_seconds()
                remaining = max(0, win - elapsed)
                alerts.append({
                    'queue_number':      queue_label(qn),
                    'queue_number_int':  qn,
                    'seconds_remaining': int(remaining),
                    'status': 'critical' if remaining <= 15 else 'warning',
                })
        return alerts

    def get_pending_link_alerts(self) -> list:
        """Staff alert list for people confirmed present but still
        unlinked past PENDING_LINK_TIMEOUT_SECONDS — the concrete
        embodiment of "ambiguous → escalate to a human, don't guess.\""""
        now = datetime.now()
        alerts = []
        for tid, p in self.active_queue.items():
            if p.identity_status != 'pending_link':
                continue
            elapsed = (now - p.entered_at).total_seconds()
            if elapsed >= self.PENDING_LINK_TIMEOUT_SECONDS:
                alerts.append({
                    'track_id':           tid,
                    'seconds_waiting':    int(elapsed),
                    'has_face_embedding': p.face_embedding is not None,
                })
        return alerts


    # INTERNAL HELPERS

    @staticmethod
    def _iou(b1, b2) -> float:
        ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if not inter:
            return 0.0
        return inter / ((b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter)

    def _is_in_done_cooldown(self, bbox, threshold=0.35) -> bool:
        return any(self._iou(e['bbox'], bbox) > threshold for e in self._done_cooldowns)

    def _tick_done_cooldowns(self):
        self._done_cooldowns = [
            {**e, 'frames_left': e['frames_left'] - 1}
            for e in self._done_cooldowns if e['frames_left'] > 1
        ]
        for p in self.active_queue.values():
            if p.dedup_immune_frames > 0:
                p.dedup_immune_frames -= 1

    @staticmethod
    def _bbox_centre(bbox) -> tuple:
        return ((bbox[0] + bbox[2]) >> 1, (bbox[1] + bbox[3]) >> 1)

    @staticmethod
    def _bbox_diagonal(bbox) -> float:
        return ((bbox[2]-bbox[0])**2 + (bbox[3]-bbox[1])**2) ** 0.5

    def _is_duplicate_of(self, bbox_a, bbox_b) -> bool:
        # High IOU: two YOLO detections for the same physical person
        if self._iou(bbox_a, bbox_b) > self.DEDUP_IOU_THRESH:
            return True
        # Nearly identical centres: same person, slightly jittered bbox
        ca = self._bbox_centre(bbox_a)
        cb = self._bbox_centre(bbox_b)
        centre_dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
        avg_diag = max(
            1.0,
            (self._bbox_diagonal(bbox_a) + self._bbox_diagonal(bbox_b)) / 2,
        )
        return centre_dist < self.DEDUP_CENTRE_FRAC * avg_diag

    def _find_overlapping_candidate(self, bbox):
        for tid, info in self._candidates.items():
            if self._is_duplicate_of(info['bbox'], bbox):
                return tid
        return None

    def _is_duplicate_of_active(self, bbox) -> bool:
        for p in self.active_queue.values():
            if p.status == 'done_pending':
                continue
            if p.missing_frames > 0:
                continue
            if self._is_duplicate_of(p.bbox, bbox):
                return True
        return False

    def _has_sufficient_motion(self, centers: list, avg_conf: float = 0.0,
                                has_face_embedding: bool = False) -> bool:
        if avg_conf >= self.STATIC_CONF_BYPASS_THRESHOLD:
            return True

        # A confidently-extracted live face embedding is stronger evidence of
        # a real person than pixel jitter ever was — this check exists to
        # reject a static poster/photo held up to the camera, but a student
        # standing still to be recognized (the actual desired behavior now)
        # would otherwise get bounced by the same heuristic.
        if has_face_embedding:
            return True

        if len(centers) < 8:
            return True

        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]

        movement = max(max(xs) - min(xs), max(ys) - min(ys))
        if movement < self.MIN_MOTION_PIXELS:
            print(f"🖼️  Static rejection — range={movement}px "
                  f"(need >{self.MIN_MOTION_PIXELS}px) — likely a picture/object")
            return False

        if len(centers) >= 10:
            x_stdev = statistics.stdev(xs)
            y_stdev = statistics.stdev(ys)
            thresh  = self.STATIC_STDEV_THRESHOLD
            if x_stdev < thresh and y_stdev < thresh:
                print(f"🖼️  Static rejection — stdev=({x_stdev:.2f}, {y_stdev:.2f}) "
                      f"both < {thresh}px — likely a picture/object")
                return False

        return True

    def _is_plausible_person_bbox(self, bbox: tuple) -> bool:
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        aspect = h / w
        if aspect < self.MIN_PORTRAIT_ASPECT:
            print(f"🖼️  Aspect-ratio rejection — h/w={aspect:.2f} "
                  f"(need ≥{self.MIN_PORTRAIT_ASPECT}) — likely not a person")
            return False
        return True

    def _make_completed_entry(self, p: QueuePerson, bump_reason: str) -> dict:
        now = datetime.now()
        entry = p.to_dict()
        entry.update({
            'completed_at':      now.strftime("%I:%M:%S %p"),
            'completed_at_full': now.strftime("%b %d, %Y %I:%M:%S %p"),
            'total_wait_time':   p.wait_time_str,
            'bump_reason':       bump_reason,
        })
        return entry


    # MAIN FRAME PROCESSOR

    def process_frame(self, tracked_persons: list) -> dict:
        self._tick_done_cooldowns()
        self._claimed_this_frame = {}
        current_in_zone = set()

        for person in tracked_persons:
            track_id = person['track_id']
            bbox     = person['bbox']
            in_zone  = self.zone.is_person_inside(bbox)

            if track_id in self.active_queue:
                p                = self.active_queue[track_id]
                a = self.BBOX_SMOOTH_ALPHA
                p.bbox           = (
                    int(a * bbox[0] + (1 - a) * p.bbox[0]),
                    int(a * bbox[1] + (1 - a) * p.bbox[1]),
                    int(a * bbox[2] + (1 - a) * p.bbox[2]),
                    int(a * bbox[3] + (1 - a) * p.bbox[3]),
                )
                p.last_seen      = datetime.now()
                if p.missing_frames > 0:
                    p.dedup_immune_frames = 60
                p.missing_frames = 0
                live_embedding = person.get('face_embedding')
                # Keep counting while still pending — the frames spent waiting
                # for a usable face are exactly what seconds_to_link measures.
                if p.identity_status == 'pending_link':
                    p.embed_attempts += 1
                    if live_embedding is not None:
                        p.embed_successes += 1
                if live_embedding is not None:
                    p.face_embedding = np.array(live_embedding, dtype=np.float32)
                if in_zone:
                    current_in_zone.add(track_id)
                    if p.status in ('missing', 'waiting') and p.went_missing_at is not None:
                        p.status          = 'waiting'
                        p.went_missing_at = None
                        self._noshow_timers.pop(p.queue_number, None)
                        label = f"Q{p.queue_number:03d}" if p.queue_number is not None else "(pending)"
                        print(f"✅ {label} back in zone (same track_id)")
                continue

            if not in_zone:
                print(f"[QueueTracker] DEBUG tid={track_id} bbox={bbox} OUTSIDE zone "
                      f"({self.zone.x1},{self.zone.y1})-({self.zone.x2},{self.zone.y2})")
                self._candidates.pop(track_id, None)
                continue

            if not self._is_plausible_person_bbox(bbox):
                self._candidates.pop(track_id, None)
                continue

            current_in_zone.add(track_id)
            _raw_embed = person.get('face_embedding')
            live_embedding = np.array(_raw_embed, dtype=np.float32) if _raw_embed is not None else None

            # First-sight re-entry check
            if live_embedding is not None:
                ret_tid, ret_person, ret_score = self._find_returning_student(
                    live_embedding, current_in_zone)
                if ret_person is not None:
                    label = f"Q{ret_person.queue_number:03d}" if ret_person.queue_number is not None else "(pending)"
                    print(f"✅ {label} re-entry (score={ret_score:.2f}, track {ret_tid}→{track_id})")
                    self._restore_missing_person(ret_person, ret_tid, track_id, bbox, live_embedding)
                    continue

            if self._is_in_done_cooldown(bbox):
                self._candidates.pop(track_id, None)
                continue

            if self._is_duplicate_of_active(bbox):
                self._candidates.pop(track_id, None)
                print(f"🔁 track {track_id} suppressed — duplicate of active person")
                continue

            cand_tid = self._find_overlapping_candidate(bbox)
            if cand_tid is not None and cand_tid != track_id:
                info = self._candidates.pop(cand_tid)
                info['bbox'] = bbox
                info['count'] = info.get('count', 0) + 1
                info.setdefault('first_seen', datetime.now())
                info['embed_attempts'] = info.get('embed_attempts', 0) + 1
                if live_embedding is not None:
                    info['embed_successes'] = info.get('embed_successes', 0) + 1
                    info.setdefault('face_embeds', []).append(live_embedding)
                info.setdefault('confs', []).append(float(person.get('conf', 0.0)))
                cx, cy = (bbox[0] + bbox[2]) >> 1, (bbox[1] + bbox[3]) >> 1
                info.setdefault('centers', []).append((cx, cy))
                if len(info['centers']) > self.MOTION_HISTORY_LEN:
                    info['centers'].pop(0)
                if len(info['confs']) > self.MOTION_HISTORY_LEN:
                    info['confs'].pop(0)
                if len(info.get('face_embeds', [])) > 8:
                    info['face_embeds'].pop(0)
                self._candidates[track_id] = info
                cand = info
            else:
                cand = self._candidates.setdefault(track_id, {
                    'count': 0, 'bbox': bbox, 'face_embeds': [], 'centers': [],
                    'confs': [], 'first_seen': datetime.now(),
                    'embed_attempts': 0, 'embed_successes': 0,
                })
                cx, cy = (bbox[0] + bbox[2]) >> 1, (bbox[1] + bbox[3]) >> 1
                cand['count'] += 1
                cand['bbox']   = bbox
                # Live face-detection rate: every frame the detector looked at
                # this person is an attempt; a returned embedding is a success.
                cand['embed_attempts'] = cand.get('embed_attempts', 0) + 1
                if live_embedding is not None:
                    cand['embed_successes'] = cand.get('embed_successes', 0) + 1
                centers = cand['centers']
                centers.append((cx, cy))
                if len(centers) > self.MOTION_HISTORY_LEN:
                    centers.pop(0)
                confs = cand['confs']
                confs.append(float(person.get('conf', 0.0)))
                if len(confs) > self.MOTION_HISTORY_LEN:
                    confs.pop(0)
                if live_embedding is not None:
                    embeds = cand['face_embeds']
                    embeds.append(live_embedding)
                    if len(embeds) > 8:
                        embeds.pop(0)

            # Mid-accumulation re-entry check
            if live_embedding is not None:
                early_tid, early_person, early_score = self._find_returning_student(
                    live_embedding, current_in_zone)
                if early_person is not None:
                    label = f"Q{early_person.queue_number:03d}" if early_person.queue_number is not None else "(pending)"
                    print(f"✅ {label} re-entry (accumulation frame {cand['count']}, score={early_score:.2f})")
                    self._restore_missing_person(early_person, early_tid, track_id, bbox, live_embedding)
                    continue

            if cand['count'] < self.MIN_CONFIRM_FRAMES:
                if cand['count'] % 5 == 0:
                    print(f"[QueueTracker] DEBUG tid={track_id} accumulating "
                          f"count={cand['count']}/{self.MIN_CONFIRM_FRAMES}")
                continue

            avg_conf = (
                sum(cand.get('confs', [])) / max(1, len(cand.get('confs', [])))
            )
            has_face = bool(cand.get('face_embeds'))
            if not self._has_sufficient_motion(cand.get('centers', []), avg_conf, has_face):
                self._candidates.pop(track_id)
                continue

            # Final gate re-entry check
            final_tid, final_person, final_score = self._find_returning_student(
                live_embedding, current_in_zone)
            if final_person is not None:
                label = f"Q{final_person.queue_number:03d}" if final_person.queue_number is not None else "(pending)"
                print(f"✅ {label} final gate (score={final_score:.2f}) — not new")
                self._restore_missing_person(final_person, final_tid, track_id, bbox, live_embedding)
                continue

            # Presence confirmed (Algorithm 2's terminal state). No queue
            # number is minted here — it is only ever set later by
            # link_queue_number() (student face match, or a kiosk number via
            # OCR) or force_new_person() (staff manual walk-in entry).
            new_p = QueuePerson(track_id=track_id, bbox=bbox)

            # Carry the accumulation-phase instrumentation onto the person so
            # the eventual link can report first-sighting-to-linked timing.
            new_p.first_seen_at   = cand.get('first_seen', new_p.entered_at)
            new_p.embed_attempts  = cand.get('embed_attempts', 0)
            new_p.embed_successes = cand.get('embed_successes', 0)

            cand_embeds = cand.get('face_embeds', [])
            if cand_embeds:
                new_p.face_embedding = np.mean(np.stack(cand_embeds), axis=0)

            self._candidates.pop(track_id)
            self.active_queue[track_id] = new_p
            print(f"👁️  Presence confirmed (track_id={track_id}) — pending kiosk-number link")

            if self.on_presence_confirmed:
                try:
                    self.on_presence_confirmed(track_id)
                except Exception as e:
                    print(f"⚠️  on_presence_confirmed callback error: {e}")

        # Absent person handling
        to_remove = []
        for tid, p in self.active_queue.items():
            if tid in current_in_zone:
                continue
            if p.is_manual:
                continue  # manual (walk-in) entries never auto-expire
            p.missing_frames += 1

            if p.status == 'done_pending':
                if p.missing_frames > 10:
                    if p.queue_number is not None:
                        self._by_queue_number.pop(p.queue_number, None)
                    if p.student_id is not None:
                        self._by_student_id.pop(p.student_id, None)
                    to_remove.append(tid)
                    for entry in self._done_cooldowns:
                        if self._iou(entry['bbox'], p.bbox) > 0.3:
                            entry['bbox'] = p.bbox
                            break
                    label = f"Q{p.queue_number:03d}" if p.queue_number is not None else "(unlinked)"
                    print(f"👋 {label} left frame — freed")

            elif p.status == 'waiting':
                if p.went_missing_at is None:
                    p.went_missing_at = datetime.now()
                if p.missing_frames > self.MAX_MISSING_FRAMES:
                    p.status = 'missing'
                    label = f"Q{p.queue_number:03d}" if p.queue_number is not None else "(unlinked)"
                    print(f"❓ {label} went missing ({p.missing_frames} frames)")

        for tid in to_remove:
            del self.active_queue[tid]

        self._dedup_active_queue()
        self._check_noshow()
        self._recalculate_positions()
        return self.get_state()


    # MARK DONE

    def mark_transaction_done(self, queue_number: int) -> bool:
        p = self._person_by_queue_number(queue_number)
        if p is None or p.status not in ('waiting', 'missing'):
            return False

        if p.pdf_path:
            try:
                from services.ticket_printer import delete_ticket
                delete_ticket(p.pdf_path)
            except Exception as e:
                print(f"[QueueTracker] ⚠️  PDF delete error for Q{queue_number:03d}: {e}")

        p.status           = 'done_pending'
        p.position_in_line = 0
        self._done_cooldowns.append(
            {'bbox': p.bbox, 'frames_left': self.DONE_COOLDOWN_FRAMES})
        with self._lock:
            self.total_served += 1
            self.completed_queue.append(self._make_completed_entry(p, 'served'))
            self._noshow_timers.pop(queue_number, None)
        self._recalculate_positions()
        print(f"✅ Q{queue_number:03d} DONE | Wait: {p.wait_time_str}")
        return True

    def force_new_person(self, queue_number: int, is_walkin: bool = True) -> dict | None:
        """Staff manual entry: link an already-printed kiosk number directly,
        bypassing face/OCR confirmation. This is the walk-in path (per the
        panel's directive, walk-ins keep their kiosk number as-is), and also
        a fallback when CV/face/OCR fails for an enrolled student. The
        created entry persists in the active queue until staff marks it
        done; it is never expired by the missing-frames timeout.
        """
        if queue_number in self._used_numbers:
            print(f"⚠️  Q{queue_number:03d} already linked today — force-new rejected")
            return None

        fake_tid = -(queue_number)
        new_p = QueuePerson(track_id=fake_tid, bbox=(0, 0, 1, 1), queue_number=queue_number)
        new_p.is_manual  = True
        new_p.is_walkin  = is_walkin
        new_p.linked_via = 'manual'
        new_p.linked_at  = datetime.now()

        self._used_numbers.add(queue_number)
        self._by_queue_number[queue_number] = fake_tid
        self.active_queue[fake_tid] = new_p
        self._recalculate_positions()

        if self.on_number_linked:
            try:
                self.on_number_linked(queue_number, new_p.wait_time_str,
                                       new_p.joined_at_str, new_p.access_token,
                                       None, 'manual', new_p.is_walkin)
            except Exception as e:
                print(f"⚠️  on_number_linked callback error (force): {e}")

        kind = "walk-in" if is_walkin else "student override"
        print(f"👤 Q{queue_number:03d} FORCE-NEW by staff ({kind})")
        return new_p.to_dict()

    def _append_on_the_way_notification(self, queue_number: int, when: datetime) -> dict:
        label = queue_label(queue_number)
        notification = {
            'id': f"{label}-{int(when.timestamp())}",
            'queue_number': queue_number,
            'queue_label': label,
            'created_at': when.isoformat(),
            'created_at_display': when.strftime("%I:%M:%S %p"),
            'message': f"{label} is on the way to the queue zone.",
        }
        if not any(item.get('id') == notification['id'] for item in self.on_way_notifications):
            self.on_way_notifications.append(notification)
            self.on_way_notifications = self.on_way_notifications[-20:]
        return notification

    def record_on_the_way_signal(self, queue_number: int) -> dict:
        notification = self._append_on_the_way_notification(queue_number, datetime.now())
        print(f"[QueueTracker] Q{queue_number:03d} on-way signal recorded")
        return notification

    def mark_on_the_way(self, queue_number: int) -> dict | None:
        p = self._person_by_queue_number(queue_number)
        if p is None or p.status not in ('waiting', 'missing'):
            return None
        if not p.on_the_way:
            p.on_the_way = True
            p.on_the_way_at = datetime.now()
            self._append_on_the_way_notification(queue_number, p.on_the_way_at)
            print(f"[QueueTracker] Q{queue_number:03d} is on the way to queue zone")
        return p.to_dict()


    # TOKEN LOOKUP

    def lookup_by_token(self, queue_number: int, token: str) -> dict | None:
        p = self._person_by_queue_number(queue_number)
        if p is None:
            return None
        expected = p.short_code if p.short_code is not None else p.access_token
        if expected != token:
            return {'error': 'invalid_token'}
        result = p.to_dict()
        qn = p.queue_number
        if qn in self._noshow_timers:
            elapsed   = (datetime.now() - self._noshow_timers[qn]).total_seconds()
            result['noshow_countdown'] = int(max(0, self.NOSHOW_WINDOW_SECONDS - elapsed))
            result['noshow_warning']   = True
        else:
            result['noshow_warning'] = False
        return result


    # HELPERS

    @staticmethod
    def _pick_dedup_survivor(p_i: QueuePerson, p_j: QueuePerson):
        """When two ghost tracks merge, keep whichever already has a real
        printed number linked; between two linked entries keep the lower
        number; between two still-pending entries keep the one seen first."""
        if p_i.queue_number is not None and p_j.queue_number is not None:
            return (p_i, p_j) if p_i.queue_number < p_j.queue_number else (p_j, p_i)
        if p_i.queue_number is not None:
            return p_i, p_j
        if p_j.queue_number is not None:
            return p_j, p_i
        return (p_i, p_j) if p_i.entered_at <= p_j.entered_at else (p_j, p_i)

    def _dedup_active_queue(self):
        tids    = list(self.active_queue.keys())
        to_drop = set()

        for i in range(len(tids)):
            if tids[i] in to_drop:
                continue
            p_i = self.active_queue[tids[i]]
            if p_i.status == 'done_pending':
                continue
            if p_i.missing_frames > 0:
                continue

            for j in range(i + 1, len(tids)):
                if tids[j] in to_drop:
                    continue
                p_j = self.active_queue[tids[j]]
                if p_j.status == 'done_pending':
                    continue
                if p_j.missing_frames > 0:
                    continue
                if p_i.dedup_immune_frames > 0 or p_j.dedup_immune_frames > 0:
                    continue

                # If both have a face embedding and look clearly different,
                # they are two distinct real people — never dedup on position alone.
                if p_i.face_embedding is not None and p_j.face_embedding is not None:
                    sim = face_service.cosine_similarity(p_i.face_embedding, p_j.face_embedding)
                    if sim < self.FACE_MATCH_THRESHOLD:
                        continue

                if self._is_duplicate_of(p_i.bbox, p_j.bbox):
                    keep, drop = self._pick_dedup_survivor(p_i, p_j)
                    drop_tid = tids[i] if drop is p_i else tids[j]
                    keep_label = f"Q{keep.queue_number:03d}" if keep.queue_number is not None else "(unlinked)"
                    drop_label = f"Q{drop.queue_number:03d}" if drop.queue_number is not None else "(unlinked)"
                    print(f"♻️  Dedup: {drop_label} is duplicate of {keep_label} — retiring ghost")
                    if drop.pdf_path:
                        try:
                            from services.ticket_printer import delete_ticket
                            delete_ticket(drop.pdf_path)
                        except Exception as e:
                            print(f"[QueueTracker] duplicate PDF delete error {drop_label}: {e}")
                    if drop.queue_number is not None:
                        self._used_numbers.discard(drop.queue_number)
                        self._by_queue_number.pop(drop.queue_number, None)
                    if drop.student_id is not None:
                        self._by_student_id.pop(drop.student_id, None)
                    self._done_cooldowns.append({
                        'bbox': drop.bbox,
                        'frames_left': min(self.DONE_COOLDOWN_FRAMES, 45),
                    })
                    to_drop.add(drop_tid)

        for tid in to_drop:
            del self.active_queue[tid]

    def _recalculate_positions(self):
        active_line = sorted(
            (p for p in self.active_queue.values()
             if p.queue_number is not None and p.status in ('waiting', 'missing')),
            key=lambda x: x.queue_number
        )

        # Recalculate positions
        for i, p in enumerate(active_line):
            p.position_in_line = i + 1

        # Keep existing counter assignments sticky — find which counters are free
        occupied = {p.counter_number for p in active_line if p.counter_number is not None}
        free_counters = sorted(
            c for c in range(1, self._num_counters + 1) if c not in occupied
        )

        # Assign free counters only to unassigned people within counter range
        newly_called = []
        free_idx = 0
        for p in active_line:
            if p.counter_number is not None:
                continue
            if p.position_in_line <= self._num_counters and free_idx < len(free_counters):
                p.counter_number = free_counters[free_idx]
                free_idx += 1
                if p.queue_number not in self._announced_numbers:
                    self._announced_numbers.add(p.queue_number)
                    newly_called.append({
                        'queue_number':   p.queue_number,
                        'queue_label':    queue_label(p.queue_number),
                        'counter_number': p.counter_number,
                    })

        self._newly_called = newly_called

    def get_state(self) -> dict:
        active = sorted(
            (p.to_dict() for p in self.active_queue.values()
             if p.queue_number is not None and p.status in ('waiting', 'missing')),
            key=lambda x: x['queue_number']
        )
        pending = [
            p.to_dict() for p in self.active_queue.values()
            if p.identity_status == 'pending_link'
        ]
        counter_assignments = [
            p for p in active if p.get('counter_number') is not None
        ]
        newly_called = list(self._newly_called)
        self._newly_called = []
        return {
            'active_queue':          active,
            'queue_count':           len(active),
            'pending_queue':         pending,
            'pending_count':         len(pending),
            'pending_link_alerts':   self.get_pending_link_alerts(),
            'total_served':          self.total_served,
            'completed':             self.completed_queue[-10:],
            'noshow_alerts':         self.get_noshow_alerts(),
            'on_way_notifications':  self.on_way_notifications[-10:],
            'appearance_rejections': self.appearance_rejections[-5:],
            'counter_assignments':   counter_assignments,
            'newly_called':          newly_called,
            'num_counters':          self._num_counters,
        }


    # DRAW ON FRAME

    def draw_on_frame(self, frame):
        h, w = frame.shape[:2]
        z    = self.zone
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        cv2.rectangle(frame, (z.x1, z.y1), (z.x2, z.y2), (0, 255, 255), 2)
        lbl = "QUEUE ZONE"
        (lw, lh), _ = cv2.getTextSize(lbl, FONT, 0.55, 1)
        lx, ly = z.x1 + 6, z.y1 + lh + 8
        cv2.rectangle(frame, (lx-2, ly-lh-4), (lx+lw+2, ly+4), (0, 0, 0), -1)
        cv2.putText(frame, lbl, (lx, ly), FONT, 0.55, (0, 255, 255), 1)

        for i, alert in enumerate(self.get_noshow_alerts()):
            color    = (0, 0, 255) if alert['status'] == 'critical' else (0, 165, 255)
            warn_txt = (f"{alert['queue_number']} NO-SHOW WARNING "
                        f"Bumping in {alert['seconds_remaining']}s")
            (aw, ah), _ = cv2.getTextSize(warn_txt, FONT, 0.5, 1)
            ay = h - 20 - i * 28
            cv2.rectangle(frame, (8, ay-ah-4), (aw+16, ay+4), (0, 0, 0), -1)
            cv2.putText(frame, warn_txt, (12, ay), FONT, 0.5, color, 1)

        for person in self.active_queue.values():
            x1, y1, x2, y2 = person.bbox
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w-1, x2); y2 = min(h-1, y2)
            label = queue_label(person.queue_number)

            if person.status == 'done_pending':
                box_color, info_text, text_color = (0,0,255), "DONE - EXIT PLEASE", (0,0,255)
                thickness = 3
            elif person.identity_status == 'pending_link':
                box_color = (255, 200, 0)
                info_text = "confirming identity..."
                text_color, thickness = (255, 200, 0), 2
            elif person.status == 'missing':
                box_color = (128,128,128)
                info_text = f"#{person.position_in_line} | {person.wait_time_str} | MISSING"
                text_color, thickness = (200,200,200), 2
            elif person.position_in_line == 1:
                box_color = (0,255,0)
                info_text = f"#1 NEXT | {person.wait_time_str}"
                text_color, thickness = (0,255,0), 2
            else:
                box_color = (0,165,255)
                info_text = f"#{person.position_in_line} | {person.wait_time_str}"
                text_color, thickness = (0,165,255), 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.7, 2)
            by1 = max(0, y1 - th - 12)
            cv2.rectangle(frame, (x1, by1), (x1+tw+10, y1), box_color, -1)
            cv2.putText(frame, label, (x1+5, y1-5), FONT, 0.7, (255,255,255), 2)

            info_y = min(h-8, y2+18)
            (iw, ih), _ = cv2.getTextSize(info_text, FONT, 0.45, 1)
            cv2.rectangle(frame, (x1, info_y-ih-3), (x1+iw+4, info_y+3), (0,0,0), -1)
            cv2.putText(frame, info_text, (x1+2, info_y), FONT, 0.45, text_color, 1)

        waiting = sum(1 for p in self.active_queue.values()
                      if p.queue_number is not None and p.status in ('waiting', 'missing'))
        summary = f"Queue: {waiting} waiting"
        (sw, sh), _ = cv2.getTextSize(summary, FONT, 0.6, 1)
        sx = w - sw - 12
        cv2.rectangle(frame, (sx-4, 4), (sx+sw+4, sh+14), (0,0,0), -1)
        cv2.putText(frame, summary, (sx, sh+10), FONT, 0.6, (0,255,255), 1)

        return frame
