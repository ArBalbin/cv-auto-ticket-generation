"""
Queue Tracker Module
====================

CORE PHILOSOPHY (v4 — pragmatic hybrid matching):
Hybrid matching strategy in priority order:
  1. Same track_id → always restore (YOLO maintained continuity)
  2. Time + zone recency → single missing person = restore by recency (no appearance needed)
  3. Appearance scoring → tiebreaker for 2+ missing persons
  4. Done blacklist → prevents served people from stealing missing slots

v4.1 — Static object / picture rejection:
  - Increased MIN_MOTION_PIXELS (8) and MIN_CONFIRM_FRAMES (15)
  - Added centre-position stdev check: rejects near-zero-variance detections
    (wall pictures, signs, static objects have stdev ≈ 0 across all frames)
  - Added portrait aspect-ratio guard: standing people are always taller than
    wide; landscape/square boxes are rejected before candidate accumulation
"""

import cv2
import numpy as np
import secrets
import statistics
from collections import OrderedDict, deque
from datetime import datetime, timedelta


class QueuePerson:
    __slots__ = (
        'queue_number', 'track_id', 'bbox', 'entered_at', 'last_seen',
        'went_missing_at', 'status', 'missing_frames', 'position_in_line',
        'access_token', 'appearance_signature', 'appearance_history'
    )

    def __init__(self, queue_number: int, track_id: int, bbox: tuple):
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
        self.appearance_signature = None
        self.appearance_history   = []

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
            'queue_label':       f"Q{self.queue_number:03d}",
            'track_id':          self.track_id,
            'status':            self.status,
            'position_in_line':  self.position_in_line,
            'wait_time':         self.wait_time_str,
            'wait_time_seconds': self.wait_time_seconds,
            'joined_at':         self.joined_at_str,
            'joined_at_full':    self.joined_at_full,
            'joined_at_iso':     self.entered_at.isoformat(),
            'bbox':              self.bbox,
            'access_token':      self.access_token,
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
    # ── Raised from 8 → 15: static objects accumulate frames effortlessly;
    #    a higher bar means a picture must be "seen" across more frames before
    #    it could ever get a queue number (combined with stdev / motion checks
    #    it will be eliminated well before this count is reached).
    MAX_MISSING_FRAMES            = 300
    MIN_CONFIRM_FRAMES            = 15   # ↑ was 8
    DONE_COOLDOWN_FRAMES          = 150
    NOSHOW_WINDOW_SECONDS         = 60
    RECENCY_WINDOW_SECONDS        = 600
    APPEARANCE_TIEBREAK_THRESHOLD = 0.35
    DONE_BLACKLIST_THRESH         = 0.70
    # ── Raised from 3 → 8: camera shake / JPEG compression can produce ~3 px
    #    of apparent movement even for a perfectly static object; 8 px is a
    #    more reliable floor for genuine human micro-movement.
    MIN_MOTION_PIXELS             = 8    # ↑ was 3
    MOTION_HISTORY_LEN            = 20
    _DONE_APP_MAX                 = 20
    _HIST_SHAPE                   = (16, 16)

    # ── NEW: stdev threshold for static-object rejection.
    #    A real person standing still still has ~2–5 px of natural micro-sway;
    #    a framed picture on a wall will have stdev < 1.5 px in both axes.
    STATIC_STDEV_THRESHOLD        = 1.5  # px — below this = likely not a person

    # ── NEW: minimum portrait aspect ratio (height / width).
    #    Standing people are always taller than wide (ratio > 1.0).
    #    We allow a slack down to 0.65 for partial / crouching detections.
    #    Landscape or square boxes (framed pictures, signs) will be < 0.65.
    MIN_PORTRAIT_ASPECT           = 0.65

    # Duplicate-detection thresholds
    DEDUP_IOU_THRESH  = 0.15
    DEDUP_CENTRE_FRAC = 0.55

    def __init__(self, zone: QueueZone = None):
        self.zone                   = zone or QueueZone()
        self.active_queue: OrderedDict[int, QueuePerson] = OrderedDict()
        self._candidates: dict      = {}
        self._used_numbers: set     = set()
        self._highest_assigned      = 0
        self._done_cooldowns: list  = []
        self.completed_queue: list  = []
        self.total_served           = 0
        self._noshow_timers: dict   = {}
        self.appearance_rejections: list = []
        self.on_new_person          = None
        self._done_appearances: list = []
        self._claimed_this_frame: dict = {}

    # =========================================================================
    # APPEARANCE HELPERS
    # =========================================================================

    @staticmethod
    def _extract_appearance(frame, bbox) -> np.ndarray | None:
        if frame is None:
            return None
        x1, y1, x2, y2 = bbox
        h_f, w_f = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_f - 1, x2), min(h_f - 1, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    @staticmethod
    def _compare_sigs(sig1: np.ndarray, sig2: np.ndarray) -> float:
        if sig1 is None or sig2 is None:
            return 0.0
        return float(cv2.compareHist(
            sig1.reshape(16, 16).astype(np.float32),
            sig2.reshape(16, 16).astype(np.float32),
            cv2.HISTCMP_CORREL
        ))

    def _best_score_against_person(self, person: QueuePerson, sig: np.ndarray) -> float:
        if sig is None or person.appearance_signature is None:
            return 0.0
        best = self._compare_sigs(person.appearance_signature, sig)
        for h in person.appearance_history:
            s = self._compare_sigs(h, sig)
            if s > best:
                best = s
        return best

    def _matches_done_person(self, sig: np.ndarray) -> bool:
        if sig is None or not self._done_appearances:
            return False
        thresh = self.DONE_BLACKLIST_THRESH
        return any(self._compare_sigs(d, sig) >= thresh for d in self._done_appearances)

    def _register_done_appearance(self, person: QueuePerson):
        sigs = []
        if person.appearance_signature is not None:
            sigs.append(person.appearance_signature.copy())
        sigs.extend(h.copy() for h in person.appearance_history)
        self._done_appearances.extend(sigs)
        if len(self._done_appearances) > self._DONE_APP_MAX:
            self._done_appearances = self._done_appearances[-self._DONE_APP_MAX:]
        print(f"📋 Q{person.queue_number:03d} added to done blacklist ({len(sigs)} sigs)")

    def _update_appearance(self, person: QueuePerson, frame, bbox):
        new_sig = self._extract_appearance(frame, bbox)
        if new_sig is None:
            return
        if person.appearance_signature is None:
            person.appearance_signature = new_sig
        else:
            person.appearance_signature = 0.6 * person.appearance_signature + 0.4 * new_sig
        history = person.appearance_history
        history.append(new_sig)
        if len(history) > 5:
            history.pop(0)

    # =========================================================================
    # HYBRID RE-ENTRY MATCHING
    # =========================================================================

    def _get_missing_persons(self, current_track_ids: set) -> list:
        claimed = self._claimed_this_frame
        absent = [
            (tid, p) for tid, p in self.active_queue.items()
            if tid not in current_track_ids
            and p.status != 'done_pending'
            and tid not in claimed
            and (p.status == 'missing' or (p.status == 'waiting' and p.went_missing_at is not None))
        ]
        absent.sort(key=lambda x: x[1].went_missing_at or datetime.min, reverse=True)
        return absent

    @staticmethod
    def _spatial_score(last_bbox: tuple, new_bbox: tuple) -> float:
        cx1 = (last_bbox[0] + last_bbox[2]) / 2
        cy1 = (last_bbox[1] + last_bbox[3]) / 2
        cx2 = (new_bbox[0]  + new_bbox[2])  / 2
        cy2 = (new_bbox[1]  + new_bbox[3])  / 2
        dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
        d1   = ((last_bbox[2]-last_bbox[0])**2 + (last_bbox[3]-last_bbox[1])**2) ** 0.5
        d2   = ((new_bbox[2] -new_bbox[0]) **2 + (new_bbox[3] -new_bbox[1]) **2) ** 0.5
        denom = (d1 + d2) / 2
        return dist / denom if denom > 0 else 999.0

    def _best_spatial_match(self, bbox: tuple, missing: list):
        best_score, best_tid, best_p = 999.0, None, None
        for tid, p in missing:
            s = self._spatial_score(p.bbox, bbox)
            if s < best_score:
                best_score, best_tid, best_p = s, tid, p
        return best_tid, best_p, best_score

    def _find_returning_person(self, bbox, new_sig, current_track_ids):
        if self._matches_done_person(new_sig):
            print("🚫 Matches done blacklist — new number will be assigned")
            return None, None, 0.0

        missing = self._get_missing_persons(current_track_ids)
        if not missing:
            return None, None, 0.0

        recency = self.RECENCY_WINDOW_SECONDS
        tbreak  = self.APPEARANCE_TIEBREAK_THRESHOLD

        if len(missing) == 1:
            tid, p = missing[0]
            secs = p.seconds_missing
            if secs <= recency:
                print(f"✅ Single absent Q{p.queue_number:03d} ({secs:.1f}s) — recency restore")
                return tid, p, 1.0
            score = self._best_score_against_person(p, new_sig)
            if score >= tbreak:
                print(f"✅ Q{p.queue_number:03d} matched by appearance (score={score:.2f})")
                return tid, p, score
            print(f"⚠️  Q{p.queue_number:03d} {secs:.0f}s missing, app={score:.2f} — new number")
            return None, None, 0.0

        app_scored = sorted(
            ((self._best_score_against_person(p, new_sig), tid, p) for tid, p in missing),
            reverse=True
        )
        best_app, best_app_tid, best_app_p = app_scored[0]
        second_app = app_scored[1][0] if len(app_scored) >= 2 else 0.0

        if best_app >= tbreak and (best_app - second_app) >= 0.10:
            print(f"✅ Q{best_app_p.queue_number:03d} appearance winner "
                  f"(score={best_app:.2f}, gap={best_app - second_app:.2f})")
            return best_app_tid, best_app_p, best_app

        sp_tid, sp_p, sp_score = self._best_spatial_match(bbox, missing)
        if sp_p is not None and sp_p.seconds_missing <= recency:
            other_scores = [
                self._spatial_score(p.bbox, bbox)
                for tid, p in missing if tid != sp_tid
            ]
            second_sp = min(other_scores) if other_scores else 999.0
            sp_gap    = second_sp - sp_score

            if sp_score < 0.50:
                print(f"✅ Q{sp_p.queue_number:03d} spatial match "
                      f"(norm_dist={sp_score:.2f}, gap={sp_gap:.2f})")
                return sp_tid, sp_p, 0.8

            if sp_score < 1.0 and sp_gap > 0.30:
                print(f"✅ Q{sp_p.queue_number:03d} spatial match (moderate "
                      f"norm_dist={sp_score:.2f}, gap={sp_gap:.2f})")
                return sp_tid, sp_p, 0.6

        rr_tid, rr_p = missing[0]
        secs = rr_p.seconds_missing
        if secs <= recency:
            print(f"⚠️  All matching inconclusive — fallback to most recent "
                  f"Q{rr_p.queue_number:03d} ({secs:.0f}s missing)")
            return rr_tid, rr_p, 0.4
        return None, None, 0.0

    # =========================================================================
    # NO-SHOW HANDLING
    # =========================================================================

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
            completed = p.to_dict()
            completed.update({
                'completed_at':      now.strftime("%I:%M:%S %p"),
                'completed_at_full': now.strftime("%b %d, %Y %I:%M:%S %p"),
                'total_wait_time':   p.wait_time_str,
                'bump_reason':       'no_show',
            })
            self.completed_queue.append(completed)
            self.total_served += 1
            self._register_done_appearance(p)
            p.status         = 'done_pending'
            p.missing_frames = self.MAX_MISSING_FRAMES + 1
            self._done_cooldowns.append({'bbox': p.bbox, 'frames_left': self.DONE_COOLDOWN_FRAMES})
            self._recalculate_positions()

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
                    'queue_number':      f"Q{qn:03d}",
                    'queue_number_int':  qn,
                    'seconds_remaining': int(remaining),
                    'status': 'critical' if remaining <= 15 else 'warning',
                })
        return alerts

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

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

    @staticmethod
    def _bbox_centre(bbox) -> tuple:
        return ((bbox[0] + bbox[2]) >> 1, (bbox[1] + bbox[3]) >> 1)

    @staticmethod
    def _bbox_diagonal(bbox) -> float:
        return ((bbox[2]-bbox[0])**2 + (bbox[3]-bbox[1])**2) ** 0.5

    @staticmethod
    def _x_column_overlap(b1, b2) -> float:
        ox     = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
        min_w  = min(b1[2]-b1[0], b2[2]-b2[0])
        return ox / min_w if min_w > 0 else 0.0

    @staticmethod
    def _y_adjacent(b1, b2, gap_frac=0.6) -> bool:
        y_gap = max(0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
        min_h = min(b1[3]-b1[1], b2[3]-b2[1])
        return y_gap < gap_frac * min_h if min_h > 0 else False

    def _is_duplicate_of(self, bbox_a, bbox_b) -> bool:
        if self._iou(bbox_a, bbox_b) > self.DEDUP_IOU_THRESH:
            return True
        return (self._x_column_overlap(bbox_a, bbox_b) > 0.60
                and self._y_adjacent(bbox_a, bbox_b, gap_frac=0.6))

    def _find_overlapping_candidate(self, bbox):
        for tid, info in self._candidates.items():
            if self._is_duplicate_of(info['bbox'], bbox):
                return tid
        return None

    def _is_duplicate_of_active(self, bbox) -> bool:
        for p in self.active_queue.values():
            if p.status == 'done_pending':
                continue
            if self._is_duplicate_of(p.bbox, bbox):
                return True
        return False

    def _restore_missing_person(self, ret_person, ret_tid, track_id, frame, bbox, new_sig):
        ret_person.track_id        = track_id
        ret_person.bbox            = bbox
        ret_person.missing_frames  = 0
        ret_person.status          = 'waiting'
        ret_person.went_missing_at = None
        self._noshow_timers.pop(ret_person.queue_number, None)
        if new_sig is not None:
            self._update_appearance(ret_person, frame, bbox)
        if ret_tid != track_id:
            self.active_queue[track_id] = ret_person
            del self.active_queue[ret_tid]
        self._candidates.pop(track_id, None)
        self._claimed_this_frame[ret_tid] = track_id

    def _has_sufficient_motion(self, centers: list) -> bool:
        """
        Reject static objects (pictures, signs, furniture) that YOLO detects
        consistently but that never actually move.

        Two independent checks:

        1. Range check (original): max displacement across the history window
           must exceed MIN_MOTION_PIXELS.  Raised to 8 px so that camera
           shake / JPEG compression artefacts don't inflate the score.

        2. Standard-deviation check (new): even if a camera wobble briefly
           displaces the box by >8 px, a static object's centre positions will
           cluster very tightly over time (stdev << 1 px).  A real person
           standing "still" still has 2–5 px of natural micro-sway in both
           axes.  If stdev_x < STATIC_STDEV_THRESHOLD AND
           stdev_y < STATIC_STDEV_THRESHOLD, the detection is rejected.
        """
        if len(centers) < 8:
            return True  # not enough data yet — give benefit of the doubt

        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]

        # Check 1: range
        movement = max(max(xs) - min(xs), max(ys) - min(ys))
        if movement < self.MIN_MOTION_PIXELS:
            print(f"🖼️  Static rejection — range={movement}px "
                  f"(need >{self.MIN_MOTION_PIXELS}px) — likely a picture/object")
            return False

        # Check 2: standard deviation (requires ≥ 10 samples for reliability)
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
        """
        Reject bounding boxes whose shape is inconsistent with a standing person.

        A standing or slightly crouching person is always taller than wide
        (portrait orientation).  Framed pictures and signs hanging on walls
        are typically landscape or square (wide >= tall).

        Aspect ratio = height / width.
          ≥ 0.65  → plausible person (standing, partial, or crouching)
          < 0.65  → likely a non-person object → reject

        This is applied BEFORE candidate accumulation so the object never
        even starts collecting confirmation frames.
        """
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

    # =========================================================================
    # MAIN FRAME PROCESSOR
    # =========================================================================

    def process_frame(self, tracked_persons: list, frame=None) -> dict:
        self._tick_done_cooldowns()
        self._claimed_this_frame = {}
        current_in_zone = set()

        for person in tracked_persons:
            track_id = person['track_id']
            bbox     = person['bbox']
            in_zone  = self.zone.is_person_inside(bbox)

            # ── Already tracked ────────────────────────────────────────────────
            if track_id in self.active_queue:
                p            = self.active_queue[track_id]
                p.bbox       = bbox
                p.last_seen  = datetime.now()
                p.missing_frames = 0
                if frame is not None and p.wait_time_seconds % 30 == 0:
                    self._update_appearance(p, frame, bbox)
                if in_zone:
                    current_in_zone.add(track_id)
                    if p.status in ('missing', 'waiting') and p.went_missing_at is not None:
                        p.status          = 'waiting'
                        p.went_missing_at = None
                        self._noshow_timers.pop(p.queue_number, None)
                        print(f"✅ Q{p.queue_number:03d} back in zone (same track_id)")
                continue

            if not in_zone:
                self._candidates.pop(track_id, None)
                continue

            # ── NEW: Portrait aspect-ratio guard ──────────────────────────────
            # Reject landscape/square boxes before they enter the candidate
            # pool.  Pictures on walls, signs, and other static rectangular
            # objects are almost always wider than tall.  This check is fast
            # and runs before any appearance extraction.
            if not self._is_plausible_person_bbox(bbox):
                self._candidates.pop(track_id, None)
                continue

            current_in_zone.add(track_id)
            new_sig = self._extract_appearance(frame, bbox)

            # ── Hybrid re-entry check ──────────────────────────────────────────
            ret_tid, ret_person, ret_score = self._find_returning_person(
                bbox, new_sig, current_in_zone)
            if ret_person is not None:
                print(f"✅ Q{ret_person.queue_number:03d} re-entry "
                      f"(score={ret_score:.2f}, track {ret_tid}→{track_id})")
                self._restore_missing_person(ret_person, ret_tid, track_id, frame, bbox, new_sig)
                continue

            # ── Done cooldown block ────────────────────────────────────────────
            if self._is_in_done_cooldown(bbox):
                self._candidates.pop(track_id, None)
                continue

            # ── Active queue duplicate guard ───────────────────────────────────
            if self._is_duplicate_of_active(bbox):
                self._candidates.pop(track_id, None)
                print(f"🔁 track {track_id} suppressed — duplicate of active person")
                continue

            # ── Candidate pool duplicate / flicker dedup ──────────────────────
            cand_tid = self._find_overlapping_candidate(bbox)
            if cand_tid is not None and cand_tid != track_id:
                info = self._candidates.pop(cand_tid)
                info['bbox'] = bbox
                if new_sig is not None:
                    info.setdefault('sigs', []).append(new_sig)
                cx, cy = (bbox[0] + bbox[2]) >> 1, (bbox[1] + bbox[3]) >> 1
                info.setdefault('centers', []).append((cx, cy))
                self._candidates[track_id] = info
                continue

            # ── Accumulate confirmation frames ─────────────────────────────────
            cand = self._candidates.setdefault(track_id, {
                'count': 0, 'bbox': bbox, 'sigs': [], 'centers': []
            })
            cx, cy = (bbox[0] + bbox[2]) >> 1, (bbox[1] + bbox[3]) >> 1
            cand['count'] += 1
            cand['bbox']   = bbox
            centers = cand['centers']
            centers.append((cx, cy))
            if len(centers) > self.MOTION_HISTORY_LEN:
                centers.pop(0)
            if new_sig is not None:
                sigs = cand['sigs']
                sigs.append(new_sig)
                if len(sigs) > 8:
                    sigs.pop(0)

            # Re-check re-entry every frame during accumulation
            early_tid, early_person, early_score = self._find_returning_person(
                bbox, new_sig, current_in_zone)
            if early_person is not None:
                print(f"✅ Q{early_person.queue_number:03d} re-entry (accumulation "
                      f"frame {cand['count']}, score={early_score:.2f})")
                self._restore_missing_person(
                    early_person, early_tid, track_id, frame, bbox, new_sig)
                continue

            # ── Assign new number once confirmed ──────────────────────────────
            if cand['count'] < self.MIN_CONFIRM_FRAMES:
                continue

            # Motion + stdev check — rejects pictures/signs that accumulate
            # frames without ever moving.  Runs at confirmation time so the
            # full centre history is available for a reliable stdev reading.
            if not self._has_sufficient_motion(cand.get('centers', [])):
                self._candidates.pop(track_id)
                continue

            # Final gate re-entry check
            final_tid, final_person, final_score = self._find_returning_person(
                bbox, new_sig, current_in_zone)
            if final_person is not None:
                print(f"✅ Q{final_person.queue_number:03d} final gate "
                      f"(score={final_score:.2f}) — not new")
                self._restore_missing_person(
                    final_person, final_tid, track_id, frame, bbox, new_sig)
                continue

            # Genuinely new person
            self._highest_assigned += 1
            num   = self._highest_assigned
            new_p = QueuePerson(queue_number=num, track_id=track_id, bbox=bbox)

            cand_sigs = cand.get('sigs', [])
            if cand_sigs:
                new_p.appearance_signature = np.mean(np.stack(cand_sigs), axis=0)
                new_p.appearance_history   = cand_sigs[-3:]
            else:
                new_p.appearance_signature = new_sig

            self._candidates.pop(track_id)
            self.active_queue[track_id] = new_p
            self._used_numbers.add(num)
            print(f"🆕 Q{num:03d} NEW person (track_id={track_id})")

            if self.on_new_person:
                try:
                    self.on_new_person(num, new_p.wait_time_str,
                                       new_p.joined_at_str, new_p.access_token)
                except Exception as e:
                    print(f"⚠️  on_new_person callback error: {e}")

        # ── Handle persons not seen this frame ────────────────────────────────
        to_remove = []
        for tid, p in self.active_queue.items():
            if tid in current_in_zone:
                continue
            p.missing_frames += 1

            if p.status == 'done_pending':
                if p.missing_frames > 10:
                    self._used_numbers.discard(p.queue_number)
                    to_remove.append(tid)
                    for entry in self._done_cooldowns:
                        if self._iou(entry['bbox'], p.bbox) > 0.3:
                            entry['bbox'] = p.bbox
                            break
                    print(f"👋 Q{p.queue_number:03d} left frame — freed")

            elif p.status == 'waiting':
                if p.went_missing_at is None:
                    p.went_missing_at = datetime.now()
                if p.missing_frames > self.MAX_MISSING_FRAMES:
                    p.status = 'missing'
                    print(f"❓ Q{p.queue_number:03d} went missing ({p.missing_frames} frames)")

        for tid in to_remove:
            del self.active_queue[tid]

        # ── Post-frame active-queue dedup pass ────────────────────────────────
        self._dedup_active_queue()

        self._recalculate_positions()
        self._check_noshow()
        return self.get_state()

    # =========================================================================
    # MARK DONE
    # =========================================================================

    def mark_transaction_done(self, queue_number: int) -> bool:
        for tid, p in self.active_queue.items():
            if p.queue_number == queue_number and p.status in ('waiting', 'missing'):
                p.status           = 'done_pending'
                p.position_in_line = 0
                self.total_served  += 1
                self._register_done_appearance(p)
                self._done_cooldowns.append(
                    {'bbox': p.bbox, 'frames_left': self.DONE_COOLDOWN_FRAMES})
                self.completed_queue.append(self._make_completed_entry(p, 'served'))
                self._noshow_timers.pop(queue_number, None)
                self._recalculate_positions()
                print(f"✅ Q{queue_number:03d} DONE | Wait: {p.wait_time_str}")
                return True
        return False

    # =========================================================================
    # TOKEN LOOKUP
    # =========================================================================

    def lookup_by_token(self, queue_number: int, token: str) -> dict | None:
        for p in self.active_queue.values():
            if p.queue_number != queue_number:
                continue
            if p.access_token != token:
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
        return None

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _dedup_active_queue(self):
        tids    = list(self.active_queue.keys())
        to_drop = set()

        for i in range(len(tids)):
            if tids[i] in to_drop:
                continue
            p_i = self.active_queue[tids[i]]
            if p_i.status == 'done_pending':
                continue

            for j in range(i + 1, len(tids)):
                if tids[j] in to_drop:
                    continue
                p_j = self.active_queue[tids[j]]
                if p_j.status == 'done_pending':
                    continue

                if self._is_duplicate_of(p_i.bbox, p_j.bbox):
                    keep, drop = (p_i, p_j) if p_i.queue_number < p_j.queue_number \
                                             else (p_j, p_i)
                    drop_tid = tids[i] if drop is p_i else tids[j]
                    print(f"♻️  Dedup: Q{drop.queue_number:03d} is duplicate of "
                          f"Q{keep.queue_number:03d} — retiring ghost")
                    drop.status         = 'done_pending'
                    drop.missing_frames = self.MAX_MISSING_FRAMES + 1
                    self._used_numbers.discard(drop.queue_number)
                    to_drop.add(drop_tid)

        for tid in to_drop:
            del self.active_queue[tid]

    def _recalculate_positions(self):
        active_line = sorted(
            (p for p in self.active_queue.values() if p.status in ('waiting', 'missing')),
            key=lambda x: x.queue_number
        )
        for i, p in enumerate(active_line):
            p.position_in_line = i + 1

    def get_state(self) -> dict:
        active = sorted(
            (p.to_dict() for p in self.active_queue.values()
             if p.status in ('waiting', 'missing')),
            key=lambda x: x['queue_number']
        )
        return {
            'active_queue':          active,
            'queue_count':           len(active),
            'next_number':           self._highest_assigned + 1,
            'total_served':          self.total_served,
            'completed':             self.completed_queue[-10:],
            'noshow_alerts':         self.get_noshow_alerts(),
            'appearance_rejections': self.appearance_rejections[-5:],
        }

    # =========================================================================
    # DRAW ON FRAME
    # =========================================================================

    def draw_on_frame(self, frame):
        h, w = frame.shape[:2]
        z    = self.zone
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        # Zone rectangle
        cv2.rectangle(frame, (z.x1, z.y1), (z.x2, z.y2), (0, 255, 255), 2)
        lbl = "QUEUE ZONE"
        (lw, lh), _ = cv2.getTextSize(lbl, FONT, 0.55, 2)
        lx, ly = z.x1 + 6, z.y1 + lh + 8
        cv2.rectangle(frame, (lx-2, ly-lh-4), (lx+lw+2, ly+4), (0, 0, 0), -1)
        cv2.putText(frame, lbl, (lx, ly), FONT, 0.55, (0, 255, 255), 2)

        # No-show warnings
        for i, alert in enumerate(self.get_noshow_alerts()):
            color    = (0, 0, 255) if alert['status'] == 'critical' else (0, 165, 255)
            warn_txt = (f"{alert['queue_number']} NO-SHOW WARNING "
                        f"Bumping in {alert['seconds_remaining']}s")
            (aw, ah), _ = cv2.getTextSize(warn_txt, FONT, 0.5, 2)
            ay = h - 20 - i * 28
            cv2.rectangle(frame, (8, ay-ah-4), (aw+16, ay+4), (0, 0, 0), -1)
            cv2.putText(frame, warn_txt, (12, ay), FONT, 0.5, color, 2)

        # Person boxes
        for person in self.active_queue.values():
            x1, y1, x2, y2 = person.bbox
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w-1, x2); y2 = min(h-1, y2)
            label = f"Q{person.queue_number:03d}"

            if person.status == 'done_pending':
                box_color, info_text, text_color = (0,0,255), "DONE - EXIT PLEASE", (0,0,255)
                thickness = 3
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

        # Summary
        waiting = sum(1 for p in self.active_queue.values()
                      if p.status in ('waiting', 'missing'))
        summary = f"Queue: {waiting} waiting"
        (sw, sh), _ = cv2.getTextSize(summary, FONT, 0.6, 2)
        sx = w - sw - 12
        cv2.rectangle(frame, (sx-4, 4), (sx+sw+4, sh+14), (0,0,0), -1)
        cv2.putText(frame, summary, (sx, sh+10), FONT, 0.6, (0,255,255), 2)

        return frame