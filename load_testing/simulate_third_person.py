"""
Third-person test: two students already hold queue numbers and a third
arrives. Measures how long the third student waits for a number, and checks
that nobody's number is lost or handed to someone else.

Runs the real backend code (services/queue_service.py, queue_tracker.py) and
the detector's real face schedule (services/face_schedule.py) on a simulated
clock. No camera, no database, nothing deployed is touched.

What is simulated, and why:

- One update reaches the backend about every second. The detector sends every
  YOLO frame (PUSH_EVERY=1), but each push waits for the backend's answer,
  and a request to the Render backend took about 0.8 s (measured 25 Nov
  2026); on 24 Nov the backend logged about one update a second.
- The detector's face worker reads one face in about 0.26 s, two people per
  round, choosing whom to read with FaceSchedule from the backend's last
  answer. A read finds the person's face with a set chance: lower when two
  people overlap. When people overlap, a read sometimes returns the
  neighbour's face instead (MISREAD_BESIDE, 10%), and on someone standing in
  front of a hidden person, that person's face peeking past (MISREAD_HIDDEN,
  3%).
- Face signatures are synthetic. Each student has a look for the day (the
  angle, light and expression the camera sees) as similar to their
  registration as real reads were in the face-match log (0.45-0.75); each
  read is close to that look (0.85-0.95), as frames a second apart are.

Scenarios, each starting with students 1 and 2 already numbered:
  beside       student 3 stands in a free spot next to them
  overlapping  student 3 stands half in front of student 1 (both visible)
  in front     student 3 walks in from the side, then stands in front of
               student 1, who is hidden from the camera until student 3 moves
  appears in front
               as above, but student 3 comes into view already in front of
               student 1, so the two were never seen together
Each runs twice: student 3 tapped Get a Queue Number before arriving, and
student 3 tapped it after standing in view for five seconds.

Run:  .venv/Scripts/python.exe load_testing/simulate_third_person.py [--trials 100]
"""
import argparse
import contextlib
import inspect
import io
import math
import random
import statistics
import sys
from datetime import datetime as _datetime
from datetime import timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "app"))

import database.database_handler as database_handler  # noqa: E402
from services import cache_service  # noqa: E402

# Nothing here may reach the live database or cache, even by a path this
# script forgot to stub: with no pool and no client, every call is a no-op.
database_handler._ensure_db_pool = lambda: None
cache_service.client = lambda: None

from services import queue_service  # noqa: E402
from services import queue_tracker as tracker_module  # noqa: E402
from services.embedding_index import EmbeddingIndex  # noqa: E402
from services.face_schedule import FaceSchedule, attach_fresh_embeddings, resolved_track_ids  # noqa: E402
from services.queue_tracker import QueueTracker, QueueZone  # noqa: E402

UPDATE_SECONDS = 1.0        # one backend update a second
FACE_SECONDS = 0.26         # one face signature on the office laptop
GIVE_UP_SECONDS = 60
OTHER_ENROLLED = 30         # registered students who are not in view
# How often a successful face read returns the wrong person's face:
# a neighbour standing half in front or behind, both visible, and a person
# hidden behind someone, whose face can only show peeking past their head.
MISREAD_BESIDE = 0.10
MISREAD_HIDDEN = 0.03

# Boxes in the 480x360 tracking frame: someone about 2 m away, and someone
# a step closer to the camera (in front).
SPOT_1 = (40, 90, 150, 350)
SPOT_2 = (330, 90, 440, 350)
SPOT_FREE = (185, 90, 295, 350)          # between them, no overlap
SPOT_OVERLAP = (95, 90, 205, 350)        # half in front of student 1
SPOT_FRONT = (30, 50, 180, 360)          # right in front of student 1

DEPLOYED_SETTINGS = (
    "MAX_MISSING_FRAMES", "MIN_CONFIRM_FRAMES", "MIN_CONFIRM_SECONDS",
    "MIN_CONFIRM_FACE_FRAMES", "MIN_MOTION_PIXELS",
    "STATIC_STDEV_THRESHOLD", "STATIC_CONF_BYPASS_THRESHOLD",
    "MIN_PORTRAIT_ASPECT", "FACE_MATCH_THRESHOLD", "FACE_MARGIN_THRESHOLD",
    "PENDING_LINK_TIMEOUT_SECONDS", "DEDUP_IOU_THRESH", "DEDUP_CENTRE_FRAC",
)


class Clock:
    now = _datetime(2026, 11, 25, 9, 0, 0)


class SimDatetime(_datetime):
    """datetime.now() for the tracker, on the simulated clock."""

    @classmethod
    def now(cls, tz=None):
        return Clock.now


def unit(v):
    return v / np.linalg.norm(v)


def registration(rng):
    return unit(rng.normal(size=512).astype(np.float32))


def near_vector(v, cosine, rng):
    """A unit vector at about the given cosine to `v`."""
    noise = unit(rng.normal(size=512).astype(np.float32))
    return unit(v + math.sqrt(1 / cosine ** 2 - 1) * noise)


def live_read(enrolled, rng):
    """A camera read of an enrolled face, as similar to the registration as
    real reads were (cosine 0.45-0.75)."""
    return near_vector(enrolled, rng.uniform(0.45, 0.75), rng)


def supports(fn, name):
    return name in inspect.signature(fn).parameters


class Trial:
    def __init__(self, scenario, tap_after, seed):
        self.scenario, self.tap_after = scenario, tap_after
        self.rng = np.random.default_rng(seed)
        self.pick = random.Random(seed)
        self.faces = {sid: registration(self.rng) for sid in (1, 2, 3)}
        enrolled = dict(self.faces)
        for k in range(OTHER_ENROLLED):
            enrolled[100 + k] = registration(self.rng)
        self.enrolled = enrolled
        self.schedule = FaceSchedule(per_round=2, retry_gap=0.15, slow_gap=1.0,
                                     warmup=6, refresh_seconds=2.0)
        self.urgent_supported = supports(self.schedule.choose, "urgent")
        self.fresh = {}
        self.looks = {}
        self.resolved, self.urgent = set(), False
        self.frame_no = 0
        self.tickets = []

    # ------------------------------------------------------------ plumbing
    def install(self, patches):
        tracker = QueueTracker(zone=QueueZone(x1=0, y1=0, x2=480, y2=360))
        for name in DEPLOYED_SETTINGS:
            setattr(tracker, name, getattr(queue_service.queue_tracker, name))
        tracker._num_counters = 3
        tracker.on_number_linked = queue_service._on_number_linked
        enrolled = self.enrolled
        patches.append((queue_service, "queue_tracker", tracker))
        patches.append((queue_service, "embedding_index", EmbeddingIndex(
            loader=lambda: [(sid, v.tolist()) for sid, v in enrolled.items()], ttl_seconds=3600)))
        patches.append((queue_service, "record_face_match_event", lambda *a, **k: None))
        patches.append((queue_service, "record_recognition_metric", lambda *a, **k: None))
        patches.append((queue_service, "update_queue_status", lambda *a, **k: None))
        patches.append((queue_service.ticket_service, "enqueue_ticket",
                        lambda **k: self.tickets.append(k["queue_number"])))
        patches.append((tracker_module, "datetime", SimDatetime))
        if hasattr(queue_service, "_detector_numbers"):
            patches.append((queue_service, "_detector_numbers", {}))
        self.tracker = tracker

    def step(self, people):
        """One second: the face worker reads faces, then one update goes to
        the backend. `people` is [(detector_id, student, box, p_face, p_wrong,
        neighbour)]."""
        start = Clock.now
        present = [d for d, *_ in people]
        info = {d: (sid, p, wrong, nb) for d, sid, _, p, wrong, nb in people}
        self.schedule.forget_absent(set(present), start.timestamp())
        t = 0.0
        while t < UPDATE_SECONDS:
            now = start.timestamp() + t
            kwargs = {"urgent": self.urgent} if self.urgent_supported else {}
            picks = self.schedule.choose(present, self.resolved, now, **kwargs)
            if not picks:
                t += 0.05
                continue
            for d in picks:
                t += FACE_SECONDS
                sid, p, wrong, nb = info[d]
                ok = self.pick.random() < p
                if ok:
                    who = nb if (nb and self.pick.random() < wrong) else sid
                    self.fresh[d] = self.read_face(who).tolist()
                self.schedule.record(d, ok, start.timestamp() + t)
        Clock.now = start + timedelta(seconds=UPDATE_SECONDS)
        self.frame_no += 1
        tracked = [{"track_id": d, "bbox": list(box), "conf": 0.85}
                   for d, _, box, *_ in people]
        payload = attach_fresh_embeddings(tracked, self.fresh)
        queue_service.process_tracked_persons(payload, self.frame_no)
        reply = {"face_resolved": queue_service.face_resolved_track_ids()}
        if hasattr(queue_service, "faces_wanted"):
            reply["faces_wanted"] = queue_service.faces_wanted()
        self.resolved = resolved_track_ids(reply)
        self.urgent = bool(reply.get("faces_wanted", False))

    def read_face(self, sid):
        """One read of a student's face: close to their look today."""
        if sid not in self.looks:
            self.looks[sid] = live_read(self.faces[sid], self.rng)
        return near_vector(self.looks[sid], self.rng.uniform(0.85, 0.95), self.rng)

    def number_of(self, sid):
        p = self.tracker.get_person_by_student_id(sid)
        return None if p is None else p.queue_number

    # ------------------------------------------------------------ the trial
    def run(self):
        patches = []
        self.install(patches)
        saved = [(obj, name, getattr(obj, name)) for obj, name, _ in patches]
        for obj, name, value in patches:
            setattr(obj, name, value)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return self._run()
        finally:
            for obj, name, value in saved:
                setattr(obj, name, value)

    def _run(self):
        tr = self.tracker
        one = (10, 1, SPOT_1, 0.65, 0.0, None)
        two = (11, 2, SPOT_2, 0.65, 0.0, None)
        tr.arm_join_intent(1)
        tr.arm_join_intent(2)
        for _ in range(15):
            self.step([one, two])
            if self.number_of(1) and self.number_of(2):
                break
        q1, q2 = self.number_of(1), self.number_of(2)
        if not (q1 and q2):
            return {"setup_failed": True}
        for _ in range(3):                     # settled, before student 3 comes
            self.step([one, two])

        if not self.tap_after:
            tr.arm_join_intent(3)
        issued_at = None
        for sec in range(GIVE_UP_SECONDS + 5):
            if self.tap_after and sec == 5:
                tr.arm_join_intent(3)
            people = [two]
            if self.scenario == "beside":
                people += [one, (12, 3, SPOT_FREE, 0.65, 0.0, None)]
            elif self.scenario == "overlapping":
                people += [(10, 1, SPOT_1, 0.45, MISREAD_BESIDE, 3),
                           (12, 3, SPOT_OVERLAP, 0.45, MISREAD_BESIDE, 1)]
            elif self.scenario == "in front" and sec < 2:
                people += [one, (12, 3, SPOT_FREE, 0.65, 0.0, None)]
            else:                              # in front of student 1
                people += [(12, 3, SPOT_FRONT, 0.6, MISREAD_HIDDEN, 1)]
            self.step(people)
            asked = 5 if self.tap_after else 0
            if issued_at is None and self.number_of(3) is not None and sec + 1 > asked:
                issued_at = sec + 1 - asked
            if issued_at is not None:
                break

        # Student 3 steps aside and student 1 is in view again.
        for _ in range(8):
            self.step([two, (12, 3, SPOT_FREE, 0.65, 0.0, None), one])

        # Each number must be on the right student AND on that student's own
        # box: a number moved onto someone else's body still says the right
        # student id, so the box is checked too. The detector numbers are 10,
        # 11 and 12 for students 1, 2 and 3; the backend may hold a person
        # under an older number it mapped one of them to.
        to_detector = {held: det for det, held in
                       getattr(queue_service, "_detector_numbers", {}).items()}
        on_box = {}
        for key, p in tr.active_queue.items():
            if p.queue_number is not None:
                on_box[p.queue_number] = (p.student_id, to_detector.get(key, key))
        q3 = self.number_of(3)
        intact = (self.number_of(1) == q1 and self.number_of(2) == q2
                  and on_box.get(q1) == (1, 10) and on_box.get(q2) == (2, 11))
        return {
            "setup_failed": False,
            "seconds": issued_at,
            "own_number": q3 is not None and q3 not in (q1, q2) and on_box.get(q3) == (3, 12),
            "others_intact": intact,
            "duplicate_tickets": len(self.tickets) != len(set(self.tickets)),
        }


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


class Crowd(Trial):
    """A busy line: CROWD_STUDENTS students arrive one after another, every
    4-16 s. Half take a free spot in the line, a quarter stand half in front
    of someone, a quarter right in front of someone (who is then hidden
    until a spot frees up). 60% tapped Get a Queue Number before arriving,
    the rest 3-8 s after. Staff serve the lowest number every 50-110 s, and
    that student leaves. Now and then the detector renumbers someone."""

    SLOTS = [(15 + 75 * k, 90, 125 + 75 * k, 350) for k in range(6)]

    @staticmethod
    def _half(box):
        """Half in front of someone: shifted 40 px, overlapping them."""
        x1, y1, x2, y2 = box
        return (x1 + 40, y1, x2 + 40, y2)

    @staticmethod
    def _front(box):
        """Right in front of someone, a step closer: bigger, hiding them."""
        x1, y1, x2, y2 = box
        return (max(0, x1 - 10), 50, min(480, x2 + 30), 360)

    def __init__(self, seed, students):
        self.students = students
        super().__init__("crowd", False, seed)
        self.faces = {sid: registration(self.rng) for sid in range(1, students + 1)}
        self.enrolled = dict(self.faces)
        for k in range(OTHER_ENROLLED):
            self.enrolled[1000 + k] = registration(self.rng)

    def _run(self):
        tr, pick = self.tracker, self.pick
        arrivals, t = [], 0.0
        for sid in range(1, self.students + 1):
            t += pick.uniform(4, 16)
            tap = t - 5 if pick.random() < 0.6 else t + pick.uniform(3, 8)
            arrivals.append((int(t), sid, int(tap)))
        next_det = [100]
        self.how, self.violations = {}, []     # for looking into a run
        here = {}      # sid -> {"det", "box", "front_of", "since", "leave_at"}
        asked_at, issued, in_view_at = {}, {}, {}
        outside = []   # arrived, waiting for room to step into view
        wrong_body = set()
        wrong_seconds = 0
        next_service = pick.uniform(50, 110)
        end = arrivals[-1][0] + 90

        def new_det():
            next_det[0] += 1
            return next_det[0]

        def hidden():
            return {h["front_of"] for h in here.values() if h["front_of"] is not None}

        def place(sid, sec):
            taken = {tuple(h["box"]) for h in here.values()}
            free = [b for b in self.SLOTS if b not in taken]
            roll = pick.random()
            hid = hidden()
            visible = [s for s in here if s not in hid and here[s]["front_of"] is None]

            def fits(box, beside=None):
                # In the picture, and not where someone else already is
                # (two people cannot share one box).
                return box[0] >= 0 and box[2] <= 480 and all(
                    _iou(box, h["box"]) < 0.6 for s, h in here.items() if s != beside)

            half = [o for o in visible if fits(self._half(here[o]["box"]))]
            front = [o for o in visible if fits(self._front(here[o]["box"]), beside=o)]
            if free and (roll < 0.5 or not (half or front)):
                box, front_of, how = free[0], None, "free spot"
            elif half and (roll < 0.75 or not front):
                other = pick.choice(half)
                box, front_of, how = self._half(here[other]["box"]), None, f"half in front of {other}"
            elif front:
                other = pick.choice(front)
                box, front_of, how = self._front(here[other]["box"]), other, f"in front of {other}"
            else:
                return False                    # no room yet
            here[sid] = {"det": new_det(), "box": box, "front_of": front_of,
                         "since": sec, "leave_at": None}
            self.how[sid] = how
            return True

        for sec in range(end):
            for at, sid, tap in arrivals:
                if tap == sec and sid not in asked_at:
                    tr.arm_join_intent(sid)
                    asked_at[sid] = sec
                if at == sec:
                    outside.append(sid)
            for sid in list(outside):
                if place(sid, sec):
                    outside.remove(sid)

            # Staff serve the lowest number in line; that student leaves.
            if sec >= next_service:
                line = sorted((p.queue_number, p.student_id) for p in tr.active_queue.values()
                              if p.queue_number is not None and p.status in ("waiting", "missing"))
                if line:
                    number, sid = line[0]
                    queue_service.mark_done(number)
                    if sid in here:
                        here[sid]["leave_at"] = sec + 3
                next_service = sec + pick.uniform(50, 110)
            for sid in list(here):
                h = here[sid]
                leave = h["leave_at"] is not None and sec >= h["leave_at"]
                gave_up = (sid not in issued and sid in asked_at and sid in in_view_at
                           and sec - max(asked_at[sid], in_view_at[sid]) > GIVE_UP_SECONDS)
                if leave or gave_up:
                    del here[sid]
            # Someone standing in front steps into a free spot once they
            # have their number (or after 15 s).
            for sid, h in here.items():
                if h["front_of"] is None:
                    continue
                if sid in issued or sec - h["since"] > 15:
                    taken = {tuple(x["box"]) for x in here.values()}
                    free = [b for b in self.SLOTS if b not in taken]
                    if free:
                        h["box"], h["front_of"] = free[0], None
            # The detector renumbers someone now and then.
            for h in here.values():
                if pick.random() < 0.005:
                    h["det"] = new_det()

            hid = hidden()
            people = []
            for sid, h in here.items():
                if sid in hid:
                    continue
                box = h["box"]
                if h["front_of"] is not None:
                    nb, p, wrong = h["front_of"], 0.6, MISREAD_HIDDEN
                else:
                    nb = next((other for other, o in here.items()
                               if other != sid and other not in hid and _iou(box, o["box"]) > 0.1), None)
                    p, wrong = (0.45, MISREAD_BESIDE) if nb is not None else (0.65, 0.0)
                people.append((h["det"], sid, box, p, wrong, nb))
            # A student's clock starts once the camera can see them: someone
            # hidden behind another person from the moment they arrive
            # cannot be given a number by any camera.
            for _, sid, *_ in people:
                in_view_at.setdefault(sid, sec)
            self.step(people)

            for sid in asked_at:
                if sid in in_view_at and sid not in issued and self.number_of(sid) is not None:
                    issued[sid] = sec + 1 - max(asked_at[sid], in_view_at[sid])
            # Is any number on someone else's body right now?
            to_detector = {held: det for det, held in queue_service._detector_numbers.items()}
            det_student = {h["det"]: sid for sid, h in here.items() if sid not in hid}
            for key, p in tr.active_queue.items():
                if p.queue_number is None or p.student_id is None or p.status == "done_pending":
                    continue
                body = det_student.get(to_detector.get(key, key))
                if body is not None and body != p.student_id:
                    wrong_body.add(p.student_id)
                    wrong_seconds += 1
                    self.violations.append((sec, p.student_id, body, key, key in to_detector))

        # Students who asked and whom the camera could see; anyone still
        # waiting outside, or hidden the whole time, is left out.
        asked = [s for s in asked_at if s in in_view_at]
        self.asked_at, self.issued, self.in_view_at = asked_at, issued, in_view_at  # for looking into a run
        return {
            "setup_failed": False,
            "asked": len(asked),
            "secs": [issued[s] for s in asked if s in issued],
            "never": sum(1 for s in asked if s not in issued),
            "wrong_body_students": len(wrong_body),
            "wrong_body_seconds": wrong_seconds,
            "duplicate_tickets": len(self.tickets) - len(set(self.tickets)),
        }


CROWD_STUDENTS = 24


def summarise(results):
    ok = [r for r in results if not r["setup_failed"]]
    secs = sorted(r["seconds"] for r in ok if r["seconds"] is not None)
    n = len(ok)
    within = lambda s: sum(1 for x in secs if x <= s)  # noqa: E731
    return {
        "trials": n,
        "issued": len(secs),
        "median": statistics.median(secs) if secs else None,
        "p90": secs[min(len(secs) - 1, int(0.9 * len(secs)))] if secs else None,
        "within5": within(5),
        "within10": within(10),
        "own": sum(r["own_number"] for r in ok),
        "intact": sum(r["others_intact"] for r in ok),
        "dupes": sum(r["duplicate_tickets"] for r in ok),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--crowds", type=int, default=30)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    scenarios = ["beside", "overlapping", "in front", "appears in front"]
    print(f"# Third person to get a number {args.label}".rstrip())
    print(f"{args.trials} trials per row; one backend update a second; "
          f"gives up after {GIVE_UP_SECONDS} s. Counts are out of {args.trials}.\n")
    print("| Student 3 | Tapped | Got a number | Median | 90th pct | Within 5 s | Within 10 s "
          "| Own number, on own box | 1 and 2 keep theirs, on own box |")
    print("|---|---|---|---|---|---|---|---|---|")
    for scenario in scenarios:
        for tap_after in (False, True):
            results = [Trial(scenario, tap_after, seed).run() for seed in range(args.trials)]
            s = summarise(results)
            fmt = lambda v: "-" if v is None else f"{v:g} s"  # noqa: E731
            print(f"| {scenario} | {'after 5 s in view' if tap_after else 'before arriving'} "
                  f"| {s['issued']} | {fmt(s['median'])} | {fmt(s['p90'])} "
                  f"| {s['within5']} | {s['within10']} | {s['own']} | {s['intact']} |"
                  + (f" ({s['dupes']} duplicate tickets)" if s["dupes"] else ""))

    if args.crowds <= 0:
        return
    runs = [Crowd(seed, CROWD_STUDENTS).run() for seed in range(args.crowds)]
    asked = sum(r["asked"] for r in runs)
    secs = sorted(x for r in runs for x in r["secs"])
    never = sum(r["never"] for r in runs)
    print(f"\n## A busy line: {CROWD_STUDENTS} students, {args.crowds} runs "
          f"({asked} students who asked for a number and came into the camera's view)\n")
    print("| Got a number | Median | 90th pct | Within 5 s | Within 10 s | Never (60 s) "
          "| Students whose number sat on someone else | Duplicate tickets |")
    print("|---|---|---|---|---|---|---|---|")
    print(f"| {len(secs)} | {statistics.median(secs):g} s | {secs[int(0.9 * len(secs))]:g} s "
          f"| {sum(1 for x in secs if x <= 5)} | {sum(1 for x in secs if x <= 10)} | {never} "
          f"| {sum(r['wrong_body_students'] for r in runs)} "
          f"({sum(r['wrong_body_seconds'] for r in runs)} student-seconds) "
          f"| {sum(r['duplicate_tickets'] for r in runs)} |")


if __name__ == "__main__":
    main()
