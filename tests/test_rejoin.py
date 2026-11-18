"""
A student served (or skipped) and still in front of the camera can ask for
another number and get it without walking away first.

Before, their finished entry stayed on their camera track until they left
the view, and for 10 s nobody new was confirmed where they stood, so a
student who wanted a second number had to walk out of view and wait. Those
guards remain for everyone who does not ask again. No real database.

Run:  .venv/Scripts/python.exe -m pytest tests -v
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.queue_tracker import QueueTracker, QueueZone  # noqa: E402

STUDENT = 42
BOX = (100, 80, 220, 420)
FACE = (np.ones(512, dtype=np.float32) / np.sqrt(512)).tolist()


def tracker():
    t = QueueTracker(zone=QueueZone(0, 0, 640, 480))
    t.MIN_CONFIRM_FRAMES = 3           # confirm by count, so the test needs no waiting
    t.MIN_CONFIRM_SECONDS = 0
    return t


def seen(t, times=1, track=7):
    for _ in range(times):
        t.process_frame([{"track_id": track, "bbox": BOX, "conf": 0.9, "face_embedding": FACE}])


def served_and_still_here():
    t = tracker()
    seen(t, 3)                                            # presence confirmed
    assert t.active_queue[7].identity_status == "pending_link"
    person = t.mint_face_only_number(7, STUDENT)          # face matched: number issued
    t.mark_transaction_done(person.queue_number)          # staff press Done
    seen(t, 2)                                            # still standing there
    assert t.active_queue[7].status == "done_pending"
    return t


def test_without_asking_again_a_served_student_is_not_requeued():
    t = served_and_still_here()
    seen(t, 5)
    assert t.active_queue[7].status == "done_pending", "the guard still holds"
    assert 7 not in t._candidates


def test_asking_again_releases_them_at_once():
    t = served_and_still_here()
    t.arm_join_intent(STUDENT)                            # taps Get a Queue Number
    seen(t, 1)
    assert t.active_queue.get(7) is None or t.active_queue[7].status != "done_pending"
    assert t.get_person_by_student_id(STUDENT) is None
    seen(t, 2)                                            # confirmed again, like anyone arriving
    assert t.active_queue[7].identity_status == "pending_link"
    assert t.active_queue[7].status != "done_pending"


def test_the_cooldown_where_they_stood_is_lifted_only_for_them():
    t = served_and_still_here()
    assert t._is_in_done_cooldown(BOX)
    t.arm_join_intent(STUDENT)
    seen(t, 1)
    assert not t._is_in_done_cooldown(BOX)


def test_asking_while_already_waiting_changes_nothing():
    t = tracker()
    seen(t, 3)
    person = t.mint_face_only_number(7, STUDENT)
    t.arm_join_intent(STUDENT)
    seen(t, 2)
    assert t.active_queue[7].queue_number == person.queue_number
