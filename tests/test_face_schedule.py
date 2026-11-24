"""
Tests for the detector's face-signature budget (services/face_schedule.py),
added after people took long to get a number when several were in view.
Nothing here needs a camera or the live database.

Run:  .venv/Scripts/python.exe -m pytest tests -v
"""
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from services import queue_service  # noqa: E402
from services.face_schedule import (  # noqa: E402
    FaceSchedule,
    attach_fresh_embeddings,
    resolved_track_ids,
)
from services.queue_tracker import QueuePerson  # noqa: E402

FACE_COST = 0.26          # seconds per face signature on the office laptop


def test_new_people_come_before_people_already_identified():
    s = FaceSchedule(per_round=2)
    # 7 and 8 are identified students; 21 and 22 just walked in.
    assert s.choose([7, 8, 21, 22], resolved={7, 8}, now=100.0) == [21, 22]


def test_identified_people_are_only_refreshed_now_and_then():
    s = FaceSchedule(per_round=2, refresh_seconds=2.0)
    s.record(7, ok=True, now=100.0)
    assert s.choose([7], resolved={7}, now=101.0) == []
    assert s.choose([7], resolved={7}, now=102.5) == [7]


def test_several_newcomers_all_get_a_turn_quickly():
    """Three people walk in together while two identified students wait."""
    s = FaceSchedule(per_round=2, retry_gap=0.15)
    tracks, resolved = [5, 6, 31, 32, 33], {5, 6}
    first_face = {}
    now = 100.0
    while len(first_face) < 3:
        for tid in s.choose(tracks, resolved, now):
            now += FACE_COST
            s.record(tid, ok=True, now=now)
            first_face.setdefault(tid, now - 100.0)
    assert set(first_face) == {31, 32, 33}
    # Every newcomer is looked at within about one second. Before, each
    # frame cost all five people (1.3 s) before anyone got a second look.
    assert max(first_face.values()) <= 1.0


def test_someone_who_never_matches_cannot_starve_the_others():
    s = FaceSchedule(per_round=1, retry_gap=0.15, slow_gap=1.0, warmup=3)
    for i in range(3):                       # a walk-in, tried three times
        s.record(40, ok=False, now=100.0 + i * 0.2)
    now = 100.8
    assert s.choose([40, 41], resolved=set(), now=now) == [41]
    assert s.choose([40], resolved=set(), now=now) == [], "tried less often after warm-up"
    assert s.choose([40], resolved=set(), now=101.5) == [40]


def test_a_person_is_not_retried_faster_than_the_gap():
    s = FaceSchedule(retry_gap=0.15)
    s.record(9, ok=False, now=100.0)
    assert s.choose([9], resolved=set(), now=100.1) == []
    assert s.choose([9], resolved=set(), now=100.2) == [9]


def test_people_who_left_are_forgotten():
    s = FaceSchedule(forget_seconds=10.0)
    s.record(3, ok=True, now=100.0)
    s.forget_absent(present=set(), now=105.0)
    assert 3 in s.state
    s.forget_absent(present=set(), now=111.0)
    assert 3 not in s.state


def test_each_signature_is_sent_exactly_once():
    fresh = {21: [0.1, 0.2]}
    persons = [{"track_id": 21, "bbox": (0, 0, 9, 9)}, {"track_id": 7, "bbox": (1, 1, 8, 8)}]
    first = attach_fresh_embeddings(persons, fresh)
    assert first[0]["face_embedding"] == [0.1, 0.2]
    assert first[1]["face_embedding"] is None
    second = attach_fresh_embeddings(persons, fresh)
    assert second[0]["face_embedding"] is None, "a repeat would count as a second sighting"


def test_the_detector_reads_which_people_are_resolved():
    assert resolved_track_ids({"face_resolved": [7, 8]}) == {7, 8}
    # An older backend: only numbered people are known to be settled.
    old = {"queue_state": {"active_queue": [
        {"track_id": 7, "queue_number": 5001},
        {"track_id": 9, "queue_number": None},
    ]}}
    assert resolved_track_ids(old) == {7}
    assert resolved_track_ids({}) == set()


def test_the_backend_lists_settled_people_but_not_those_still_unidentified(monkeypatch):
    tracker = queue_service.queue_tracker
    linked = QueuePerson(track_id=1, bbox=(0, 0, 10, 20))
    linked.queue_number, linked.identity_status = 5001, "linked"
    bystander = QueuePerson(track_id=2, bbox=(0, 0, 10, 20))
    bystander.identity_status = "bystander"
    pending = QueuePerson(track_id=3, bbox=(0, 0, 10, 20))
    pending.identity_status = "pending_link"
    monkeypatch.setattr(tracker, "active_queue", {1: linked, 2: bystander, 3: pending})
    assert sorted(queue_service.face_resolved_track_ids()) == [1, 2]


def test_while_a_student_waits_for_a_number_nobody_is_slowed_down():
    # Someone tried a few times from afar, or who stood in view before
    # asking, is read again at the quick pace once a student is waiting.
    s = FaceSchedule(per_round=1, retry_gap=0.15, slow_gap=1.0, warmup=3)
    for i in range(3):
        s.record(40, ok=False, now=100.0 + i * 0.2)
    assert s.choose([40], resolved=set(), now=100.8) == []
    assert s.choose([40], resolved=set(), now=100.8, urgent=True) == [40]
    assert s.choose([40, 41], resolved=set(), now=100.8, urgent=True) == [41], \
        "still in turn, fewest tries first"
