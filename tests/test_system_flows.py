"""
End-to-end tests of the student (mobile app) and cashier (staff dashboard)
flows, run against the real queue logic with no database, network or camera.

Detector frames go through queue_service.process_tracked_persons() - the same
entry point the live detector pushes to - so presence confirmation, face
matching and number issuance run exactly as deployed. The student app's
endpoints are called directly, and staff actions go through the same service
functions the dashboard's routes call.

Nothing here touches the live database. That matters: face_match_events there
is the source of the SOP 2 results, and test rows would contaminate it. Every
database write is replaced with an in-memory stub, and enrolled students are
synthetic face vectors, not anyone's real face.

Run:  .venv/Scripts/python.exe -m pytest tests -v
"""
import asyncio
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from core.config import CONSENT_VERSION  # noqa: E402
from services import queue_service  # noqa: E402
from services.queue_tracker import QueueTracker, QueueZone  # noqa: E402
from services.embedding_index import EmbeddingIndex  # noqa: E402
from routers import students  # noqa: E402

# Settings copied from the deployed tracker so the tests exercise the real
# thresholds (0.30 / 0.15, 20-frame confirmation) rather than class defaults.
DEPLOYED_SETTINGS = (
    "MAX_MISSING_FRAMES", "MIN_CONFIRM_FRAMES", "MIN_CONFIRM_SECONDS",
    "MIN_CONFIRM_FACE_FRAMES", "MIN_MOTION_PIXELS",
    "STATIC_STDEV_THRESHOLD", "STATIC_CONF_BYPASS_THRESHOLD",
    "MIN_PORTRAIT_ASPECT", "FACE_MATCH_THRESHOLD", "FACE_MARGIN_THRESHOLD",
    "PENDING_LINK_TIMEOUT_SECONDS", "DEDUP_IOU_THRESH", "DEDUP_CENTRE_FRAC",
    "_num_counters",
)


# ---------------------------------------------------------------- helpers

def face(seed: int) -> np.ndarray:
    """A synthetic, normalised 512-d face embedding."""
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def near(v: np.ndarray, seed: int, noise: float) -> np.ndarray:
    """Another capture of the same face: the embedding plus a little noise."""
    n = np.random.default_rng(seed).normal(size=v.shape).astype(np.float32)
    w = v + noise * n / np.linalg.norm(n)
    return w / np.linalg.norm(w)


def frame(*people):
    """One detector payload. Each person is (track_id, x_offset, embedding)."""
    return [
        {"track_id": tid, "bbox": [x, 100, x + 110, 420], "conf": 0.9,
         "face_embedding": None if emb is None else emb.tolist()}
        for tid, x, emb in people
    ]


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def system(monkeypatch):
    fresh = QueueTracker(zone=QueueZone(x1=0, y1=0, x2=1920, y2=1080))
    for name in DEPLOYED_SETTINGS:
        setattr(fresh, name, getattr(queue_service.queue_tracker, name))
    fresh._num_counters = 3
    fresh.on_number_linked = queue_service._on_number_linked
    monkeypatch.setattr(queue_service, "queue_tracker", fresh)

    enrolled: dict[int, np.ndarray] = {}
    events: list[tuple] = []
    tickets: list[int] = []
    monkeypatch.setattr(queue_service, "get_all_active_embeddings",
                        lambda: [(sid, v.tolist()) for sid, v in enrolled.items()])
    # Every synthetic student has agreed to the data-privacy notice; joining
    # the queue checks for it.
    monkeypatch.setattr(students, "get_student_by_id",
                        lambda sid: {"id": sid, "consent_version": CONSENT_VERSION})
    monkeypatch.setattr(students, "touch_student_activity", lambda sid: None)
    # Tests enrol students as they go, so the index must reload every time.
    monkeypatch.setattr(queue_service, "embedding_index", EmbeddingIndex(
        loader=lambda: [(sid, v.tolist()) for sid, v in enrolled.items()], ttl_seconds=0))
    monkeypatch.setattr(queue_service, "record_face_match_event",
                        lambda *a, **k: events.append(a))
    monkeypatch.setattr(queue_service, "record_recognition_metric", lambda *a, **k: None)
    monkeypatch.setattr(queue_service, "update_queue_status", lambda *a, **k: None)
    monkeypatch.setattr(queue_service.ticket_service, "enqueue_ticket",
                        lambda **k: tickets.append(k["queue_number"]))

    def stand(*people, frames: int = 30):
        for i in range(frames):
            queue_service.process_tracked_persons(frame(*people), i)

    return SimpleNamespace(tracker=fresh, enrolled=enrolled, events=events,
                           tickets=tickets, stand=stand)


def app_state(student_id: int) -> dict:
    """What the mobile app's home screen polls."""
    return run(students.get_my_queue_entry(student_id=student_id))


def tap_join(student_id: int) -> dict:
    """The mobile app's Join the Queue button."""
    return run(students.join_queue(student_id=student_id))


# ------------------------------------------------ student / mobile app flow

def test_student_who_joined_is_issued_a_number_without_manual_input(system):
    ana = face(1)
    system.enrolled[101] = ana
    tap_join(101)

    system.stand((1, 100, near(ana, 11, 0.3)))

    state = app_state(101)
    assert state["state"] == "active"
    assert state["queue_number"] >= 5000, "system-issued numbers use their own range"
    person = system.tracker.get_person_by_student_id(101)
    assert person.linked_via == "face_only", "no kiosk ticket and no staff input"
    assert system.tickets == [state["queue_number"]], "a ticket is generated"


def test_a_face_in_the_zone_for_a_second_is_confirmed_without_the_full_count(system):
    # The cloud backend gets 2-3 updates a second, so the 14-update count
    # alone took 5-7 s. With a face seen, one second and three updates do.
    ana = face(3)
    for i in range(3):
        queue_service.process_tracked_persons(frame((7, 100, near(ana, 70 + i, 0.3))), i)
    assert 7 not in system.tracker.active_queue, "under a second: not yet"

    system.tracker._candidates[7]["first_seen"] -= timedelta(seconds=1.5)
    queue_service.process_tracked_persons(frame((7, 100, near(ana, 79, 0.3))), 3)
    assert 7 in system.tracker.active_queue


def test_several_students_arriving_together_each_get_a_number_and_ticket(system):
    sids = [140, 141, 142, 143]
    faces = {sid: face(sid) for sid in sids}
    system.enrolled.update(faces)
    for sid in sids:
        tap_join(sid)

    def people(update):
        return [(60 + i, 100 + 400 * i, near(faces[sid], 1000 * update + i, 0.3))
                for i, sid in enumerate(sids)]

    for k in range(3):
        queue_service.process_tracked_persons(frame(*people(k)), k)
    assert not system.tickets, "under a second: nobody confirmed yet"

    for candidate in system.tracker._candidates.values():
        candidate["first_seen"] -= timedelta(seconds=1.5)
    queue_service.process_tracked_persons(frame(*people(3)), 3)

    numbers = [app_state(sid)["queue_number"] for sid in sids]
    assert None not in numbers
    assert len(set(numbers)) == 4, "each student gets their own number"
    assert sorted(system.tickets) == sorted(numbers), "one ticket each"
    assert sorted(system.tracker.get_position(n) for n in numbers) == [1, 2, 3, 4]


def test_a_face_and_a_faceless_detection_together(system):
    # A student facing the camera is confirmed at once; someone with their
    # back turned in the same frames waits for the full count, then joins too.
    ana = face(144)
    system.enrolled[144] = ana
    tap_join(144)
    for k in range(3):
        queue_service.process_tracked_persons(frame((70, 100, near(ana, 2000 + k, 0.3)), (71, 900, None)), k)
    for candidate in system.tracker._candidates.values():
        candidate["first_seen"] -= timedelta(seconds=1.5)
    queue_service.process_tracked_persons(frame((70, 100, near(ana, 2003, 0.3)), (71, 900, None)), 3)

    assert app_state(144)["queue_number"] is not None
    assert 71 not in system.tracker.active_queue

    for k in range(4, system.tracker.MIN_CONFIRM_FRAMES + 1):
        queue_service.process_tracked_persons(frame((70, 100, near(ana, 2000 + k, 0.3)), (71, 900, None)), k)
    assert 71 in system.tracker.active_queue, "confirmed by count; staff link a walk-in by hand"


def test_served_student_who_steps_away_and_rejoins_gets_a_new_number(system):
    ana = face(150)
    system.enrolled[150] = ana
    tap_join(150)
    system.stand((80, 100, near(ana, 3000, 0.3)))
    first = app_state(150)["queue_number"]
    queue_service.mark_done(first)

    # Still standing there: still the served person, no new number.
    system.stand((80, 100, near(ana, 3001, 0.3)), frames=5)
    assert system.tracker.get_person_by_student_id(150).status == "done_pending"

    # Out of view for a second (two or three updates on the cloud backend).
    system.tracker.get_person_by_student_id(150).last_seen -= timedelta(seconds=1.5)
    system.stand(frames=3)
    assert system.tracker.get_person_by_student_id(150) is None, "gone after a second"

    # Back in the same spot before the cooldown ends: not taken for someone new.
    tap_join(150)
    system.stand((81, 100, near(ana, 3002, 0.3)), frames=5)
    assert system.tickets == [first]

    # Ten seconds after being served, the spot is free again.
    for entry in system.tracker._done_cooldowns:
        entry["until"] -= timedelta(seconds=11)
    system.stand((81, 100, near(ana, 3003, 0.3)))
    second = app_state(150)["queue_number"]
    assert second is not None and second != first
    assert system.tickets == [first, second]


def test_a_skipped_student_is_told_so_by_the_app(system, monkeypatch):
    from routers import queue as queue_router

    ana = face(151)
    system.enrolled[151] = ana
    tap_join(151)
    system.stand((90, 100, near(ana, 4000, 0.3)))
    number = app_state(151)["queue_number"]
    person = system.tracker.get_person_by_student_id(151)
    token = person.short_code or person.access_token

    queue_service.mark_no_show(number)

    # The database row is written in the background; until it says no_show,
    # the tracker's answer must already say so.
    monkeypatch.setattr(queue_router, "get_db_pool", lambda: object())
    monkeypatch.setattr(queue_router, "get_ticket_record_by_short_code",
                        lambda t, q, pool: {"queue_number": q, "status": "waiting"})
    status = queue_router.queue_status(number, token)
    assert status["status"] == "no_show"
    assert "skipped" in status["message"]

    # Once the row says no_show, the record's own answer says the same.
    monkeypatch.setattr(queue_router, "get_ticket_record_by_short_code",
                        lambda t, q, pool: {"queue_number": q, "status": "no_show"})
    status = queue_router.queue_status(number, token)
    assert status["status"] == "no_show" and "skipped" in status["message"]

    # The dashboard screen of the app shows the skipped card, not "just served".
    assert app_state(151)["state"] == "recently_skipped"


def test_a_served_student_is_still_reported_as_served(system):
    ana = face(152)
    system.enrolled[152] = ana
    tap_join(152)
    system.stand((91, 100, near(ana, 4100, 0.3)))
    queue_service.mark_done(app_state(152)["queue_number"])
    assert app_state(152)["state"] == "recently_served"
    assert system.tracker.get_person_by_student_id(152).to_dict()["status"] == "done_pending"


def test_without_a_face_the_full_count_still_applies(system):
    for i in range(3):
        queue_service.process_tracked_persons(frame((8, 400, None)), i)
    system.tracker._candidates[8]["first_seen"] -= timedelta(seconds=5)
    queue_service.process_tracked_persons(frame((8, 400, None)), 3)
    assert 8 not in system.tracker.active_queue, "time alone is not enough without a face"

    for i in range(4, system.tracker.MIN_CONFIRM_FRAMES + 1):
        queue_service.process_tracked_persons(frame((8, 400, None)), i)
    assert 8 in system.tracker.active_queue


def test_app_shows_the_right_state_before_and_after_joining(system):
    system.enrolled[102] = face(2)
    assert app_state(102)["state"] == "not_joined"
    tap_join(102)
    assert app_state(102)["state"] == "waiting_for_camera"


def test_recognised_without_joining_gets_no_number_until_the_student_joins(system):
    ben = face(3)
    system.enrolled[103] = ben
    system.stand((2, 100, near(ben, 13, 0.3)))

    assert system.tracker.get_person_by_student_id(103) is None
    assert app_state(103)["state"] == "recognized_not_joined"
    assert system.tickets == [], "no request, no number"

    tap_join(103)
    # Joining revives the recognised track straight back to pending, so the
    # app shows "Confirming your identity". The "rejoining" state in
    # routers/students.py is never reached for this reason: it waits for a
    # bystander that arm_join_intent() has already cleared.
    assert app_state(103)["state"] == "identifying"
    system.stand((2, 100, near(ben, 14, 0.3)), frames=3)
    assert app_state(103)["state"] == "active", "joining revives the same track"


def test_person_not_enrolled_is_refused_and_raised_to_staff(system, monkeypatch):
    system.enrolled[104] = face(4)
    tap_join(104)
    system.tracker.PENDING_LINK_TIMEOUT_SECONDS = 0

    system.stand((3, 100, face(99)))

    assert system.tickets == [], "a stranger is never given a number"
    assert all(accepted is False for *_, accepted in system.events)
    alerts = system.tracker.get_pending_link_alerts()
    assert len(alerts) == 1, "staff are alerted instead of the system guessing"


def test_two_lookalike_students_are_refused_by_the_margin_rule(system):
    twin = face(5)
    system.enrolled[105] = twin
    system.enrolled[106] = near(twin, 15, 0.05)  # nearly identical enrolment
    tap_join(105)
    tap_join(106)

    system.stand((4, 100, near(twin, 16, 0.2)))

    scores = [(score, margin) for _, _, score, margin, _ in system.events]
    assert scores and all(score >= 0.30 for score, _ in scores), "the score alone would accept"
    assert all(margin < 0.15 for _, margin in scores), "but no candidate leads clearly"
    assert system.tickets == [], "so neither student is issued a number"


def test_a_student_holds_only_one_number(system):
    cy = face(6)
    system.enrolled[107] = cy
    tap_join(107)
    system.stand((5, 100, near(cy, 17, 0.3)))
    first = app_state(107)["queue_number"]

    tap_join(107)
    system.stand((5, 100, near(cy, 18, 0.3)))

    assert app_state(107)["queue_number"] == first
    assert system.tickets == [first]


def test_position_moves_up_when_the_student_ahead_is_served(system):
    a, b = face(7), face(8)
    system.enrolled[108], system.enrolled[109] = a, b
    tap_join(108)
    tap_join(109)
    system.stand((6, 100, near(a, 19, 0.3)), (7, 700, near(b, 20, 0.3)))

    qa = system.tracker.get_person_by_student_id(108).queue_number
    qb = system.tracker.get_person_by_student_id(109).queue_number
    assert system.tracker.get_position(qb) == 2

    queue_service.mark_done(qa)
    assert system.tracker.get_position(qb) == 1


def test_new_person_in_a_vacated_spot_gets_their_own_number(system):
    # Regression: the second student used to inherit the first one's number
    # because the track remap matched on position alone.
    a, b = face(20), face(21)
    system.enrolled[120], system.enrolled[121] = a, b
    tap_join(120)
    tap_join(121)
    system.stand((30, 100, near(a, 22, 0.3)))
    qa = app_state(120)["queue_number"]

    system.stand(frames=10)                      # the first student steps away
    system.stand((31, 100, near(b, 23, 0.3)))    # the next one takes the same spot

    qb = app_state(121)["queue_number"]
    assert app_state(121)["state"] == "active"
    assert qb != qa, "the second student gets a new number"
    assert system.tracker.get_person_by_student_id(120).queue_number == qa
    assert system.tickets == [qa, qb]


def test_same_person_on_a_new_track_keeps_their_number(system):
    a = face(24)
    system.enrolled[122] = a
    tap_join(122)
    system.stand((40, 100, near(a, 25, 0.3)))
    q = app_state(122)["queue_number"]

    system.stand(frames=3)                       # tracker briefly loses them
    system.stand((41, 100, near(a, 26, 0.3)))    # and gives them a new track id

    assert app_state(122)["queue_number"] == q
    assert system.tickets == [q], "no second number for the same student"


def test_two_students_standing_close_each_get_a_number(system):
    a, b = face(27), face(28)
    system.enrolled[123], system.enrolled[124] = a, b
    tap_join(123)
    tap_join(124)
    system.stand((50, 100, near(a, 29, 0.3)))
    # The second student stands right behind the first: their boxes overlap.
    system.stand((50, 100, near(a, 30, 0.3)), (51, 130, near(b, 31, 0.3)))

    qa, qb = app_state(123)["queue_number"], app_state(124)["queue_number"]
    assert qa is not None and qb is not None and qa != qb


def _five_joined_students(system, base):
    sids = [base + i for i in range(5)]
    faces = {sid: face(sid) for sid in sids}
    system.enrolled.update(faces)
    for sid in sids:
        tap_join(sid)
    return sids, faces


def _assert_five_tickets(system, sids):
    numbers = [app_state(sid)["queue_number"] for sid in sids]
    assert all(app_state(sid)["state"] == "active" for sid in sids), numbers
    assert len(set(numbers)) == 5, f"five different numbers expected, got {numbers}"
    assert sorted(system.tickets) == sorted(numbers), "one ticket per student"


def test_five_students_side_by_side_get_five_tickets(system):
    sids, faces = _five_joined_students(system, 300)
    system.stand(*[(60 + i, 100 + 250 * i, near(faces[s], 70 + i, 0.3)) for i, s in enumerate(sids)])
    _assert_five_tickets(system, sids)


def test_five_students_forming_a_line_get_five_tickets(system):
    # They arrive one at a time and stand close behind each other, so their
    # boxes overlap heavily, as in a real queue seen from the camera.
    sids, faces = _five_joined_students(system, 310)
    line = []
    for i, s in enumerate(sids):
        line.append((80 + i, 100 + 40 * i, near(faces[s], 90 + i, 0.3)))
        system.stand(*line)
    _assert_five_tickets(system, sids)


def test_five_students_taking_the_same_spot_in_turn_get_five_tickets(system):
    # Each one steps into the spot the previous one just left.
    sids, faces = _five_joined_students(system, 320)
    for i, s in enumerate(sids):
        system.stand((100 + i, 100, near(faces[s], 110 + i, 0.3)))
        system.stand(frames=10)
    _assert_five_tickets(system, sids)


# ------------------------------------------------ cashier / staff dashboard

def test_staff_mark_done_counts_as_served(system):
    q = system.tracker.force_new_person(5200)["queue_number"]
    assert queue_service.mark_done(q) is not None
    assert system.tracker.total_served == 1
    assert system.tracker.completed_queue[-1]["bump_reason"] == "served"


def test_staff_no_show_is_recorded_but_not_counted_as_served(system):
    q = system.tracker.force_new_person(5201)["queue_number"]
    assert queue_service.mark_no_show(q) is not None
    assert system.tracker.total_served == 0, "a no-show was never served"
    assert system.tracker.completed_queue[-1]["bump_reason"] == "no_show"


def test_done_on_an_unknown_number_is_rejected(system):
    assert queue_service.mark_done(9999) is None
    assert queue_service.mark_no_show(9999) is None


def test_emergency_manual_entry_and_duplicate_rejection(system):
    entry = queue_service.force_new_person(5300, is_walkin=False)
    assert entry is not None and entry["is_manual"]
    assert queue_service.force_new_person(5300) is None, "a number is used once per day"


def test_staff_can_link_a_person_the_camera_could_not_identify(system):
    system.tracker.PENDING_LINK_TIMEOUT_SECONDS = 0
    system.stand((8, 100, face(98)))
    track_id = system.tracker.get_pending_link_alerts()[0]["track_id"]

    linked = queue_service.link_pending_person(track_id, 5400)
    assert linked is not None and linked["queue_number"] == 5400
    assert system.tracker.get_pending_link_alerts() == []


def test_counters_go_to_the_front_of_the_queue_and_stay_assigned(system):
    numbers = [system.tracker.force_new_person(n)["queue_number"] for n in (5501, 5502, 5503, 5504)]
    counters = {p["queue_number"]: p["counter_number"] for p in system.tracker.get_state()["active_queue"]}
    assert [counters[n] for n in numbers[:3]] == [1, 2, 3]
    assert counters[numbers[3]] is None, "only three counters are open"

    queue_service.mark_done(numbers[1])  # counter 2 frees up
    counters = {p["queue_number"]: p["counter_number"] for p in system.tracker.get_state()["active_queue"]}
    assert counters[numbers[0]] == 1 and counters[numbers[2]] == 3, "assignments are sticky"
    assert counters[numbers[3]] == 2, "the next student takes the freed counter"


# ------------------------------------------------- queue engine (services/queue_engine.py)

def service_minutes() -> float:
    return queue_service.runtime_config()["avg_service_time"]


def test_people_at_a_counter_wait_nothing_and_the_rest_wait_their_turn(system, monkeypatch):
    monkeypatch.setattr(queue_service, "USE_QUEUE_ENGINE", True)
    for n in (5801, 5802, 5803, 5804, 5805, 5806, 5807):
        system.tracker.force_new_person(n)
    service = service_minutes()
    waits = [queue_service.estimate_wait_for_position(p, 3) for p in range(1, 9)]

    assert waits[:3] == [0.0, 0.0, 0.0], "three counters, three people at them"
    assert waits[3] == waits[4] == waits[5] == pytest.approx(service, abs=0.1), "next three take the counters as they free"
    assert waits[6] == pytest.approx(2 * service, abs=0.1)
    assert waits[7] == pytest.approx(2 * service, abs=0.1), "the next arrival is fifth in the waiting line"


def test_a_long_line_is_not_under_estimated(system, monkeypatch):
    monkeypatch.setattr(queue_service, "USE_QUEUE_ENGINE", True)
    for n in range(5900, 5933):                     # 33 in line: 3 at counters, 30 waiting
        system.tracker.force_new_person(n)
    service = service_minutes()
    # The 31st waiting person goes after ten full rounds of three counters.
    assert queue_service.estimate_wait_for_position(34, 3) == pytest.approx(11 * service, abs=0.1)


def test_fewer_counters_mean_a_longer_wait(system, monkeypatch):
    monkeypatch.setattr(queue_service, "USE_QUEUE_ENGINE", True)
    for n in range(6000, 6010):
        system.tracker.force_new_person(n)
    assert queue_service.estimate_wait_for_position(11, 1) > queue_service.estimate_wait_for_position(11, 3)


def test_nobody_is_called_while_counters_are_closed_for_lunch(system, monkeypatch):
    monkeypatch.setattr(queue_service, "USE_QUEUE_ENGINE", True)
    monkeypatch.setattr(queue_service, "_counters_closed_until", None)
    reopen = datetime.now() + timedelta(minutes=30)
    queue_service.set_counters_closed_until(reopen)

    for n in (6101, 6102):
        system.tracker.force_new_person(n)
    line = system.tracker.line_snapshot()
    assert all(q["counter_number"] is None for q in line), "no one is called during the closure"
    assert queue_service.estimate_wait_for_position(1, 3) == pytest.approx(30, abs=0.2)

    queue_service.set_counters_closed_until(None)
    line = system.tracker.line_snapshot()
    assert all(q["counter_number"] is not None for q in line), "reopening calls them at once"
    assert queue_service.estimate_wait_for_position(1, 3) == 0.0


def test_a_queue_reset_keeps_the_counters_closed(system, monkeypatch):
    monkeypatch.setattr(queue_service, "_counters_closed_until", None)
    monkeypatch.setattr(queue_service, "delete_all_tickets", lambda: 0)
    monkeypatch.setattr(queue_service, "record_queue_reset", lambda *a: None)
    reopen = datetime.now() + timedelta(minutes=30)
    queue_service.set_counters_closed_until(reopen)
    queue_service.reset_queue()
    try:
        assert queue_service.counters_closed_until() == reopen
    finally:
        queue_service.set_counters_closed_until(None)


def test_the_dashboard_forecast_comes_from_the_engine(system, monkeypatch):
    monkeypatch.setattr(queue_service, "USE_QUEUE_ENGINE", True)
    for n in range(6200, 6206):
        system.tracker.force_new_person(n)
    monkeypatch.setitem(queue_service.state.latest_state, "arrival_rate", 1.0)   # one more person a minute

    forecast = queue_service.build_queue_prediction()
    assert forecast["wait_estimator"] == "queue-engine"
    now = forecast["forecast"]["now"]["estimated_wait_time_minutes"]
    later = [forecast["forecast"][f"in_{h}min"]["estimated_wait_time_minutes"] for h in (5, 15, 30)]
    assert now == forecast["new_arrival"]["estimated_wait_time_minutes"]
    assert now <= later[0] <= later[1] <= later[2], "a growing line means longer waits later"


# ------------------------------------------------- wait log (WAIT_TIME_MODEL.md)

def test_each_ticket_logs_the_line_it_joined_and_its_estimate(system, monkeypatch):
    jobs = []
    monkeypatch.setattr(queue_service.ticket_service, "enqueue_ticket", lambda **k: jobs.append(k))
    for n in (5601, 5602, 5603, 5604):
        system.tracker.force_new_person(n)

    assert [j["people_ahead"] for j in jobs] == [0, 1, 2, 3]
    assert [j["active_counters"] for j in jobs] == [3, 3, 3, 3]
    assert jobs[3]["est_wait_min"] == queue_service.estimate_wait_for_position(4, 3)


def test_staff_actions_log_when_the_person_was_called(system, monkeypatch):
    saved = {}
    finished = threading.Event()

    def record(queue_number, status, actor, called_seconds_ago=None, cashier_window=None):
        saved[queue_number] = (called_seconds_ago, cashier_window)
        if len(saved) == 2:
            finished.set()

    monkeypatch.setattr(queue_service, "_update_status_then_invalidate", record)
    numbers = [system.tracker.force_new_person(n)["queue_number"] for n in (5701, 5702, 5703, 5704)]
    assert system.tracker.get_called_at(numbers[3]) is None, "fourth in line, no counter yet"

    queue_service.mark_no_show(numbers[3])      # never called
    queue_service.mark_done(numbers[0])         # called the moment a counter was free
    assert finished.wait(2)
    assert saved[numbers[3]] == (None, None)
    called_seconds_ago, window = saved[numbers[0]]
    assert called_seconds_ago is not None and window == 1
