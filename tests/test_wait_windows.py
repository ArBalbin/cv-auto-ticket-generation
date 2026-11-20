"""
The wait estimate counts the windows that are actually serving.

In the user acceptance test of 17 Nov 2026 three windows were set open while
one staff member worked Window 1. The estimate assumed three, so every
student was told there was no wait, while the real waits reached 139 s.
With staff calling, a window now counts only while a staff member holds it
(or while someone is being served at it). No real database.

Run:  .venv/Scripts/python.exe -m pytest tests -v
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services import queue_service  # noqa: E402
from services.queue_tracker import QueueTracker, QueueZone  # noqa: E402

SERVICE = 2.0   # minutes per transaction in these tests


@pytest.fixture
def office(monkeypatch):
    t = QueueTracker(zone=QueueZone(0, 0, 640, 480))
    t.auto_call = False          # staff press Call next, as deployed
    t._num_counters = 3          # three windows set open on the dashboard
    monkeypatch.setattr(queue_service, "queue_tracker", t)
    monkeypatch.setattr(queue_service, "_window_claims", {})
    monkeypatch.setattr(queue_service, "USE_QUEUE_ENGINE", True)
    monkeypatch.setattr(queue_service, "runtime_config",
                        lambda: {"avg_service_time": SERVICE, "active_counters": 3})
    monkeypatch.setattr(queue_service, "counters_closed_until", lambda: None)
    monkeypatch.setattr(queue_service, "invalidate_ticket_cache", lambda q=None: None)
    monkeypatch.setattr(queue_service, "update_queue_status", lambda *a, **k: None)
    return t


def wait_of(t, number):
    return queue_service.estimate_wait_for_position(t.get_position(number), 3)


def test_the_17_november_case_one_window_staffed_of_three_open(office):
    for n in (5000, 5001, 5002):
        office.force_new_person(n)
    queue_service.call_next(1, "alice")                 # Window 1 serves 5000
    assert queue_service.serving_windows(3) == [1]
    # One window: 5001 is next when 5000 is done, 5002 a service later.
    assert wait_of(office, 5001) == pytest.approx(SERVICE, abs=0.1)
    assert wait_of(office, 5002) == pytest.approx(2 * SERVICE, abs=0.1)


def test_two_staffed_windows_share_the_line(office):
    for n in (5000, 5001, 5002, 5003, 5004):
        office.force_new_person(n)
    queue_service.call_next(1, "alice")
    queue_service.call_next(2, "ben")
    assert queue_service.serving_windows(3) == [1, 2]
    assert wait_of(office, 5002) == pytest.approx(SERVICE, abs=0.1)
    assert wait_of(office, 5003) == pytest.approx(SERVICE, abs=0.1)
    assert wait_of(office, 5004) == pytest.approx(2 * SERVICE, abs=0.1)


def test_automatic_calling_counts_every_open_window(office):
    office.auto_call = True
    assert queue_service.serving_windows(3) == [1, 2, 3]


def test_with_no_window_held_one_is_assumed_rather_than_no_wait(office):
    for n in (5000, 5001, 5002):
        office.force_new_person(n)
    assert queue_service.serving_windows(3) == [1]
    assert wait_of(office, 5002) == pytest.approx(2 * SERVICE, abs=0.1), "never 'no wait'"


def test_a_window_still_serving_counts_after_its_hold_lapses(office):
    for n in (5000, 5001):
        office.force_new_person(n)
    queue_service.call_next(2, "ben")
    queue_service._window_claims.clear()                # ben's browser closed
    assert queue_service.serving_windows(3) == [2]


def test_each_ticket_records_the_windows_its_estimate_counted(office, monkeypatch):
    jobs = []
    monkeypatch.setattr(queue_service.ticket_service, "enqueue_ticket", lambda **kw: jobs.append(kw))
    office.force_new_person(5000)
    queue_service.call_next(1, "alice")
    office.force_new_person(5001)
    queue_service._on_number_linked(5001, "0m", "now", "tok", student_id=7, linked_via="face_only")
    assert jobs[0]["active_counters"] == 1
    assert jobs[0]["est_wait_min"] == pytest.approx(SERVICE, abs=0.1)


# ---------------------------------------------- real time (the 20 Nov panel's case)

@pytest.fixture
def untimed(monkeypatch):
    """No services timed yet in this test."""
    monkeypatch.setattr(queue_service, "_recent_services", queue_service.deque(maxlen=50))


def shown(t, number):
    return queue_service.prediction_for_position(t.get_position(number), 3)


def test_the_panels_case_estimates_move_while_window_1_runs_long(office, untimed):
    """Q5000 is called to Window 1 and staff take long to mark it done. The
    estimates of Q5001 and Q5002 must change with it, not stay at 2 and 4."""
    for n in (5000, 5001, 5002):
        office.force_new_person(n)
    queue_service.call_next(1, "alice")
    at_window = office._person_by_queue_number(5000)

    seen = {}
    for minutes in (0, 1, 3, 6, 12):
        at_window.called_at = datetime.now() - timedelta(minutes=minutes)
        seen[minutes] = (shown(office, 5001), shown(office, 5002))

    q2 = [seen[m][0]["estimated_wait_seconds"] for m in (0, 1, 3, 6, 12)]
    q3 = [seen[m][1]["estimated_wait_seconds"] for m in (0, 1, 3, 6, 12)]
    assert q2[1] < q2[0] and q3[1] < q3[0], "a usual transaction: the waits count down"
    assert q2[2] < q2[3] < q2[4], "the longer Window 1 takes, the longer Q5001 waits"
    assert q3[2] < q3[3] < q3[4], "and Q5002 behind it"

    # Here the clock stands still and the call moves back, so Q5001's time of
    # service is read from the call: minutes already spent plus the wait.
    from_call = [m * 60 + w for m, w in zip((0, 1, 3, 6, 12), q2)]
    assert from_call == sorted(from_call) and from_call[0] < from_call[-1], \
        "Q5001's time of service moves later, never earlier"

    late = seen[12][0]
    assert late["adjustment_note"].startswith("Window 1 has been serving Q5000 for 12 min")
    assert late["avg_service_seconds"] > seen[0][0]["avg_service_seconds"], \
        "the long transaction raises the average everyone sees"
    assert seen[0][0]["adjustment_note"] == "", "nothing to explain while Window 1 is on time"


def test_the_student_sees_minutes_seconds_and_a_clock_time(office, untimed):
    for n in (5000, 5001):
        office.force_new_person(n)
    queue_service.call_next(1, "alice")
    p = shown(office, 5001)
    assert p["estimated_wait_seconds"] == pytest.approx(SERVICE * 60, abs=2)
    assert p["estimated_wait_time_label"] in ("1 min 59 sec", "2 min 00 sec")
    assert p["estimated_service_start_label"].endswith(("AM", "PM"))
    assert p["estimated_service_start_at"].endswith("+08:00"), "on the school's clock"
    assert p["avg_service_label"] == "2 min 00 sec"
    assert p["avg_service_basis"] == "usual pace"


def test_each_service_marked_done_counts_in_the_average_at_once(office, untimed):
    for n in (5000, 5001, 5002):
        office.force_new_person(n)
    queue_service.call_next(1, "alice")
    office._person_by_queue_number(5000).called_at -= timedelta(minutes=5)
    queue_service.mark_done(5000)
    assert queue_service.recent_service_minutes() == [pytest.approx(5.0, abs=0.05)]

    queue_service.call_next(1, "alice")
    p = shown(office, 5002)
    assert p["avg_service_basis"] == "last 1 service"
    k = queue_service.queue_engine.PRIOR_WEIGHT
    assert p["avg_service_seconds"] == pytest.approx((k * SERVICE + 5.0) / (k + 1) * 60, abs=2)


def test_a_mis_click_is_not_timed(untimed):
    assert not queue_service.record_service_time(2)
    assert not queue_service.record_service_time(None)
    assert not queue_service.record_service_time(45 * 60), "a Done forgotten for 45 minutes"
    assert queue_service.recent_service_minutes() == []


def test_only_people_the_tracker_holds_are_in_line(office, untimed):
    """20 Nov 2026: after a redeploy two tickets were still 'waiting' in the
    database but nobody was in the queue area, and the app said "2 in line"
    and put every estimate behind them. The line is now the tracker's."""
    forecast = queue_service.build_queue_prediction()
    assert forecast["queue_length"] == 0
    assert forecast["new_arrival"]["estimated_wait_seconds"] == 0
    assert forecast["new_arrival"]["people_ahead"] == 0


def test_the_app_is_told_what_each_estimate_rests_on(office, untimed):
    for n in (5000, 5001):
        office.force_new_person(n)

    # Nobody holds a window: one is assumed, and the line is not moving.
    forecast = queue_service.build_queue_prediction()
    arrival = forecast["new_arrival"]
    assert arrival["people_ahead"] == 2
    assert arrival["windows_counted"] == 1 and arrival["windows_calling"] == 0
    assert forecast["windows_calling"] == 0 and forecast["line_moving"] is False
    assert arrival["estimated_wait_seconds"] == pytest.approx(2 * SERVICE * 60, abs=2), \
        "two ahead at one assumed window"

    # Window 1 calls 5000: one window calling, and the line moves.
    queue_service.call_next(1, "alice")
    forecast = queue_service.build_queue_prediction()
    assert forecast["windows_calling"] == 1 and forecast["line_moving"] is True
    mine = shown(office, 5001)
    assert mine["people_ahead"] == 0 and mine["line_moving"] is True
