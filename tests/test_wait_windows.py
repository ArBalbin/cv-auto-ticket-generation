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
