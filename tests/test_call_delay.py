"""
The time staff take to call the next student, counted in every wait.

In the test sessions of 20-25 Nov 2026, students with nobody ahead were told
0:00 and called about a minute later. The calls logged then come in two kinds:
right after Done, with the student already waiting (about 13 s), and a student
arriving at a window that is already free (about 85 s). The engine learns each
kind from the calls, starts from the calls in the database after a restart,
and never shows 0:00 for a call that is late. No real database.

Run:  .venv/Scripts/python.exe -m pytest tests -v
"""
import sys
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services import queue_engine, queue_service  # noqa: E402
from services.queue_engine import CounterState, ServiceModel  # noqa: E402
from services.queue_tracker import QueueTracker, QueueZone  # noqa: E402

NOW = datetime(2026, 11, 25, 10, 0, 0)


def minutes(later: datetime) -> float:
    return (later - NOW).total_seconds() / 60


def wait(counter: CounterState, model) -> float:
    return queue_engine.wait_minutes(NOW, [counter], 0, model)


# ------------------------------------------------------------------ engine

def test_a_free_window_takes_the_next_student_once_staff_call():
    model = ServiceModel(mean_min=2.0, call_after_done_min=0.25, call_when_free_min=1.5)
    assert wait(CounterState(ready_since=NOW), model) == pytest.approx(1.5)
    sooner = wait(CounterState(ready_since=NOW - timedelta(seconds=30)), model)
    assert 0 < sooner < 1.5, "the call comes nearer as the student waits"


def test_a_call_later_than_usual_never_reads_0_00():
    model = ServiceModel(mean_min=2.0, call_when_free_min=1.0)
    for late in (1, 2, 4, 8):
        left = wait(CounterState(ready_since=NOW - timedelta(minutes=late)), model)
        assert left > 0.2, f"{late} min waiting for a call: still expected, not now"


def test_a_call_right_after_done_is_the_quick_kind():
    model = ServiceModel(mean_min=2.0, call_after_done_min=0.25, call_when_free_min=1.5)
    assert wait(CounterState(ready_since=NOW, ready_after_done=True), model) == pytest.approx(0.25)


def test_a_busy_window_frees_and_then_staff_call():
    model = ServiceModel(mean_min=2.0, call_after_done_min=0.25, call_when_free_min=1.5)
    busy = CounterState(busy_since=NOW - timedelta(minutes=1))
    starts = queue_engine.start_times(NOW, [busy], 3, model)
    assert minutes(starts[0]) == pytest.approx(model.remaining(1.0) + 0.25)
    # Each later turn: one average service, then Done and the quick call.
    assert minutes(starts[1]) - minutes(starts[0]) == pytest.approx(2.25)
    assert minutes(starts[2]) - minutes(starts[1]) == pytest.approx(2.25)


def test_without_call_times_nothing_changes():
    busy = CounterState(busy_since=NOW - timedelta(minutes=1))
    with_model = queue_engine.wait_minutes(NOW, [busy], 2, ServiceModel(mean_min=2.0))
    with_number = queue_engine.wait_minutes(NOW, [busy], 2, 2.0)
    assert with_model == pytest.approx(with_number)
    assert wait(CounterState(ready_since=NOW - timedelta(minutes=5)), 2.0) == 0.0


def test_after_lunch_staff_still_have_to_call():
    model = ServiceModel(mean_min=2.0, call_when_free_min=0.5)
    closed = CounterState(closed_until=NOW + timedelta(minutes=10))
    assert wait(closed, model) == pytest.approx(10.5)


# ------------------------------------------------------- learning the delays

@pytest.fixture
def desk(monkeypatch):
    t = QueueTracker(zone=QueueZone(0, 0, 640, 480))
    t.auto_call = False          # the deployed mode: staff press Call next
    t._num_counters = 3
    monkeypatch.setattr(queue_service, "queue_tracker", t)
    monkeypatch.setattr(queue_service, "_window_claims", {})
    monkeypatch.setattr(queue_service, "_calls_after_done", deque(maxlen=50))
    monkeypatch.setattr(queue_service, "_calls_when_free", deque(maxlen=50))
    monkeypatch.setattr(queue_service, "_recent_services", deque(maxlen=50))
    monkeypatch.setattr(queue_service, "_window_free_since", {})
    monkeypatch.setattr(queue_service, "_window_opened_at", {})
    monkeypatch.setattr(queue_service, "invalidate_ticket_cache", lambda q=None: None)
    monkeypatch.setattr(queue_service, "update_queue_status", lambda *a, **k: None)
    return t


def _issued(desk, number, seconds_ago):
    desk.force_new_person(number)
    person = desk._person_by_queue_number(number)
    person.linked_at = datetime.now() - timedelta(seconds=seconds_ago)
    return person


def _seconds(delays):
    return tuple(round(d * 60, 1) for d in delays)


def test_with_no_calls_timed_yet_there_is_no_delay(desk):
    assert queue_service.current_call_delays() == (0.0, 0.0)


def test_each_kind_of_call_is_learned_on_its_own(desk):
    queue_service.record_call_gap(60, after_done=False)
    assert _seconds(queue_service.current_call_delays()) == (0.0, 15.0), \
        "one 60 s call against three of no delay"
    for _ in range(9):
        queue_service.record_call_gap(60, after_done=False)
    for _ in range(10):
        queue_service.record_call_gap(13, after_done=True)
    after_done, when_free = _seconds(queue_service.current_call_delays())
    assert when_free == pytest.approx(600 / 13, abs=0.1)
    assert after_done == pytest.approx(130 / 13, abs=0.1)


def test_a_break_is_not_a_call(desk):
    assert queue_service.record_call_gap(20 * 60, after_done=False) is False
    assert queue_service.record_call_gap(-5, after_done=True) is False
    assert queue_service.current_call_delays() == (0.0, 0.0)


def test_automatic_calling_has_no_delay(desk):
    queue_service.record_call_gap(60, after_done=False)
    desk.auto_call = True
    assert queue_service.current_call_delays() == (0.0, 0.0)


def test_each_call_is_timed_from_when_it_could_have_been_made(desk):
    queue_service.claim_window("alice", 1)
    queue_service._window_opened_at[1] = datetime.now() - timedelta(minutes=10)
    _issued(desk, 7001, seconds_ago=70)          # arrived at a free window 70 s ago
    queue_service.call_next(1, "alice")
    assert list(queue_service._calls_when_free) == [pytest.approx(70, abs=2)]

    # The next call is timed from the Done that freed the window, not from
    # when that student got their number much earlier, and is the quick kind.
    _issued(desk, 7002, seconds_ago=300)
    queue_service.mark_done(7001)
    queue_service._window_free_since[1] = datetime.now() - timedelta(seconds=15)
    queue_service.call_next(1, "alice")
    assert list(queue_service._calls_after_done) == [pytest.approx(15, abs=2)]


def test_a_student_with_nobody_ahead_is_not_told_0_00(desk):
    for _ in range(10):
        queue_service.record_call_gap(85, after_done=False)
        queue_service.record_call_gap(13, after_done=True)
    queue_service.claim_window("alice", 1)
    _issued(desk, 7001, seconds_ago=0)
    desk._recalculate_positions()
    told = queue_service.prediction_for_position(1, 3, queue_service.line_state(3))
    expected = 850 / 13
    assert told["estimated_wait_seconds"] == pytest.approx(expected, abs=2)
    assert told["avg_call_when_free_seconds"] == pytest.approx(expected, abs=1)
    assert told["avg_call_after_done_seconds"] == pytest.approx(10, abs=1)
    assert told["line_moving"] is True, "the call is due, so the clock runs"

    # Three minutes on and still not called: later than usual, but the clock
    # still shows a time, not 0:00.
    desk._person_by_queue_number(7001).linked_at -= timedelta(minutes=3)
    queue_service._window_opened_at[1] -= timedelta(minutes=3)
    late = queue_service.prediction_for_position(1, 3, queue_service.line_state(3))
    assert late["estimated_wait_seconds"] > 15


def test_with_no_window_held_the_clock_waits(desk):
    queue_service.record_call_gap(70, after_done=False)
    _issued(desk, 7001, seconds_ago=0)
    desk._recalculate_positions()
    told = queue_service.prediction_for_position(1, 3, queue_service.line_state(3))
    assert told["line_moving"] is False, "nobody is calling yet"


def test_after_a_restart_the_call_times_come_from_the_database(desk, monkeypatch):
    newest_first = [(248, False), (29, True), (6, True), (93, False), (400, False)]
    monkeypatch.setattr(queue_service, "fetch_recent_call_gaps", lambda: newest_first)
    monkeypatch.setattr(queue_service, "_calls_seeded", False)
    queue_service.seed_recent_calls()
    assert list(queue_service._calls_when_free) == [93, 248], "oldest first; the 400 s break dropped"
    assert list(queue_service._calls_after_done) == [6, 29]
    queue_service.seed_recent_calls()
    assert len(queue_service._calls_when_free) == 2, "read once"
