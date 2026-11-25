"""
The time staff take to call the next student, counted in every wait.

In the test sessions of 20-24 Nov 2026, 14 of 20 students had nobody ahead and
were told 0:00, and staff pressed Call next about 71 s later on average. The
engine now learns that delay from the calls and adds it. No real database.

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


# ------------------------------------------------------------------ engine

def test_a_free_window_takes_the_next_student_once_staff_call():
    model = ServiceModel(mean_min=2.0, call_delay_min=1.0)
    # Free for 20 s with the student waiting: the call is due in 40 s.
    free = CounterState(ready_since=NOW - timedelta(seconds=20))
    assert queue_engine.wait_minutes(NOW, [free], 0, model) == pytest.approx(40 / 60)
    # Past the usual delay, the estimate stays at 0:00 rather than going below.
    late = CounterState(ready_since=NOW - timedelta(minutes=3))
    assert queue_engine.wait_minutes(NOW, [late], 0, model) == 0.0


def test_a_busy_window_frees_and_then_staff_call():
    model = ServiceModel(mean_min=2.0, call_delay_min=0.5)
    busy = CounterState(busy_since=NOW - timedelta(minutes=1))
    left = model.remaining(1.0)
    starts = queue_engine.start_times(NOW, [busy], 3, model)
    assert minutes(starts[0]) == pytest.approx(left + 0.5)
    # Each later turn: one average service, then the call after it.
    assert minutes(starts[1]) - minutes(starts[0]) == pytest.approx(2.0 + 0.5)
    assert minutes(starts[2]) - minutes(starts[1]) == pytest.approx(2.0 + 0.5)


def test_without_a_call_delay_nothing_changes():
    busy = CounterState(busy_since=NOW - timedelta(minutes=1))
    with_model = queue_engine.wait_minutes(NOW, [busy], 2, ServiceModel(mean_min=2.0))
    with_number = queue_engine.wait_minutes(NOW, [busy], 2, 2.0)
    assert with_model == pytest.approx(with_number)
    assert queue_engine.wait_minutes(NOW, [CounterState()], 0, 2.0) == 0.0


def test_after_lunch_staff_still_have_to_call():
    model = ServiceModel(mean_min=2.0, call_delay_min=0.5)
    closed = CounterState(closed_until=NOW + timedelta(minutes=10))
    assert queue_engine.wait_minutes(NOW, [closed], 0, model) == pytest.approx(10.5)


# ------------------------------------------------------- learning the delay

@pytest.fixture
def desk(monkeypatch):
    t = QueueTracker(zone=QueueZone(0, 0, 640, 480))
    t.auto_call = False          # the deployed mode: staff press Call next
    t._num_counters = 3
    monkeypatch.setattr(queue_service, "queue_tracker", t)
    monkeypatch.setattr(queue_service, "_window_claims", {})
    monkeypatch.setattr(queue_service, "_recent_call_gaps", deque(maxlen=50))
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


def test_with_no_calls_timed_yet_the_old_assumption_holds(desk):
    assert queue_service.current_call_delay_minutes() == 0.0


def test_a_few_slow_calls_move_it_but_one_does_not_swing_it(desk):
    queue_service.record_call_gap(60)
    one = queue_service.current_call_delay_minutes() * 60
    assert one == pytest.approx(15), "one 60 s call against three of no delay"
    for _ in range(9):
        queue_service.record_call_gap(60)
    assert queue_service.current_call_delay_minutes() * 60 == pytest.approx(600 / 13)


def test_a_break_is_not_a_call(desk):
    assert queue_service.record_call_gap(20 * 60) is False
    assert queue_service.record_call_gap(-5) is False
    assert queue_service.current_call_delay_minutes() == 0.0


def test_automatic_calling_has_no_delay(desk):
    queue_service.record_call_gap(60)
    desk.auto_call = True
    assert queue_service.current_call_delay_minutes() == 0.0


def test_each_call_is_timed_from_when_it_could_have_been_made(desk):
    queue_service.claim_window("alice", 1)
    queue_service._window_opened_at[1] = datetime.now() - timedelta(minutes=10)
    _issued(desk, 7001, seconds_ago=70)          # the student got their number 70 s ago
    queue_service.call_next(1, "alice")
    (_, gap), = queue_service._recent_call_gaps
    assert gap == pytest.approx(70, abs=2)

    # The next call is timed from the Done that freed the window, not from
    # when that student got their number much earlier.
    _issued(desk, 7002, seconds_ago=300)
    queue_service.mark_done(7001)
    queue_service._window_free_since[1] = datetime.now() - timedelta(seconds=15)
    queue_service.call_next(1, "alice")
    assert queue_service._recent_call_gaps[-1][1] == pytest.approx(15, abs=2)


def test_a_student_with_nobody_ahead_is_no_longer_told_0_00(desk):
    for _ in range(10):
        queue_service.record_call_gap(70)        # how the test sessions went
    queue_service.claim_window("alice", 1)
    _issued(desk, 7001, seconds_ago=0)
    desk._recalculate_positions()
    state = queue_service.line_state(3)
    told = queue_service.prediction_for_position(1, 3, state)
    expected = 700 / 13                           # ten 70 s calls against three of none
    assert told["estimated_wait_seconds"] == pytest.approx(expected, abs=2)
    assert told["avg_call_seconds"] == pytest.approx(expected, abs=1)
    assert told["line_moving"] is True, "the call is due, so the clock runs"

    # Twenty seconds on, still not called: the estimate has counted down.
    desk._person_by_queue_number(7001).linked_at -= timedelta(seconds=20)
    queue_service._window_opened_at[1] -= timedelta(seconds=20)
    later = queue_service.prediction_for_position(1, 3, queue_service.line_state(3))
    assert later["estimated_wait_seconds"] == pytest.approx(expected - 20, abs=2)


def test_with_no_window_held_the_clock_waits(desk):
    queue_service.record_call_gap(70)
    _issued(desk, 7001, seconds_ago=0)
    desk._recalculate_positions()
    told = queue_service.prediction_for_position(1, 3, queue_service.line_state(3))
    assert told["line_moving"] is False, "nobody is calling yet"
