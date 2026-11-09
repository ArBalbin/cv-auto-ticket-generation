"""
Staff call each number with Call next, from the window their account holds.
No real database.

Run:  .venv/Scripts/python.exe -m pytest tests -v
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from routers import queue as queue_router  # noqa: E402
from services import queue_service  # noqa: E402
from services.queue_tracker import QueueTracker, QueueZone  # noqa: E402


@pytest.fixture
def tracker(monkeypatch):
    t = QueueTracker(zone=QueueZone(0, 0, 640, 480))
    t.auto_call = False          # the deployed mode
    t._num_counters = 3
    monkeypatch.setattr(queue_service, "queue_tracker", t)
    monkeypatch.setattr(queue_service, "_window_claims", {})
    monkeypatch.setattr(queue_service, "invalidate_ticket_cache", lambda q=None: None)
    monkeypatch.setattr(queue_service, "update_queue_status", lambda *a, **k: None)
    return t


def counters(t):
    return {p["queue_number"]: p["counter_number"] for p in t.get_state()["active_queue"]}


def test_nobody_is_called_until_staff_press_call_next(tracker):
    for n in (7001, 7002, 7003):
        tracker.force_new_person(n)
    assert set(counters(tracker).values()) == {None}, "free windows wait for staff"


def test_call_next_sends_the_first_in_line_to_that_window(tracker):
    for n in (7001, 7002, 7003):
        tracker.force_new_person(n)

    outcome, person = tracker.call_next(2)
    assert outcome == "called" and person["queue_number"] == 7001
    assert person["counter_number"] == 2 and person["seconds_at_counter"] == 0
    assert tracker.get_state()["newly_called"] == [
        {"queue_number": 7001, "queue_label": "Q7001", "counter_number": 2}], "the board announces it"

    assert tracker.call_next(2)[0] == "busy", "finish the person at the window first"
    assert tracker.call_next(1)[1]["queue_number"] == 7002

    tracker.mark_transaction_done(7001)
    assert tracker.call_next(2)[1]["queue_number"] == 7003
    assert counters(tracker) == {7002: 1, 7003: 2}


def test_call_next_when_nobody_waits_closed_or_no_such_window(tracker):
    assert tracker.call_next(1) == ("empty", None)
    assert tracker.call_next(4)[0] == "invalid"
    tracker.force_new_person(7004)
    tracker.counters_closed_until = datetime.now() + timedelta(minutes=30)
    assert tracker.call_next(1)[0] == "closed"


def test_auto_mode_still_calls_by_itself(tracker):
    tracker.auto_call = True
    for n in (7001, 7002):
        tracker.force_new_person(n)
    assert counters(tracker) == {7001: 1, 7002: 2}


def test_one_window_per_account_and_one_account_per_window(tracker):
    queue_service.claim_window("alice", 1)
    with pytest.raises(queue_service.WindowTaken) as taken:
        queue_service.claim_window("bob", 1)
    assert "alice" in str(taken.value)

    queue_service.claim_window("bob", 2)
    queue_service.claim_window("alice", 3)             # alice moves; window 1 is free again
    queue_service.claim_window("carol", 1)
    overview = queue_service.window_overview("bob")
    assert overview["my_window"] == 2 and overview["call_mode"] == "staff"
    assert [w["claimed_by"] for w in overview["windows"]] == ["carol", "bob", "alice"]

    queue_service.release_window("bob")
    assert queue_service.window_overview("bob")["my_window"] is None

    with pytest.raises(ValueError):
        queue_service.claim_window("dave", 9)


def test_a_window_left_open_in_a_closed_browser_frees_itself(tracker, monkeypatch):
    queue_service.claim_window("alice", 1)
    stale = queue_service._window_claims[1][1] - queue_service.WINDOW_CLAIM_TTL_SECONDS - 1
    queue_service._window_claims[1] = ("alice", stale)
    queue_service.claim_window("bob", 1)                # no error: alice's claim lapsed


def test_the_call_next_route(tracker):
    for n in (7001, 7002):
        tracker.force_new_person(n)

    called = queue_router.queue_call_next(queue_router.WindowBody(counter_number=1), username="alice")
    assert called["called"]["queue_number"] == 7001

    with pytest.raises(HTTPException) as other_account:
        queue_router.queue_call_next(queue_router.WindowBody(counter_number=1), username="bob")
    assert other_account.value.status_code == 409 and "alice" in other_account.value.detail

    with pytest.raises(HTTPException) as busy:
        queue_router.queue_call_next(queue_router.WindowBody(counter_number=1), username="alice")
    assert busy.value.status_code == 409 and "Q7001" in busy.value.detail

    queue_service.mark_done(7001)
    assert queue_router.queue_call_next(
        queue_router.WindowBody(counter_number=1), username="alice")["called"]["queue_number"] == 7002
    queue_service.mark_done(7002)
    empty = queue_router.queue_call_next(queue_router.WindowBody(counter_number=1), username="alice")
    assert empty == {"called": None, "message": "No one is waiting."}


def test_staff_can_switch_between_call_next_and_automatic(tracker, monkeypatch):
    monkeypatch.setattr(queue_service, "_call_mode", "staff")
    for n in (7001, 7002, 7003, 7004):
        tracker.force_new_person(n)
    tracker.call_next(3)                                   # one manual call first
    assert counters(tracker) == {7001: 3, 7002: None, 7003: None, 7004: None}

    overview = queue_router.queue_call_mode(queue_router.CallModeBody(mode="auto"), username="alice")
    assert overview["call_mode"] == "auto"
    assert counters(tracker) == {7001: 3, 7002: 1, 7003: 2, 7004: None}, \
        "free windows are filled at once, the existing call is kept"

    queue_router.queue_call_mode(queue_router.CallModeBody(mode="staff"), username="alice")
    queue_service.mark_done(7002)
    assert counters(tracker)[7004] is None, "back on Call next: window 1 waits for staff"

    with pytest.raises(HTTPException) as bad:
        queue_router.queue_call_mode(queue_router.CallModeBody(mode="sometimes"), username="alice")
    assert bad.value.status_code == 400


def test_a_queue_reset_keeps_the_chosen_call_mode(tracker, monkeypatch):
    monkeypatch.setattr(queue_service, "_call_mode", "staff")
    monkeypatch.setattr(queue_service, "delete_all_tickets", lambda: 0)
    monkeypatch.setattr(queue_service, "record_queue_reset", lambda actor=None: None)
    queue_service.set_call_mode("auto", "alice")
    queue_service.reset_queue("alice")
    assert queue_service.queue_tracker.auto_call is True
    queue_service.set_call_mode("staff", "alice")
    queue_service.reset_queue("alice")
    assert queue_service.queue_tracker.auto_call is False


def test_closed_message_names_the_reopening_time(tracker, monkeypatch):
    tracker.force_new_person(7001)
    until = datetime.now() + timedelta(minutes=30)
    tracker.counters_closed_until = until
    monkeypatch.setattr(queue_service, "counters_closed_until", lambda: until)
    with pytest.raises(HTTPException) as closed:
        queue_router.queue_call_next(queue_router.WindowBody(counter_number=1), username="alice")
    assert closed.value.status_code == 409 and "closed until" in closed.value.detail
