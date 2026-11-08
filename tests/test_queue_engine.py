"""
The queue engine on its own: counters, a line, a service time, and when each
person in that line reaches a counter. No tracker, database or clock.

Run:  .venv/Scripts/python.exe -m pytest tests -v
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.queue_engine import CounterState, expected_remaining, start_times, wait_minutes  # noqa: E402

NOW = datetime(2026, 11, 9, 10, 0)
IDLE = CounterState()


def busy(minutes_ago: float) -> CounterState:
    return CounterState(busy_since=NOW - timedelta(minutes=minutes_ago))


def test_free_counters_take_the_first_people_at_once():
    assert [wait_minutes(NOW, [IDLE] * 3, k, 2.0) for k in (0, 1, 2)] == [0, 0, 0]
    assert wait_minutes(NOW, [IDLE] * 3, 3, 2.0) == pytest.approx(2.0)


def test_people_go_to_whichever_counter_frees_first():
    counters = [busy(0), busy(1.5), busy(0)]       # the second counter is nearly done
    first = start_times(NOW, counters, 1, 2.0)[0]
    assert (first - NOW).total_seconds() / 60 == pytest.approx(expected_remaining(1.5, 2.0))


def test_each_round_of_counters_adds_one_service_time():
    counters = [busy(0)] * 3
    assert wait_minutes(NOW, counters, 0, 2.0) == pytest.approx(2.0)
    assert wait_minutes(NOW, counters, 3, 2.0) == pytest.approx(4.0)
    assert wait_minutes(NOW, counters, 30, 2.0) == pytest.approx(22.0)


def test_an_overdue_transaction_still_has_time_left():
    assert expected_remaining(0, 2.0) == pytest.approx(2.0)
    assert expected_remaining(1, 2.0) == pytest.approx(1.5)
    assert expected_remaining(5, 2.0) == pytest.approx(1.5), "three quarters of a service, as in the October log"


def test_a_closed_counter_takes_no_one_before_it_reopens():
    lunch = CounterState(closed_until=NOW + timedelta(minutes=30))
    assert wait_minutes(NOW, [lunch] * 3, 0, 2.0) == pytest.approx(30.0)
    assert wait_minutes(NOW, [lunch] * 3, 3, 2.0) == pytest.approx(32.0)


def test_a_counter_closing_mid_transaction_reopens_after_both():
    counter = CounterState(busy_since=NOW, closed_until=NOW + timedelta(minutes=1))
    assert wait_minutes(NOW, [counter], 0, 2.0) == pytest.approx(2.0)


def test_no_counters_gives_no_estimate():
    assert start_times(NOW, [], 3, 2.0) == []
    assert wait_minutes(NOW, [], 3, 2.0) == 0.0
