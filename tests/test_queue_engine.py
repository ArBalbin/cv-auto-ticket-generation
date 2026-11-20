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

from services.queue_engine import (  # noqa: E402
    HISTORY_MEAN_MIN,
    PRIOR_WEIGHT,
    RECENT_SERVICES,
    CounterState,
    ServiceModel,
    current_average,
    expected_remaining,
    service_model,
    start_times,
    wait_minutes,
)

NOW = datetime(2026, 11, 9, 10, 0)
IDLE = CounterState()


def busy(minutes_ago: float, now: datetime = NOW) -> CounterState:
    return CounterState(busy_since=now - timedelta(minutes=minutes_ago))


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


def test_the_time_left_follows_the_october_cashier_log():
    model = ServiceModel(mean_min=HISTORY_MEAN_MIN)
    assert model.remaining(0) == pytest.approx(HISTORY_MEAN_MIN)
    # The log (whole minutes, read as intervals) had 1.13 left at 1 minute and
    # 0.78 at 2; the fitted shape gives 0.99 and 0.74.
    assert model.remaining(1) == pytest.approx(0.99, abs=0.02)
    assert model.remaining(2) == pytest.approx(0.74, abs=0.02)
    # A transaction that has run long is not about to end, and the longer it
    # runs the more it has left.
    assert model.remaining(4) > 0.6
    assert model.remaining(20) > model.remaining(10) > model.remaining(4)


def test_with_nothing_timed_the_average_is_the_usual_pace():
    assert current_average([], []) == pytest.approx(HISTORY_MEAN_MIN)
    assert service_model([]).timed == 0


def test_services_just_timed_move_the_average():
    slow = current_average([4.0] * 10)
    assert slow == pytest.approx((PRIOR_WEIGHT * HISTORY_MEAN_MIN + 40) / (PRIOR_WEIGHT + 10))
    assert current_average([1.0] * 10) < HISTORY_MEAN_MIN < slow
    assert service_model([2.0] * 45).timed == RECENT_SERVICES, "only the most recent count"


def test_a_transaction_running_long_raises_the_average_while_it_runs():
    finished = [2.0] * 4
    usual = current_average(finished)
    assert current_average(finished, [1.0]) == pytest.approx(usual), "within the average: no evidence yet"
    five, ten = current_average(finished, [5.0]), current_average(finished, [10.0])
    assert usual < five < ten


def test_the_panels_case_a_long_transaction_pushes_everyone_behind_later():
    """Q1 is called to the only window and staff take long to mark it done.
    Q2 and Q3 wait. Their estimates must not stay frozen: they count down
    while Q1 takes a usual time, then grow as Q1 keeps running, and their
    estimated time of service moves later at every step."""
    called = NOW
    finished = [2.0] * 4                       # four services timed earlier

    def estimates(minutes_in: float):
        now = called + timedelta(minutes=minutes_in)
        model = service_model(finished, [minutes_in])
        starts = start_times(now, [busy(minutes_in, now)], 2, model)
        q2, q3 = ((s - now).total_seconds() for s in starts)
        return q2, q3, starts[0], starts[1]

    q2_0, q3_0, eta2_0, eta3_0 = estimates(0)
    q2_1, q3_1, _, _ = estimates(1)
    assert q2_1 < q2_0 and q3_1 < q3_0, "a usual transaction: the waits count down"

    q2_3, q3_3, eta2_3, eta3_3 = estimates(3)
    q2_5, q3_5, eta2_5, eta3_5 = estimates(5)
    q2_10, q3_10, eta2_10, eta3_10 = estimates(10)
    assert q2_3 < q2_5 < q2_10, "Q2's wait grows while Q1 overruns"
    assert q3_3 < q3_5 < q3_10, "Q3's wait grows while Q1 overruns"
    assert eta2_0 < eta2_3 < eta2_5 < eta2_10, "Q2's time of service keeps moving later"
    assert eta3_0 < eta3_3 < eta3_5 < eta3_10, "Q3's time of service keeps moving later"
    # Q3 is always one more transaction behind Q2.
    assert q3_10 - q2_10 == pytest.approx(current_average(finished, [10.0]) * 60, abs=1)


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
