"""
When will each person in line reach a counter? (Estimated time of service.)

The line is simulated rather than fitted. Each counter is free now, free when
the transaction at it is expected to end, or closed until a set time (lunch).
People still waiting are taken in queue order, each by whichever counter frees
up first, and that counter is then busy for one service time. The moment a
person is taken is when their turn starts.

This uses what the system knows exactly at the moment: who is at which counter
and since when, the order of the line, how many counters are open, and whether
they are closed. The trained regression (wait_model.py) only knows a count of
people ahead, and was fitted on an overcounted reconstruction of it, which is
why it under-estimates long lines (WAIT_TIME_MODEL.md).

Pure functions, no I/O: the caller passes the line's state.
"""
import heapq
from dataclasses import dataclass
from datetime import datetime, timedelta

# A transaction that has already run past its usual length is not about to
# end: in the October log, one that had taken 1, 2 or 3 minutes so far went on
# for another 1.40, 1.55 and 1.73 minutes on average (median service 2.0). So
# the expected remainder never drops below three quarters of a service time.
MIN_REMAINING_FRACTION = 0.75


@dataclass(frozen=True)
class CounterState:
    busy_since: datetime | None = None    # when the person now at it was called
    closed_until: datetime | None = None  # no one is called before this time


def expected_remaining(elapsed_min: float, service_min: float) -> float:
    """Minutes left in a transaction that has run `elapsed_min` so far."""
    return max(service_min - max(0.0, elapsed_min), MIN_REMAINING_FRACTION * service_min)


def free_at(counter: CounterState, now: datetime, service_min: float) -> datetime:
    """When this counter can take the next person."""
    free = now
    if counter.busy_since is not None:
        elapsed = (now - counter.busy_since).total_seconds() / 60
        free = now + timedelta(minutes=expected_remaining(elapsed, service_min))
    if counter.closed_until is not None and counter.closed_until > free:
        free = counter.closed_until
    return free


def start_times(
    now: datetime,
    counters: list[CounterState],
    waiting: int,
    service_min: float,
) -> list[datetime]:
    """
    When each of the next `waiting` people in line reaches a counter, in queue
    order. With no counters there is nowhere to go, so nobody gets a time.
    """
    if not counters or waiting <= 0:
        return []
    service = timedelta(minutes=max(0.1, service_min))
    free = [free_at(c, now, service_min) for c in counters]
    heapq.heapify(free)
    starts = []
    for _ in range(waiting):
        start = heapq.heappop(free)
        starts.append(start)
        heapq.heappush(free, start + service)
    return starts


def wait_minutes(
    now: datetime,
    counters: list[CounterState],
    waiting_ahead: int,
    service_min: float,
) -> float:
    """Minutes until someone with `waiting_ahead` people still waiting in front
    of them (not counting those already at a counter) reaches a counter."""
    starts = start_times(now, counters, waiting_ahead + 1, service_min)
    if not starts:
        return 0.0
    return max(0.0, (starts[-1] - now).total_seconds() / 60)
