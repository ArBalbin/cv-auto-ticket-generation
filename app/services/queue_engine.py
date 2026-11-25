"""
When will each person in line reach a window? (Estimated time of service.)

Worked out again on every request (the app asks every few seconds) from what
the system knows at that moment: who is at which window and for how long, the
order of the line, which windows are open, and how long the transactions
finished recently took. Nothing is added or taken away in fixed steps; every
estimate moves with the line.

Two things are learned from data.

The SHAPE of a transaction's length. How long a transaction that has already
run e minutes still has to go is its mean residual life, E[S - e | S > e].
It comes from the 511 transactions timed at the NCF cashier in October 2026,
logged in whole minutes: 191 took 1, 238 took 2, 56 took 3, 12 took 4, 11 took
5, 2 took 6 and 1 took 8. A lognormal fitted to those by maximum likelihood
(each logged minute v standing for v - 0.5 to v + 0.5) has a mean of 1.87
minutes and a log-spread of 0.425. The lognormal is the usual finding for
service times (Brown et al., 2005). Its mean residual life falls while a
transaction runs normally and rises again once one runs long, as the log does:
on average one that had taken 1 minute had 0.99 left (log: 1.13), one at 2
minutes 0.74 (log: 0.78), and one at 3 to 4 minutes about 0.75 (log: about 1).

The SPEED of the cashier right now: the average length of the last 20
services finished in the past two hours, pulled toward the long-run average when only a
few have been timed. A student still at a window counts too, at the time
taken so far, once that is longer than the average. That window is
evidently slower than usual, so everyone behind it is told so at once.

A window's remaining time is then average x shape: avg * m(elapsed / avg),
where m is the fitted mean residual life in units of its own mean.

The TIME STAFF TAKE TO CALL the next student: with staff calling, a free
window takes nobody until a member of staff presses Call next. It is learned
from the calls themselves (queue_service.current_call_delay_minutes) and is
zero when windows call by themselves. On 20-24 Nov 2026 the engine said 0:00
to students with nobody ahead, who were then called about 71 s later.

The line is then played forward. Each window is free now, free when its
transaction is expected to end, or closed until a set time (lunch), and then
takes the next person once staff call. People still waiting are taken in queue
order, each by whichever window frees up first, and that window is then busy
for one average service and the call after it. The moment a person is taken is
their estimated time of service.

Pure functions, no I/O: the caller passes the line's state.
"""
import heapq
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

# The October 2026 cashier log, as fitted above.
HISTORY_MEAN_MIN = 1.87
SHAPE_SIGMA = 0.425

# The services that set the current average: the last twenty that finished in
# the past two hours, with the long-run average counted as ten more. Chosen on
# replayed lines (normal, 40 % slower, 25 % faster, and turning slower
# mid-day): ten services with a weight of three swung too much on normal days,
# thirty with twenty followed a slower or faster day too slowly.
PRIOR_WEIGHT = 10
RECENT_SERVICES = 20
RECENT_WINDOW_MIN = 120.0

# A transaction is never assumed to run past this many average lengths.
_MAX_FRACTION = 60.0


def _phi(z: float) -> float:
    """Standard normal distribution function, accurate far into the tails."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def remaining_fraction(x: float) -> float:
    """
    Mean remaining length of a transaction that has run `x` average lengths
    so far, in average lengths: E[X - x | X > x] for a lognormal X with mean 1.
    """
    if x <= 0:
        return 1.0
    x = min(x, _MAX_FRACTION)
    s = SHAPE_SIGMA
    log_x = math.log(x)
    # With X = exp(N(-s^2/2, s^2)):  P(X > x) = phi((-s^2/2 - ln x) / s)
    # and E[X; X > x] = phi((s^2/2 - ln x) / s).
    survive = _phi((-s * s / 2 - log_x) / s)
    beyond = _phi((s * s / 2 - log_x) / s)
    return max(0.0, beyond / survive - x)


@dataclass(frozen=True)
class ServiceModel:
    mean_min: float = HISTORY_MEAN_MIN   # the current average service time
    timed: int = 0                       # recent services behind that average
    call_delay_min: float = 0.0          # how long staff take to call the next student

    def remaining(self, elapsed_min: float) -> float:
        """Expected minutes left in a transaction that has run `elapsed_min`."""
        mean = max(0.1, self.mean_min)
        return mean * remaining_fraction(max(0.0, elapsed_min) / mean)


def current_average(
    finished_min: list[float],
    at_window_min: list[float] = (),
    prior_min: float = HISTORY_MEAN_MIN,
    prior_weight: float = PRIOR_WEIGHT,
) -> float:
    """
    The average service time now: the services just finished, plus the
    long-run average counted as `prior_weight` services, plus any transaction
    still at a window that has already run longer than that average, counted at
    its length so far (it will be at least that long).
    """
    total = prior_weight * prior_min + sum(finished_min)
    count = prior_weight + len(finished_min)
    average = total / count
    long_running = [e for e in at_window_min if e > average]
    return (total + sum(long_running)) / (count + len(long_running))


def service_model(
    finished_min: list[float],
    at_window_min: list[float] = (),
    prior_min: float = HISTORY_MEAN_MIN,
    call_delay_min: float = 0.0,
) -> ServiceModel:
    recent = list(finished_min)[-RECENT_SERVICES:]
    return ServiceModel(
        mean_min=current_average(recent, list(at_window_min), prior_min),
        timed=len(recent),
        call_delay_min=max(0.0, call_delay_min),
    )


def _as_model(service) -> ServiceModel:
    """A bare number of minutes is a model with that average."""
    if isinstance(service, ServiceModel):
        return service
    return ServiceModel(mean_min=max(0.1, float(service)))


def expected_remaining(elapsed_min: float, service) -> float:
    """Minutes left in a transaction that has run `elapsed_min` so far."""
    return _as_model(service).remaining(elapsed_min)


@dataclass(frozen=True)
class CounterState:
    busy_since: datetime | None = None    # when the person now at it was called
    closed_until: datetime | None = None  # no one is called before this time
    # Free, with someone waiting: since when staff could have called them
    # (the later of the window freeing and that person getting their number).
    ready_since: datetime | None = None


def free_at(counter: CounterState, now: datetime, service) -> datetime:
    """When this counter takes the next person: once its transaction is
    expected to end, after any closure, and once staff call."""
    model = _as_model(service)
    call = model.call_delay_min
    if counter.busy_since is not None:
        elapsed = (now - counter.busy_since).total_seconds() / 60
        free = now + timedelta(minutes=model.remaining(elapsed) + call)
    else:
        # Already free: the call is due that long after it could first have
        # been made, so the estimate counts down with the student's clock.
        idle = 0.0
        if counter.ready_since is not None:
            idle = max(0.0, (now - counter.ready_since).total_seconds() / 60)
        free = now + timedelta(minutes=max(0.0, call - idle))
    if counter.closed_until is not None:
        reopens = counter.closed_until + timedelta(minutes=call)
        if reopens > free:
            free = reopens
    return free


def start_times(
    now: datetime,
    counters: list[CounterState],
    waiting: int,
    service,
) -> list[datetime]:
    """
    When each of the next `waiting` people in line reaches a counter, in queue
    order. With no counters there is nowhere to go, so nobody gets a time.
    """
    if not counters or waiting <= 0:
        return []
    model = _as_model(service)
    # One average service, then the call of the next person.
    length = timedelta(minutes=max(0.1, model.mean_min) + model.call_delay_min)
    free = [free_at(c, now, model) for c in counters]
    heapq.heapify(free)
    starts = []
    for _ in range(waiting):
        start = heapq.heappop(free)
        starts.append(start)
        heapq.heappush(free, start + length)
    return starts


def wait_minutes(
    now: datetime,
    counters: list[CounterState],
    waiting_ahead: int,
    service,
) -> float:
    """Minutes until someone with `waiting_ahead` people still waiting in front
    of them (not counting those already at a counter) reaches a counter."""
    starts = start_times(now, counters, waiting_ahead + 1, service)
    if not starts:
        return 0.0
    return max(0.0, (starts[-1] - now).total_seconds() / 60)
