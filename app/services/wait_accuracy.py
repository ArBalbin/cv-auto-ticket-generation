"""
How far the wait estimates shown to students were from their real waits.

Every ticket logs the estimate it was given when issued (predicted_wait_min).
Once it is served, the log also holds when it was called (called_at). The
real wait is called_at minus created_at. The database clock writes both, so
the server's time zone does not matter.

This is the only accuracy evidence the system can produce. The October
cashier records have no real head count, so they cannot validate any
estimator, the queue engine included (WAIT_TIME_MODEL.md).

The same tickets are also scored against a fixed 2-minute guess. That is the
model the separate prediction repo ships (its global median). An estimator
that cannot beat it on real tickets is not earning its place.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from statistics import mean, median

from database.database_handler import fetch_wait_observations

FIXED_GUESS_MINUTES = 2.0
CLOSE_ENOUGH_MINUTES = 2.0
# A median miss smaller than this is reported as neither too high nor too low.
EVEN_MINUTES = 0.5
RECENT_LIMIT = 8
LOOKBACK_DAYS = 30
CACHE_SECONDS = 30

_cache_lock = threading.Lock()
_cache: tuple[float, dict] | None = None


def _minutes_between(start: datetime, end: datetime) -> float:
    # A call can land a moment before the ticket row is written (the row is
    # saved by a background worker). That is no wait, not a negative one.
    return max(0.0, (end - start).total_seconds() / 60)


def summarize(observations: list[dict]) -> dict:
    """Accuracy of the logged estimates, from rows of fetch_wait_observations()."""
    tickets = []
    for o in observations:
        created, called, predicted = o.get("created_at"), o.get("called_at"), o.get("predicted_wait_min")
        if created is None or called is None or predicted is None:
            continue
        tickets.append({
            "queue_number": o.get("queue_number"),
            "service_date": str(o.get("service_date") or created.date()),
            "predicted": float(predicted),
            "actual": _minutes_between(created, called),
        })

    summary = {
        "tickets_checked": len(tickets),
        "days_covered": len({t["service_date"] for t in tickets}),
        "lookback_days": LOOKBACK_DAYS,
        "close_enough_minutes": CLOSE_ENOUGH_MINUTES,
        "fixed_guess_minutes": FIXED_GUESS_MINUTES,
        "average_miss_minutes": None,
        "close_enough_percent": None,
        "fixed_guess_average_miss_minutes": None,
        "typical_direction": None,
        "typical_miss_minutes": None,
        "recent": [],
    }
    if not tickets:
        return summary

    misses = [abs(t["actual"] - t["predicted"]) for t in tickets]
    signed = median(t["actual"] - t["predicted"] for t in tickets)
    summary.update({
        "average_miss_minutes": round(mean(misses), 2),
        "close_enough_percent": round(100 * sum(m <= CLOSE_ENOUGH_MINUTES for m in misses) / len(misses)),
        "fixed_guess_average_miss_minutes": round(
            mean(abs(t["actual"] - FIXED_GUESS_MINUTES) for t in tickets), 2
        ),
        # "too_low" means students waited longer than they were told.
        "typical_direction": "too_low" if signed > EVEN_MINUTES else "too_high" if signed < -EVEN_MINUTES else "even",
        "typical_miss_minutes": round(abs(signed), 1),
        "recent": [
            {
                "queue_number": t["queue_number"],
                "service_date": t["service_date"],
                "predicted_minutes": round(t["predicted"], 1),
                "actual_minutes": round(t["actual"], 1),
            }
            for t in reversed(tickets[-RECENT_LIMIT:])
        ],
    })
    return summary


def current_summary() -> dict:
    """The summary for the last LOOKBACK_DAYS days, cached briefly so every
    open dashboard does not query the database on each poll."""
    global _cache
    with _cache_lock:
        if _cache is not None and time.monotonic() - _cache[0] < CACHE_SECONDS:
            return _cache[1]
    try:
        summary = {**summarize(fetch_wait_observations(days=LOOKBACK_DAYS)), "available": True}
    except Exception as exc:  # the dashboard shows "unavailable", not an error page
        print(f"[WaitAccuracy] Could not read the wait log: {exc}")
        return {**summarize([]), "available": False}
    with _cache_lock:
        _cache = (time.monotonic(), summary)
    return summary
