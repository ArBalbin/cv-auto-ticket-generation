"""
The accuracy summary the dashboard shows: logged estimates scored against the
real waits of served tickets. No real database.

Run:  .venv/Scripts/python.exe -m pytest tests -v
"""
import sys
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services import wait_accuracy  # noqa: E402

T0 = datetime(2026, 11, 10, 9, 0)


def served(number, predicted, waited_min, day=date(2026, 11, 10)):
    created = T0 + timedelta(minutes=number)
    return {
        "service_date": day,
        "queue_number": number,
        "created_at": created,
        "called_at": created + timedelta(minutes=waited_min),
        "predicted_wait_min": predicted,
    }


def test_no_tickets_yet():
    s = wait_accuracy.summarize([])
    assert s["tickets_checked"] == 0
    assert s["average_miss_minutes"] is None
    assert s["recent"] == []


def test_scores_estimates_and_the_fixed_guess_on_the_same_tickets():
    rows = [served(1, 0.0, 0.0), served(2, 2.0, 3.0), served(3, 4.0, 9.0)]
    s = wait_accuracy.summarize(rows)
    assert s["tickets_checked"] == 3
    assert s["average_miss_minutes"] == 2.0             # (0 + 1 + 5) / 3
    assert s["close_enough_percent"] == 67              # 2 of 3 within 2 min
    assert s["fixed_guess_average_miss_minutes"] == 3.33  # (2 + 1 + 7) / 3
    assert s["days_covered"] == 1


def test_reports_which_way_the_estimates_lean():
    too_low = wait_accuracy.summarize([served(n, 1.0, 4.0) for n in range(1, 4)])
    assert too_low["typical_direction"] == "too_low"   # waited longer than told
    assert too_low["typical_miss_minutes"] == 3.0
    too_high = wait_accuracy.summarize([served(n, 5.0, 1.0) for n in range(1, 4)])
    assert too_high["typical_direction"] == "too_high"
    even = wait_accuracy.summarize([served(n, 2.0, 2.2) for n in range(1, 4)])
    assert even["typical_direction"] == "even"


def test_newest_first_and_capped():
    rows = [served(n, 1.0, 1.0) for n in range(1, 20)]
    recent = wait_accuracy.summarize(rows)["recent"]
    assert len(recent) == wait_accuracy.RECENT_LIMIT
    assert recent[0]["queue_number"] == 19


def test_database_decimals_and_early_calls():
    # MySQL returns DECIMAL columns as Decimal; a call logged a moment before
    # the ticket row was written counts as no wait.
    row = served(1, Decimal("1.5"), 0.0)
    row["called_at"] = row["created_at"] - timedelta(seconds=5)
    s = wait_accuracy.summarize([row])
    assert s["recent"][0] == {
        "queue_number": 1, "service_date": "2026-11-10",
        "predicted_minutes": 1.5, "actual_minutes": 0.0,
    }


def test_rows_without_an_estimate_are_skipped():
    row = served(1, None, 3.0)
    assert wait_accuracy.summarize([row])["tickets_checked"] == 0


def test_database_failure_is_reported_not_raised(monkeypatch):
    def broken(days=None):
        raise RuntimeError("database down")
    monkeypatch.setattr(wait_accuracy, "fetch_wait_observations", broken)
    monkeypatch.setattr(wait_accuracy, "_cache", None)
    s = wait_accuracy.current_summary()
    assert s["available"] is False
    assert s["tickets_checked"] == 0


def test_summary_is_cached_between_polls(monkeypatch):
    calls = []

    def fetch(days=None):
        calls.append(days)
        return [served(1, 2.0, 2.0)]
    monkeypatch.setattr(wait_accuracy, "fetch_wait_observations", fetch)
    monkeypatch.setattr(wait_accuracy, "_cache", None)
    first = wait_accuracy.current_summary()
    second = wait_accuracy.current_summary()
    assert first is second and first["available"] is True
    assert calls == [wait_accuracy.LOOKBACK_DAYS]
