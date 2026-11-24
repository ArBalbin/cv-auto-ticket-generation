"""
The staff dashboard's Queue Records page: a school day's tickets, on the
school's clock, with the day's totals. No real database.

Run:  .venv/Scripts/python.exe -m pytest tests -v
"""
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from routers import queue as queue_router  # noqa: E402
from services import queue_service  # noqa: E402

SERVED = {
    "id": 7, "queue_number": 5001, "status": "served",
    # The database stamps UTC: 06:30:00 UTC is 2:30:00 PM in Naga.
    "created_at": datetime(2026, 11, 24, 6, 30, 0),
    "called_at": datetime(2026, 11, 24, 6, 32, 15),
    "served_at": datetime(2026, 11, 24, 6, 34, 0),
    "cashier_window": 2, "payment_type": "Tuition", "payment_amount": 5000,
    "payment_note": None, "linked_via": "face_only", "is_walkin": 0,
    "predicted_wait_min": 2.5, "people_ahead_at_issue": 1,
    "student_name": "Test Student", "student_school_id": "21-0001", "served_by": "cashier1",
}
LEFT_WAITING = {**SERVED, "id": 8, "queue_number": 5002, "status": "waiting",
                "called_at": None, "served_at": None, "served_by": None}
WALK_IN = {**SERVED, "id": 9, "queue_number": 12, "is_walkin": 1,
           "student_name": None, "student_school_id": None}


@pytest.fixture
def database(monkeypatch):
    calls = []

    def fake(start, end, status, search, limit, offset):
        calls.append({"start": start, "end": end, "status": status, "search": search,
                      "limit": limit, "offset": offset})
        return {"rows": [SERVED, LEFT_WAITING, WALK_IN], "total": 3,
                "summary": {"tickets": 3, "served": 2, "no_show": 0, "waiting": 1,
                            "avg_wait_seconds": 135.4, "avg_service_seconds": 105.0,
                            "served_amount": 10000}}

    monkeypatch.setattr(queue_service, "fetch_queue_records", fake)
    return calls


def test_a_school_day_is_asked_for_in_the_databases_utc():
    start, end = queue_service.school_day_bounds_utc(date(2026, 11, 24))
    assert start == datetime(2026, 11, 23, 16, 0) and end == datetime(2026, 11, 24, 16, 0)


def test_each_record_is_shown_on_the_school_clock_with_its_wait_and_service(database):
    page = queue_service.queue_records_page("2026-11-24")
    row = page["records"][0]
    assert row["queue_label"] == "Q5001"
    assert row["joined_at"] == "02:30:00 PM" and row["called_at"] == "02:32:15 PM"
    assert row["finished_at"] == "02:34:00 PM"
    assert row["wait_seconds"] == 135 and row["service_seconds"] == 105
    assert row["status_label"] == "Served"
    assert row["student_name"] == "Test Student" and row["walk_in"] is False
    assert row["estimated_wait_seconds"] == 150, "the estimate the student was shown"
    assert row["payment_amount"] == 5000.0 and row["served_by"] == "cashier1"


def test_a_ticket_left_waiting_on_an_earlier_day_reads_not_finished(database):
    earlier = queue_service.queue_records_page("2026-11-20")
    assert earlier["records"][1]["status_label"] == "Not finished"
    assert earlier["records"][1]["wait_seconds"] is None
    assert earlier["records"][1]["finished_at"] is None


def test_a_walk_in_is_marked_as_one(database):
    page = queue_service.queue_records_page("2026-11-24")
    assert page["records"][2]["walk_in"] is True


def test_the_days_totals_and_paging(database):
    page = queue_service.queue_records_page("2026-11-24", "served", " 5001 ", page=2, page_size=25)
    assert database[-1]["status"] == "served" and database[-1]["search"] == " 5001 "
    assert database[-1]["limit"] == 25 and database[-1]["offset"] == 25
    assert page["search"] == "5001" and page["status"] == "served"
    assert page["summary"] == {"tickets": 3, "served": 2, "skipped": 0, "waiting": 1,
                               "avg_wait_seconds": 135, "avg_service_seconds": 105,
                               "served_amount": 10000.0}


def test_a_bad_date_or_status_falls_back_to_today_and_all(database):
    page = queue_service.queue_records_page("not-a-date", "everything")
    assert page["is_today"] is True and page["status"] == "all"
    assert database[-1]["status"] is None


def test_the_page_is_for_staff_only():
    route = next(r for r in queue_router.router.routes if getattr(r, "path", "") == "/api/queue/records")
    names = [d.call.__name__ for d in route.dependant.dependencies]
    assert "require_staff" in names


def test_no_database_gives_service_unavailable(monkeypatch):
    monkeypatch.setattr(queue_service, "fetch_queue_records", lambda *a, **k: None)
    with pytest.raises(queue_router.HTTPException) as err:
        queue_router.queue_records(username="staff")
    assert err.value.status_code == 503
