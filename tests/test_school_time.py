"""
Times shown to staff and students are on the school's clock (UTC+8), not the
server's. On Render the server clock is UTC, so the dashboard showed "Joined
12:30:31 PM" for a student who joined at 8:30 PM. No real database.

Run:  .venv/Scripts/python.exe -m pytest tests -v
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from core.school_time import CLOCK_FULL, SCHOOL_TZ, clock, db_clock  # noqa: E402
from services import queue_service  # noqa: E402


def test_a_database_time_is_shown_in_naga_time():
    # The database stamps NOW() in UTC: 12:30:31 UTC is 8:30:31 PM in Naga.
    assert db_clock(datetime(2026, 11, 19, 12, 30, 31)) == "08:30:31 PM"
    assert db_clock(datetime(2026, 11, 19, 12, 30, 31), CLOCK_FULL) == "Nov 19, 2026 08:30:31 PM"


def test_a_server_time_on_a_utc_server_is_shown_in_naga_time():
    # What datetime.now() means on Render, made explicit.
    on_render = datetime(2026, 11, 19, 12, 30, 31, tzinfo=timezone.utc)
    assert clock(on_render) == "08:30:31 PM"


def test_a_server_time_on_this_machine_matches_the_school_clock():
    shown = clock(datetime.now())
    expected = datetime.now(SCHOOL_TZ).strftime("%I:%M:%S %p")
    # Allow the second to tick over between the two calls.
    assert shown[:5] == expected[:5]


def test_a_ticket_record_reports_joined_in_naga_time():
    record = {"queue_number": 5001, "created_at": "2026-11-19T12:30:31", "status": "waiting"}
    body = queue_service.ticket_record_waiting_response(record)
    assert body["joined_at"] == "08:30:31 PM"
    assert body["joined_at_full"] == "Nov 19, 2026 08:30:31 PM"
    # The machine-readable value is unchanged.
    assert body["joined_at_iso"] == "2026-11-19T12:30:31"
