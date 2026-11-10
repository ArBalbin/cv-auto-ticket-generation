"""
The serving window logged at done and no-show, and the anonymous
service-records download. No real database.

Run:  .venv/Scripts/python.exe -m pytest tests -v
"""
import inspect
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from core.security import require_staff  # noqa: E402
from database import database_handler  # noqa: E402
from routers import queue  # noqa: E402
from services import queue_service  # noqa: E402
from services.queue_tracker import QueueTracker, QueueZone  # noqa: E402


def test_tracker_reports_each_person_s_counter():
    tracker = QueueTracker(zone=QueueZone(0, 0, 640, 480))
    for n in (7001, 7002, 7003, 7004):
        tracker.force_new_person(n)
    assert [tracker.get_counter_number(n) for n in (7001, 7002, 7003, 7004)] == [1, 2, 3, None]
    assert tracker.get_counter_number(9999) is None


def test_done_and_no_show_pass_the_window_to_the_database(monkeypatch):
    tracker = QueueTracker(zone=QueueZone(0, 0, 640, 480))
    monkeypatch.setattr(queue_service, "queue_tracker", tracker)
    writes = []
    monkeypatch.setattr(queue_service, "update_queue_status",
                        lambda q, status, actor, **kw: writes.append((q, status, kw["cashier_window"])))
    monkeypatch.setattr(queue_service, "invalidate_ticket_cache", lambda q=None: None)

    class InlineThread:
        def __init__(self, target, args, **_):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(queue_service.threading, "Thread", InlineThread)

    for n in (7101, 7102):
        tracker.force_new_person(n)
    queue_service.mark_done(7101, "staff")
    queue_service.mark_no_show(7102, "staff")
    assert writes == [(7101, "served", 1), (7102, "no_show", 2)]


class RecordingCursor:
    def __init__(self, columns):
        self.columns, self.sql, self.params = columns, None, None
        self.rowcount = 1

    def execute(self, sql, params=()):
        self.sql, self.params = " ".join(sql.split()), tuple(params)

    def fetchall(self):
        return []


def test_service_records_select_no_student_identifiers(monkeypatch):
    cursor = RecordingCursor({
        "created_at", "called_at", "served_at", "status", "people_ahead_at_issue",
        "active_counters_at_issue", "cashier_window", "predicted_wait_min",
        "wait_model_version", "student_id", "short_code", "jwt_token",
    })

    class Conn:
        def cursor(self, dictionary=False):
            return cursor

    class Pool:
        def get_connection(self):
            return Conn()

    monkeypatch.setattr(database_handler, "_ensure_db_pool", lambda: Pool())
    monkeypatch.setattr(database_handler, "_get_table_columns", lambda c, t: cursor.columns)
    monkeypatch.setattr(database_handler, "close_db_resources", lambda *a: None)

    assert database_handler.fetch_completed_transactions() == []
    selected = cursor.sql.split(" FROM ")[0]
    for private in ("student_id", "short_code", "jwt_token", "school_id", "full_name"):
        assert private not in selected
    assert "cashier_window" in selected and "status = 'served'" in cursor.sql


def test_download_is_staff_only_and_csv(monkeypatch):
    route = queue.queue_service_records
    assert any(getattr(p.default, "dependency", None) is require_staff
               for p in inspect.signature(route).parameters.values())
    monkeypatch.setattr(queue, "fetch_completed_transactions", lambda: [{
        "ticket_created_at": datetime(2026, 11, 10, 9, 0), "service_started_at": datetime(2026, 11, 10, 9, 4),
        "service_completed_at": datetime(2026, 11, 10, 9, 6), "actual_wait_seconds": 240,
        "actual_service_seconds": 120, "people_ahead_at_issue": 3, "active_counters_at_issue": 3,
        "cashier_window": 2, "payment_type": "Tuition fee", "predicted_wait_min": 2.0,
        "wait_model_version": "queue-engine",
    }])
    response = route(username="staff")
    body = response.body.decode()
    assert response.media_type == "text/csv"
    assert body.splitlines()[0] == ",".join(queue.SERVICE_RECORD_FIELDS)
    assert ",240,120,3,3,2,Tuition fee,2.0,queue-engine" in body
