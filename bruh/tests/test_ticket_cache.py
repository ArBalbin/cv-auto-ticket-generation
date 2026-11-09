"""
The ticket-validation cache that keeps student status polls from exhausting
the database connection pool under load. No real database is used.
"""
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from database import database_handler  # noqa: E402
from services import ticket_printer  # noqa: E402


class FakeCursor:
    def __init__(self, pool):
        self.pool = pool

    def execute(self, sql, params):
        if self.pool.fail:
            raise RuntimeError("query failed")

    def fetchone(self):
        return dict(self.pool.row) if self.pool.row else None

    def close(self):
        pass


class FakeConn:
    def __init__(self, pool):
        self.pool = pool

    def cursor(self, dictionary=False):
        return FakeCursor(self.pool)

    def close(self):
        self.pool.open -= 1


class FakePool:
    def __init__(self, status="waiting"):
        self.row = {"id": 1, "queue_number": 5001, "short_code": "ABCD-EFGH",
                    "jwt_token": ticket_printer.generate_jwt_token(5001), "status": status}
        self.fail = False
        self.connections = 0
        self.open = 0

    def get_connection(self):
        self.connections += 1
        self.open += 1
        return FakeConn(self)


@pytest.fixture(autouse=True)
def empty_cache():
    ticket_printer.invalidate_ticket_cache()
    yield
    ticket_printer.invalidate_ticket_cache()


def lookup(pool, code="ABCD-EFGH", q=5001):
    return ticket_printer.get_ticket_record_by_short_code(code, q, pool)


def test_repeated_polls_of_a_waiting_ticket_use_one_connection():
    pool = FakePool()
    for _ in range(20):
        assert lookup(pool)["status"] == "waiting"
    assert pool.connections == 1


def test_short_code_case_does_not_split_the_cache():
    pool = FakePool()
    lookup(pool, "abcd-efgh")
    lookup(pool, "ABCD-EFGH")
    assert pool.connections == 1


def test_a_finished_ticket_is_never_cached():
    pool = FakePool(status="served")
    lookup(pool)
    lookup(pool)
    assert pool.connections == 2


def test_a_status_change_is_seen_on_the_next_poll():
    pool = FakePool()
    lookup(pool)
    pool.row["status"] = "served"                 # staff mark the ticket done
    ticket_printer.invalidate_ticket_cache(5001)
    assert lookup(pool)["status"] == "served"
    assert pool.connections == 2


def test_a_failed_query_reports_busy_and_returns_its_connection():
    pool = FakePool()
    pool.fail = True
    with pytest.raises(ticket_printer.TicketLookupBusy):
        lookup(pool)
    assert pool.open == 0, "the connection was handed back to the pool"


def test_a_wrong_code_is_remembered_so_it_cannot_flood_the_database():
    pool = FakePool()
    pool.row = None
    for _ in range(10):
        assert lookup(pool, "WRONG-CODE") is None
    assert pool.connections == 1


def test_pool_exhaustion_is_recognised_as_busy_not_as_a_crash():
    class PoolError(Exception):
        pass
    assert database_handler._is_pool_exhausted(PoolError("Failed getting connection; pool exhausted"))
    assert not database_handler._is_pool_exhausted(ValueError("something else"))


def test_a_storage_outage_does_not_stop_the_ticket(monkeypatch, tmp_path):
    # Regression: an upload failure aborted the ticket, so the student never
    # got a short code and the app stayed on "Join the Queue".
    from services import object_storage_service

    class S3UploadFailedError(Exception):
        pass

    class FailingStorage:
        def upload_file(self, **kwargs):
            raise S3UploadFailedError("Failed to upload ticket: An error occurred () when calling the PutObject operation")

    monkeypatch.setattr(object_storage_service, "client", lambda: FailingStorage())
    monkeypatch.setattr(ticket_printer, "TICKETS_OUTPUT_DIR", str(tmp_path))

    ticket = ticket_printer.issue_ticket(queue_number=9002, position=1, est_wait_min=0)

    assert ticket is not None, "the ticket is still issued"
    assert ticket["short_code"] and ticket["jwt_token"]
    assert ticket["storage_url"] is None, "it simply keeps the local PDF"


def test_simultaneous_polls_share_one_database_read():
    import threading
    import time as _time

    pool = FakePool()
    original = pool.get_connection

    def slow_connection():
        _time.sleep(0.05)            # a real query to the cloud database
        return original()

    pool.get_connection = slow_connection
    threads = [threading.Thread(target=lookup, args=(pool,)) for _ in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert pool.connections == 1, "thirty pollers, one database read"


def test_a_busy_database_answers_with_the_last_known_ticket(monkeypatch):
    pool = FakePool()
    assert lookup(pool)["status"] == "waiting"
    key = ("ABCD-EFGH", 5001)
    expires, row = ticket_printer._ticket_cache[key]
    ticket_printer._ticket_cache[key] = (expires - 30, row)    # expired 20 s ago
    pool.fail = True                                            # pool exhausted
    assert lookup(pool)["status"] == "waiting", "not reported as an invalid ticket"


def test_waiting_list_refreshes_once_in_the_background(monkeypatch):
    import threading
    import time as _time
    from services import queue_service as qs

    calls = []
    release = threading.Event()

    def slow_fetch():
        calls.append(1)
        release.wait(2)
        return [{"queue_number": len(calls)}]

    monkeypatch.setattr(qs, "fetch_waiting_queue_records", slow_fetch)
    monkeypatch.setattr(qs, "_waiting_records_cache", [{"queue_number": 0}])
    monkeypatch.setattr(qs, "_waiting_records_read_at", _time.monotonic() - 60)   # expired
    monkeypatch.setattr(qs, "_waiting_records_refreshing", False)

    started = _time.perf_counter()
    answers = [qs._waiting_queue_records() for _ in range(20)]
    assert _time.perf_counter() - started < 0.5, "nobody waited for the database"
    assert all(a == [{"queue_number": 0}] for a in answers), "callers got the last list"
    release.set()
    _time.sleep(0.2)
    assert len(calls) == 1, "one background read, not twenty"


def test_an_expired_ticket_answers_at_once_and_refreshes_in_the_background():
    import time as _time

    pool = FakePool()
    lookup(pool)
    key = ("ABCD-EFGH", 5001)
    expires, row = ticket_printer._ticket_cache[key]
    ticket_printer._ticket_cache[key] = (expires - 15, row)       # just expired

    pool.row["status"] = "served"                                  # changed without invalidation
    started = _time.perf_counter()
    assert lookup(pool)["status"] == "waiting", "the last known answer, at once"
    assert _time.perf_counter() - started < 0.05
    _time.sleep(0.2)                                               # background refresh
    assert key not in ticket_printer._ticket_cache, "a finished ticket is dropped"
    assert lookup(pool)["status"] == "served"
