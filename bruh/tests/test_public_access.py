"""
What the API shows without a staff login: the display board gets queue
numbers and positions only, and the camera needs a staff session. No real
database or sessions.

Run:  .venv/Scripts/python.exe -m pytest tests -v
"""
import inspect
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from core import security  # noqa: E402
from core.security import require_staff, require_staff_for_media  # noqa: E402
from routers import crowd, queue  # noqa: E402

STAFF_TOKEN = "staff-session"


def request(query: str = "", headers: dict | None = None) -> Request:
    return Request({
        "type": "http",
        "query_string": query.encode(),
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
    })


@pytest.fixture(autouse=True)
def fake_sessions(monkeypatch):
    monkeypatch.setattr(security, "_lookup_session", lambda t: "staff" if t == STAFF_TOKEN else None)


def test_camera_accepts_the_session_header():
    assert require_staff_for_media(request(headers={"X-Session-Token": STAFF_TOKEN})) == "staff"


def test_camera_accepts_the_session_as_a_query_token():
    # The dashboard's <img> tag cannot send headers.
    assert require_staff_for_media(request(f"token={STAFF_TOKEN}")) == "staff"


@pytest.mark.parametrize("query", ["", "token=", "token=guessed"])
def test_camera_refuses_anyone_without_a_session(query):
    with pytest.raises(HTTPException) as e:
        require_staff_for_media(request(query))
    assert e.value.status_code == 401


def test_query_token_is_not_accepted_on_other_staff_routes():
    with pytest.raises(HTTPException):
        require_staff(request(f"token={STAFF_TOKEN}"))


def _depends_on(route_fn, dependency) -> bool:
    return any(
        getattr(p.default, "dependency", None) is dependency
        for p in inspect.signature(route_fn).parameters.values()
    )


def test_camera_routes_and_full_list_require_staff():
    assert _depends_on(crowd.get_video_stream, require_staff_for_media)
    assert _depends_on(crowd.get_snapshot, require_staff_for_media)
    assert _depends_on(queue.queue_list, require_staff)


def test_display_board_gets_no_ticket_codes(monkeypatch):
    person = {
        "queue_number": 12, "queue_label": "Q012", "status": "waiting",
        "position_in_line": 2, "counter_number": None, "wait_time": "1m 5s",
        "wait_time_seconds": 65, "access_token": "SECRET1", "track_id": 44,
        "bbox": (1, 2, 3, 4), "identity_status": "linked", "linked_via": "face",
        "is_walkin": False, "has_face_embedding": True,
    }
    at_counter = {**person, "queue_number": 11, "queue_label": "Q011",
                  "counter_number": 1, "access_token": "SECRET2"}
    monkeypatch.setattr(queue.queue_service.queue_tracker, "get_state", lambda: {
        "counter_assignments": [at_counter], "active_queue": [at_counter, person],
        "newly_called": [], "num_counters": 3, "queue_count": 2, "total_served": 0,
    })
    monkeypatch.setattr(queue.queue_service, "runtime_config", lambda: {"active_counters": 3})

    board = queue.queue_display()

    shown = board["active_queue"] + board["counter_assignments"]
    assert all(set(p) == set(queue._PUBLIC_QUEUE_FIELDS) for p in shown)
    assert "SECRET" not in repr(board)
    assert board["active_queue"][1]["position_in_line"] == 2
