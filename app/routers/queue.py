import csv
from datetime import datetime, timedelta
from io import StringIO

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

import state
from database.database_handler import fetch_completed_transactions, get_db_pool
from core.security import require_staff
from services import queue_service, wait_accuracy
from services.ticket_printer import TicketLookupBusy, get_ticket_record_by_short_code


router = APIRouter()


class DoneBody(BaseModel):
    queue_number: int


class ForceNewBody(BaseModel):
    queue_number: int
    is_walkin: bool = True


class LinkPendingBody(BaseModel):
    track_id: int
    queue_number: int
    student_id: int | None = None


class OnWayBody(BaseModel):
    queue_number: int
    token: str


class ZoneBody(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class CountersBody(BaseModel):
    counters: int


class WindowBody(BaseModel):
    counter_number: int


class CallModeBody(BaseModel):
    mode: str  # "staff" (Call next) or "auto"


class CloseCountersBody(BaseModel):
    until: str  # 24-hour clock today, e.g. "13:00"


# What the public display board may show about each person in line. Ticket
# codes, camera track ids and boxes, and identity details stay staff-only: a
# queue number plus its ticket code is all /api/queue/status asks for.
_PUBLIC_QUEUE_FIELDS = (
    "queue_number", "queue_label", "status", "position_in_line",
    "counter_number", "wait_time", "wait_time_seconds",
)


def _public_entry(entry: dict) -> dict:
    return {key: entry.get(key) for key in _PUBLIC_QUEUE_FIELDS}


@router.get("/api/queue/list", summary="Live queue state - staff only", tags=["Queue"])
def queue_list(username: str = Depends(require_staff)):
    state_dict = dict(queue_service.queue_tracker.get_state())
    # Pending (not-yet-linked) people are a staff-only concern — see
    # /api/queue/pending and /api/queue/data for that detail.
    state_dict.pop("pending_queue", None)
    state_dict.pop("pending_link_alerts", None)
    return state_dict


@router.get("/api/queue/display", summary="Queue display board data - PUBLIC", tags=["Queue"])
def queue_display():
    state = queue_service.queue_tracker.get_state()
    config = queue_service.runtime_config()
    return {
        "counter_assignments": [_public_entry(p) for p in state["counter_assignments"]],
        "active_queue":        [_public_entry(p) for p in state["active_queue"]],
        "newly_called":        state["newly_called"],
        "num_counters":        state["num_counters"],
        "queue_count":         state["queue_count"],
        "total_served":        state["total_served"],
        "active_counters":     config.get("active_counters", state["num_counters"]),
    }


@router.get("/api/queue/status", summary="Ticket holder check - PUBLIC", tags=["Queue"])
def queue_status(q: int, token: str):
    db_pool = get_db_pool()
    if db_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable.",
        )
    try:
        ticket_record = get_ticket_record_by_short_code(token, q, db_pool)
    except TicketLookupBusy:
        raise HTTPException(status_code=503, detail="Server is busy. Please try again shortly.")
    if not ticket_record:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token. Please check your ticket.",
        )
    if ticket_record.get("status") != "waiting":
        return queue_service.ticket_record_status_response(ticket_record)

    result = queue_service.queue_tracker.lookup_by_token(q, token)
    if result is None:
        return queue_service.ticket_record_waiting_response(ticket_record)
    if result.get("error") == "invalid_token":
        raise HTTPException(
            status_code=403,
            detail="Invalid token. Please check your ticket.",
        )

    # A skipped person stays 'done_pending' in the tracker while they walk
    # away, as a served one does; until the database row says 'no_show' too,
    # tell the app here, or it shows "transaction complete" or nothing at all.
    if result.get("status") == "done_pending" and result.get("retired_as") == "no_show":
        result["status"] = "no_show"
        result["message"] = queue_service.NO_SHOW_MESSAGE

    position = queue_service.as_int(result.get("position_in_line"), 0)
    config = queue_service.runtime_config()
    result["prediction"] = queue_service.prediction_for_position(
        position,
        config["avg_service_time"],
        config["active_counters"],
    )
    result.pop("access_token", None)
    result.pop("jwt_token", None)
    return result


@router.post(
    "/api/queue/on_way",
    summary="Ticket holder - notify staff you are on the way",
    tags=["Queue"],
)
def queue_on_way(body: OnWayBody):
    db_pool = get_db_pool()
    if db_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable.",
        )

    token = body.token.strip().upper()
    try:
        ticket_record = get_ticket_record_by_short_code(
            token,
            body.queue_number,
            db_pool,
        )
    except TicketLookupBusy:
        raise HTTPException(status_code=503, detail="Server is busy. Please try again shortly.")
    if not ticket_record:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token. Please check your ticket.",
        )
    if ticket_record.get("status") != "waiting":
        raise HTTPException(
            status_code=409,
            detail="This queue ticket is no longer waiting.",
        )

    result = queue_service.queue_tracker.lookup_by_token(body.queue_number, token)
    if result is None:
        notification = queue_service.record_on_the_way_signal(body.queue_number)
        print(f"[QueueAPI] Q{body.queue_number:03d} on-way signal accepted")
        return {
            "success": True,
            "message": "Staff has been notified that you are on the way.",
            "notification": notification,
        }
    if result.get("error") == "invalid_token":
        raise HTTPException(
            status_code=403,
            detail="Invalid token. Please check your ticket.",
        )
    person = queue_service.mark_on_the_way(body.queue_number)
    if person is None:
        notification = queue_service.record_on_the_way_signal(body.queue_number)
        print(f"[QueueAPI] Q{body.queue_number:03d} on-way signal accepted")
        return {
            "success": True,
            "message": "Staff has been notified that you are on the way.",
            "notification": notification,
        }
    person.pop("access_token", None)
    print(f"[QueueAPI] Q{body.queue_number:03d} on-way signal accepted")
    return {
        "success": True,
        "message": "Staff has been notified that you are on the way.",
        "queue": person,
    }


@router.get(
    "/api/queue/prediction",
    summary="Current wait-time prediction - PUBLIC",
    tags=["Queue"],
)
def queue_prediction():
    return queue_service.build_queue_prediction()


@router.get(
    "/api/queue/analytics",
    summary="Queue analytics - staff only",
    tags=["Queue"],
)
def queue_analytics(username: str = Depends(require_staff)):
    return queue_service.build_queue_analytics()


@router.get(
    "/api/queue/prediction_accuracy",
    summary="How close the wait estimates were to real waits - staff only",
    tags=["Queue"],
)
def queue_prediction_accuracy(username: str = Depends(require_staff)):
    return wait_accuracy.current_summary()


# Columns of the service-records download, in order. Times and counts only:
# no names, school IDs or student links.
SERVICE_RECORD_FIELDS = (
    "ticket_created_at", "service_started_at", "service_completed_at",
    "actual_wait_seconds", "actual_service_seconds", "people_ahead_at_issue",
    "active_counters_at_issue", "cashier_window", "payment_type",
    "predicted_wait_min", "wait_model_version",
)


@router.get(
    "/api/queue/service_records.csv",
    summary="Staff - download served tickets' real waits and service times (CSV)",
    tags=["Queue"],
)
def queue_service_records(username: str = Depends(require_staff)):
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=SERVICE_RECORD_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(fetch_completed_transactions())
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="queuex_service_records.csv"',
            "Cache-Control": "no-store",
        },
    )


@router.get(
    "/api/queue/on_way_notifications",
    summary="Staff - customer on-way notifications",
    tags=["Queue"],
)
def queue_on_way_notifications(username: str = Depends(require_staff)):
    return queue_service.on_way_notification_state()


@router.get("/api/queue/zone", summary="Get queue zone - PUBLIC", tags=["Queue"])
def get_zone():
    return queue_service.zone_dict()


@router.get(
    "/api/queue/data",
    summary="Combined crowd + prediction + queue state - staff only",
    tags=["Queue"],
)
def queue_data(username: str = Depends(require_staff)):
    return {
        **state.crowd_prediction_fields(),
        **queue_service.queue_tracker.get_state(),
    }


@router.post("/api/queue/force-new", summary="Staff - manually link an already-printed kiosk number (walk-in entry / CV-miss override)", tags=["Queue"])
def queue_force_new(body: ForceNewBody, username: str = Depends(require_staff)):
    entry = queue_service.force_new_person(body.queue_number, is_walkin=body.is_walkin)
    if entry is None:
        raise HTTPException(status_code=409, detail=f"Q{body.queue_number:03d} is already linked today")
    return {
        "success": True,
        "message": f"Q{entry['queue_number']:03d} manually added",
        "person": entry,
    }


@router.get(
    "/api/queue/pending",
    summary="Staff - people confirmed present but not yet linked to a kiosk number",
    tags=["Queue"],
)
def pending_queue(username: str = Depends(require_staff)):
    state_dict = queue_service.queue_tracker.get_state()
    return {
        "pending_queue": state_dict["pending_queue"],
        "pending_count": state_dict["pending_count"],
        "pending_link_alerts": state_dict["pending_link_alerts"],
    }


@router.post(
    "/api/queue/link-pending",
    summary="Staff - manually resolve a pending person to a printed kiosk number",
    tags=["Queue"],
)
def link_pending(body: LinkPendingBody, username: str = Depends(require_staff)):
    person = queue_service.link_pending_person(
        body.track_id, body.queue_number, student_id=body.student_id,
    )
    if person is None:
        raise HTTPException(
            status_code=409,
            detail="Could not link - track is not pending, or that number is already linked today",
        )
    return {
        "success": True,
        "message": f"Q{body.queue_number:03d} linked",
        "person": person,
    }


@router.get("/api/queue/windows", summary="Staff - the windows and who holds each", tags=["Queue"])
def queue_windows(username: str = Depends(require_staff)):
    return queue_service.window_overview(username)


@router.post("/api/queue/window/claim", summary="Staff - take a window for this account", tags=["Queue"])
def queue_claim_window(body: WindowBody, username: str = Depends(require_staff)):
    try:
        queue_service.claim_window(username, body.counter_number)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except queue_service.WindowTaken as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return queue_service.window_overview(username)


@router.post("/api/queue/window/release", summary="Staff - give up this account's window", tags=["Queue"])
def queue_release_window(username: str = Depends(require_staff)):
    queue_service.release_window(username)
    return queue_service.window_overview(username)


@router.post("/api/queue/call_mode", summary="Staff - switch between Call next and automatic calling", tags=["Queue"])
def queue_call_mode(body: CallModeBody, username: str = Depends(require_staff)):
    try:
        queue_service.set_call_mode(body.mode.strip().lower(), username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return queue_service.window_overview(username)


@router.post("/api/queue/call_next", summary="Staff - call the next number to my window", tags=["Queue"])
def queue_call_next(body: WindowBody, username: str = Depends(require_staff)):
    try:
        outcome, person = queue_service.call_next(body.counter_number, username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except queue_service.WindowTaken as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    if outcome == "called":
        return {"called": person, "message": f"{person['queue_label']} called to window {body.counter_number}."}
    if outcome == "empty":
        return {"called": None, "message": "No one is waiting."}
    if outcome == "busy":
        raise HTTPException(
            status_code=409,
            detail=f"Window {body.counter_number} is still serving {person['queue_label']}. "
                   "Mark it Done or No-show first.",
        )
    if outcome == "closed":
        until = queue_service.counters_closed_until()
        when = queue_service.school_time_label(until) if until else "later"
        raise HTTPException(status_code=409, detail=f"All windows are closed until {when}.")
    raise HTTPException(status_code=400, detail=f"There is no window {body.counter_number}.")


@router.post("/api/queue/done", summary="Staff - mark done", tags=["Queue"])
def queue_done(body: DoneBody, username: str = Depends(require_staff)):
    queue_state = queue_service.mark_done(body.queue_number, username)
    if queue_state:
        return {
            "success": True,
            "message": f"Q{body.queue_number:03d} completed",
            "queue_state": queue_state,
        }
    raise HTTPException(
        status_code=404,
        detail=f"Q{body.queue_number:03d} not found in active queue",
    )


@router.post("/api/queue/no_show", summary="Staff - mark no-show", tags=["Queue"])
def queue_no_show(body: DoneBody, username: str = Depends(require_staff)):
    """Staff decision, not a detection. Nothing in the system watches a
    countdown or removes anybody on its own; this endpoint exists so the
    person at the desk can say that the student whose turn it is has not
    arrived, and have the queue move on."""
    queue_state = queue_service.mark_no_show(body.queue_number, username)
    if queue_state:
        return {
            "success": True,
            "message": f"Q{body.queue_number:03d} marked as no-show",
            "queue_state": queue_state,
        }
    raise HTTPException(
        status_code=404,
        detail=f"Q{body.queue_number:03d} not found in active queue",
    )


@router.post("/api/queue/reset", summary="Staff - reset entire queue", tags=["Queue"])
def queue_reset(username: str = Depends(require_staff)):
    queue_service.reset_queue(username)
    return {"success": True, "message": "Queue reset successfully"}


@router.post("/api/queue/zone", summary="Staff - update queue zone", tags=["Queue"])
def set_zone(body: ZoneBody, username: str = Depends(require_staff)):
    zone = queue_service.set_zone(body.x1, body.y1, body.x2, body.y2)
    return {"success": True, "zone": zone}


@router.post(
    "/api/queue/adjust_counters",
    summary="Staff - adjust service counters",
    tags=["Queue"],
)
def adjust_counters(
    body: CountersBody,
    username: str = Depends(require_staff),
):
    if not (1 <= body.counters <= 10):
        raise HTTPException(
            status_code=400,
            detail="counters must be between 1 and 10",
        )
    config = queue_service.set_active_counters(body.counters, username)
    return {"success": True, "counters": config["active_counters"]}


@router.post(
    "/api/queue/close_counters",
    summary="Staff - close every counter until a time today (e.g. lunch)",
    tags=["Queue"],
)
def close_counters(
    body: CloseCountersBody,
    username: str = Depends(require_staff),
):
    try:
        clock = datetime.strptime(body.until.strip(), "%H:%M").time()
    except ValueError:
        raise HTTPException(status_code=400, detail="until must be a time like 13:00")
    # The time is on the school's clock; the server may be on UTC.
    until = queue_service.school_time_today(clock)
    if not datetime.now() < until <= datetime.now() + timedelta(hours=12):
        raise HTTPException(status_code=400, detail="until must be later today")
    closed = queue_service.set_counters_closed_until(until)
    print(f"[Queue] Counters closed until {body.until} (school time) by {username}")
    return {"success": True, "closed_until": queue_service.school_iso(closed)}


@router.post(
    "/api/queue/open_counters",
    summary="Staff - reopen the counters before the closure ends",
    tags=["Queue"],
)
def open_counters(username: str = Depends(require_staff)):
    queue_service.set_counters_closed_until(None)
    print(f"[Queue] Counters reopened by {username}")
    return {"success": True, "closed_until": None}


@router.get(
    "/api/queue/appearance_log",
    summary="Staff - appearance rejection log",
    tags=["Queue"],
)
def appearance_log(username: str = Depends(require_staff)):
    return {"rejections": queue_service.queue_tracker.appearance_rejections}
