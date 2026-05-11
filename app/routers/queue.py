from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import state
from database.database_handler import get_db_pool
from core.security import require_staff
from services import queue_service
from services.ticket_printer import get_ticket_record_by_short_code


router = APIRouter()


class DoneBody(BaseModel):
    queue_number: int


class OnWayBody(BaseModel):
    queue_number: int
    token: str


class ZoneBody(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class NoshowConfigBody(BaseModel):
    seconds: int


class CountersBody(BaseModel):
    counters: int


@router.get("/api/queue/list", summary="Live queue state - PUBLIC", tags=["Queue"])
def queue_list():
    return queue_service.queue_tracker.get_state()


@router.get("/api/queue/display", summary="Queue display board data - PUBLIC", tags=["Queue"])
def queue_display():
    state = queue_service.queue_tracker.get_state()
    config = queue_service.runtime_config()
    return {
        "counter_assignments": state["counter_assignments"],
        "active_queue":        state["active_queue"],
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
    ticket_record = get_ticket_record_by_short_code(token, q, db_pool)
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
    ticket_record = get_ticket_record_by_short_code(
        token,
        body.queue_number,
        db_pool,
    )
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


@router.post("/api/queue/force-new", summary="Staff - manually add a queue entry (twin/CV-miss override)", tags=["Queue"])
def queue_force_new(username: str = Depends(require_staff)):
    entry = queue_service.force_new_person()
    return {
        "success": True,
        "message": f"Q{entry['queue_number']:03d} manually added",
        "person": entry,
    }


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


@router.post("/api/queue/reset", summary="Staff - reset entire queue", tags=["Queue"])
def queue_reset(username: str = Depends(require_staff)):
    queue_service.reset_queue(username)
    return {"success": True, "message": "Queue reset successfully"}


@router.post("/api/queue/zone", summary="Staff - update queue zone", tags=["Queue"])
def set_zone(body: ZoneBody, username: str = Depends(require_staff)):
    zone = queue_service.set_zone(body.x1, body.y1, body.x2, body.y2)
    return {"success": True, "zone": zone}


@router.get("/api/queue/noshow_alerts", summary="Staff - no-show alerts", tags=["Queue"])
def noshow_alerts(username: str = Depends(require_staff)):
    return {"alerts": queue_service.queue_tracker.get_noshow_alerts()}


@router.get("/api/queue/noshow_config", summary="Staff - get no-show window", tags=["Queue"])
def get_noshow_config(username: str = Depends(require_staff)):
    return {
        "noshow_window_seconds": queue_service.queue_tracker.NOSHOW_WINDOW_SECONDS
    }


@router.post("/api/queue/noshow_config", summary="Staff - set no-show window", tags=["Queue"])
def set_noshow_config(
    body: NoshowConfigBody,
    username: str = Depends(require_staff),
):
    if not (30 <= body.seconds <= 300):
        raise HTTPException(
            status_code=400,
            detail="seconds must be between 30 and 300",
        )
    queue_service.queue_tracker.NOSHOW_WINDOW_SECONDS = body.seconds
    return {"success": True, "noshow_window_seconds": body.seconds}


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


@router.get(
    "/api/queue/appearance_log",
    summary="Staff - appearance rejection log",
    tags=["Queue"],
)
def appearance_log(username: str = Depends(require_staff)):
    return {"rejections": queue_service.queue_tracker.appearance_rejections}
