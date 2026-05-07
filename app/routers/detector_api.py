from fastapi import APIRouter, Body, Depends

import state
from core.security import verify_cam_token
from services import queue_service


router = APIRouter()


@router.post(
    "/yolo/push-frame",
    summary="Detector pushes crowd + queue metadata",
    tags=["Detector"],
)
def push_frame(body: dict = Body(...), _=Depends(verify_cam_token)):
    state.update_from_detector_payload(body)

    queue_state = queue_service.process_tracked_persons(
        raw_tracked=body.get("tracked_persons", []),
        yolo_frame_idx=body.get("yolo_frame_idx", 0),
    )

    return {
        "ok": True,
        "queue_state": queue_state,
        "done_pending": queue_service.done_pending_people(),
        "config": queue_service.runtime_config(),
    }


@router.post(
    "/yolo/update",
    summary="Detector pushes annotated JPEG snapshot",
    tags=["Detector"],
)
def push_snapshot(
    snapshot: bytes = Body(..., media_type="image/jpeg"),
    _=Depends(verify_cam_token),
):
    state.set_snapshot(snapshot)
    return {"ok": True}
