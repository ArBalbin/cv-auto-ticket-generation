from fastapi import APIRouter, Body, Depends, Header

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
    # The detector's frame counter. Optional: a detector that does not send it
    # keeps the old behaviour of accepting every frame in arrival order.
    x_snap_seq: int | None = Header(default=None, alias="X-SNAP-SEQ"),
    _=Depends(verify_cam_token),
):
    accepted = state.set_snapshot(snapshot, source_seq=x_snap_seq)
    return {"ok": True, "accepted": accepted}
