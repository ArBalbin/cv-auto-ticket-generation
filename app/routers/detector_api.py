from fastapi import APIRouter, Depends, Request

import state
from core.security import verify_cam_token
from services import queue_service


router = APIRouter()


@router.post(
    "/yolo/push-frame",
    summary="Detector pushes crowd + queue metadata",
    tags=["Detector"],
)
async def push_frame(request: Request, _=Depends(verify_cam_token)):
    body = await request.json()
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
async def push_snapshot(request: Request, _=Depends(verify_cam_token)):
    state.set_snapshot(await request.body())
    return {"ok": True}
