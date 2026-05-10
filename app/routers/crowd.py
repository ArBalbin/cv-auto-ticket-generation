import asyncio

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse

import state
from core.security import require_staff


router = APIRouter()


@router.get("/api/stats", summary="Current crowd stats", tags=["Crowd"])
async def get_stats():
    return state.crowd_stats()


@router.get("/api/crowd/data", summary="Alias for /api/stats", tags=["Crowd"])
async def get_crowd_data():
    return state.crowd_stats()


@router.get("/api/crowd/video", summary="MJPEG annotated camera stream", tags=["Crowd"])
async def get_video_stream():
    async def _generate():
        last_seq = 0
        while True:
            frame, seq = await asyncio.to_thread(state.wait_for_snapshot, last_seq, 2.0)
            if frame is None:
                frame = state.get_snapshot()
                if frame is None:
                    await asyncio.sleep(0.5)
                    continue
            last_seq = seq
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/api/snapshot", summary="Latest annotated JPEG frame", tags=["Crowd"])
async def get_snapshot():
    snapshot = state.get_snapshot()
    if snapshot is None:
        raise HTTPException(status_code=503, detail="No snapshot available yet")
    return Response(
        content=snapshot,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/api/history", summary="Rolling crowd count history", tags=["Crowd"])
async def get_history():
    return state.history_data()


@router.get("/api/data", summary="Full current data - staff only", tags=["Crowd"])
async def get_data(username: str = Depends(require_staff)):
    return state.full_data()
