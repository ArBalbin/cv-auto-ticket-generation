from fastapi import APIRouter, Depends, HTTPException, Response

import state
from core.security import require_staff


router = APIRouter()


@router.get("/api/stats", summary="Current crowd stats", tags=["Crowd"])
async def get_stats():
    return state.crowd_stats()


@router.get("/api/crowd/data", summary="Alias for /api/stats", tags=["Crowd"])
async def get_crowd_data():
    return state.crowd_stats()


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
