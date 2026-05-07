from datetime import datetime

from fastapi import APIRouter

import state
from core.config import APP_ENV, PORTAL_BASE_URL
from database.database_handler import is_database_available
from services.cache_service import is_available as is_cache_available
from services.cache_service import is_configured as is_cache_configured


router = APIRouter()


@router.get("/health", summary="Health check", tags=["System"])
async def health():
    return {
        "status": "ok",
        "environment": APP_ENV,
        "portal_base_url": PORTAL_BASE_URL,
        "db": is_database_available(),
        "cache": {
            "configured": is_cache_configured(),
            "available": is_cache_available(),
        },
        "snapshot": state.has_snapshot(),
        "timestamp": datetime.now().isoformat(),
    }
