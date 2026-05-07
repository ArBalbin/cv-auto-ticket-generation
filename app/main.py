#!/usr/bin/env python3
import os
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
for import_path in (str(APP_ROOT), str(PROJECT_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from core.config import (
    API_HOST,
    API_PORT,
    APP_ENV,
    CORS_ORIGINS,
    TRUSTED_HOSTS,
    validate_cloud_config,
)
from database.database_handler import warm_up_db_pool
from routers import auth, crowd, detector_api, health, pages, queue


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_cloud_config()
    warm_up_db_pool()
    yield


app = FastAPI(
    title="QueueFlow API",
    description="FastAPI backend for QueueFlow crowd monitoring + queue management",
    version="2.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials="*" not in CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

if TRUSTED_HOSTS != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)


HTTP_LABELS = {
    400: "Bad request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not found",
    405: "Method not allowed",
    422: "Unprocessable entity",
    500: "Internal server error",
    503: "Service unavailable",
}


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    if request.url.path.startswith("/api") or request.url.path.startswith("/yolo"):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": HTTP_LABELS.get(exc.status_code, "Error"),
                "message": exc.detail,
            },
        )
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    print(f"[API] Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Something went wrong. Please try again.",
        },
    )


app.include_router(pages.router)
app.include_router(auth.router)
app.include_router(detector_api.router)
app.include_router(crowd.router)
app.include_router(queue.router)
app.include_router(health.router)


def port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


if __name__ == "__main__":
    import uvicorn

    api_port = API_PORT
    if not port_is_available(API_HOST, api_port):
        print(f"[API] Port {api_port} is already in use.")
        print("[API] Close the previous backend terminal, then run main.py again.")
        print(f"[API] To find the process: Get-NetTCPConnection -LocalPort {api_port} -State Listen")
        sys.exit(1)

    reload_enabled = os.getenv("API_RELOAD", "0").strip() == "1"
    reload_options = {}
    if reload_enabled:
        reload_options = {
            "reload_dirs": [str(APP_ROOT)],
            "reload_excludes": [
                ".venv/*",
                "../.venv/*",
                "__pycache__/*",
                "*.pyc",
            ],
        }

    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=api_port,
        reload=reload_enabled,
        access_log=os.getenv("API_ACCESS_LOG", "0").strip() == "1",
        proxy_headers=True,
        forwarded_allow_ips=os.getenv("FORWARDED_ALLOW_IPS", "*"),
        server_header=APP_ENV != "production",
        **reload_options,
    )
