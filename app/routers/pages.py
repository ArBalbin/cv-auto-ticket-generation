from fastapi import APIRouter, Form, HTTPException, Request, Response
from fastapi.responses import RedirectResponse, StreamingResponse

import state
from core.config import MJPEG_HEADERS, TEMPLATES_DIR
from core.security import (
    authenticate_user,
    clear_session,
    create_session,
    delete_session_cookie,
    get_session_token,
    get_session_username,
    get_user_profile,
    is_authenticated,
    register_staff_user,
    registration_code_is_required,
    registration_is_enabled,
    set_session_cookie,
    touch_last_login,
)


router = APIRouter()
templates = None


def get_templates():
    global templates
    if templates is None:
        from fastapi.templating import Jinja2Templates

        templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    return templates


@router.get("/", tags=["Dashboard"])
async def home(request: Request):
    if is_authenticated(request):
        return RedirectResponse(url="/dashboard/computer-vision", status_code=302)
    return RedirectResponse(url="/login", status_code=302)


@router.get("/login", tags=["Dashboard"])
async def login_page(request: Request):
    if is_authenticated(request):
        return RedirectResponse(url="/dashboard/computer-vision", status_code=302)
    return get_templates().TemplateResponse(
        request,
        "login.html",
        {
            "request": request,
            "registration_enabled": registration_is_enabled(),
        },
    )


@router.get("/register", tags=["Dashboard"])
async def register_page(request: Request):
    if is_authenticated(request):
        return RedirectResponse(url="/dashboard/computer-vision", status_code=302)
    if not registration_is_enabled():
        return RedirectResponse(url="/login", status_code=302)
    return get_templates().TemplateResponse(
        request,
        "register.html",
        {
            "request": request,
            "registration_code_required": registration_code_is_required(),
        },
    )


@router.post("/register", tags=["Dashboard"])
async def register_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    full_name: str = Form(""),
    registration_code: str = Form(""),
):
    if password != confirm_password:
        return get_templates().TemplateResponse(
            request,
            "register.html",
            {
                "request": request,
                "error": "Passwords do not match.",
                "username": username,
                "full_name": full_name,
                "registration_code_required": registration_code_is_required(),
            },
        )

    try:
        user = register_staff_user(
            username=username,
            password=password,
            full_name=full_name,
            registration_code=registration_code,
        )
    except HTTPException as exc:
        return get_templates().TemplateResponse(
            request,
            "register.html",
            {
                "request": request,
                "error": exc.detail,
                "username": username,
                "full_name": full_name,
                "registration_code_required": registration_code_is_required(),
            },
            status_code=exc.status_code,
        )

    token = create_session(user["username"])
    touch_last_login(user["username"])

    redir = RedirectResponse(url="/dashboard/computer-vision", status_code=303)
    set_session_cookie(redir, token)
    return redir


@router.post("/login", tags=["Dashboard"])
async def login_submit(
    request: Request,
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
):
    username = username.strip()
    try:
        authenticate_user(username, password)
    except HTTPException as exc:
        if exc.status_code == 401:
            error = "Invalid username or password."
        elif exc.status_code == 403:
            error = "Account is disabled. Contact your administrator."
        elif exc.status_code == 503:
            error = "Database unavailable. Please try again later."
        else:
            error = "Database error. Please try again."
        return get_templates().TemplateResponse(
            request,
            "login.html",
            {
                "request": request,
                "error": error,
                "registration_enabled": registration_is_enabled(),
            },
        )

    token = create_session(username)
    touch_last_login(username)

    redir = RedirectResponse(url="/dashboard/computer-vision", status_code=303)
    set_session_cookie(redir, token)
    print(f"[Login] {username} logged in")
    return redir


@router.get("/logout", tags=["Dashboard"])
async def logout(request: Request):
    token = get_session_token(request)
    if token:
        username = clear_session(token) or "unknown"
        print(f"[Logout] {username} logged out")
    redir = RedirectResponse(url="/login", status_code=302)
    delete_session_cookie(redir)
    return redir


@router.get("/dashboard/computer-vision", tags=["Dashboard"])
async def cv_dashboard(request: Request):
    if not is_authenticated(request):
        return RedirectResponse(url="/login", status_code=302)
    username = get_session_username(request)
    return get_templates().TemplateResponse(
        request,
        "index.html",
        {"request": request, "username": username},
    )


@router.get("/dashboard/queueflow", tags=["Dashboard"])
async def queueflow_dashboard(request: Request):
    if not is_authenticated(request):
        return RedirectResponse(url="/login", status_code=302)
    username = get_session_username(request)
    return get_templates().TemplateResponse(
        request,
        "queueflow_dashboard.html",
        {"request": request, "username": username},
    )


@router.get("/dashboard/queue-analytics", tags=["Dashboard"])
async def queue_analytics_page(request: Request):
    if not is_authenticated(request):
        return RedirectResponse(url="/login", status_code=302)
    username = get_session_username(request)
    return get_templates().TemplateResponse(
        request,
        "queue_analytics.html",
        {"request": request, "username": username},
    )


@router.get("/dashboard/profile", tags=["Dashboard"])
async def profile_page(request: Request):
    if not is_authenticated(request):
        return RedirectResponse(url="/login", status_code=302)

    username = get_session_username(request)
    error = None
    profile = {"username": username}
    try:
        profile = get_user_profile(username)
    except HTTPException as exc:
        error = exc.detail

    return get_templates().TemplateResponse(
        request,
        "profile.html",
        {
            "request": request,
            "username": username,
            "profile": profile,
            "error": error,
        },
    )


def mjpeg_generator():
    last_seq = -1
    while True:
        snap, seq = state.wait_for_snapshot(last_seq, timeout=2.0)
        if snap is None or seq == last_seq:
            continue
        last_seq = seq

        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + snap
            + b"\r\n"
        )


@router.get("/video", tags=["Dashboard"])
async def video_stream(request: Request):
    if not is_authenticated(request):
        return RedirectResponse(url="/login", status_code=302)
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=MJPEG_HEADERS,
    )


@router.get("/api/crowd/video", tags=["Dashboard"])
async def api_crowd_video(request: Request):
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=MJPEG_HEADERS,
    )
