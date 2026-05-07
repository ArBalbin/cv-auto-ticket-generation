from fastapi import APIRouter, Depends, Response, Request

from core.security import (
    LoginBody,
    RegisterBody,
    authenticate_user,
    clear_session,
    create_session,
    delete_session_cookie,
    get_user_profile,
    get_session_token,
    register_staff_user,
    require_staff,
    set_session_cookie,
    touch_last_login,
)


router = APIRouter()


@router.post("/api/auth/login", summary="React JSON login", tags=["Auth"])
async def api_login(body: LoginBody, response: Response):
    username = body.username.strip()
    authenticate_user(username, body.password)
    token = create_session(username)
    set_session_cookie(response, token)
    return {"access_token": token, "user": {"id": "1", "username": username}}


@router.post("/api/auth/register", summary="Staff JSON registration", tags=["Auth"])
async def api_register(body: RegisterBody, response: Response):
    user = register_staff_user(
        username=body.username,
        password=body.password,
        full_name=body.full_name,
        registration_code=body.registration_code,
    )
    token = create_session(user["username"])
    set_session_cookie(response, token)
    touch_last_login(user["username"])
    return {"access_token": token, "user": {"id": "1", **user}}


@router.get("/api/auth/me", summary="Session check", tags=["Auth"])
async def api_me(username: str = Depends(require_staff)):
    return {"userId": "1", "username": username}


@router.get("/api/auth/profile", summary="Current user profile", tags=["Auth"])
async def api_profile(username: str = Depends(require_staff)):
    return {"user": get_user_profile(username)}


@router.post("/api/auth/logout", summary="Invalidate session", tags=["Auth"])
async def api_logout(request: Request, response: Response):
    clear_session(get_session_token(request))
    delete_session_cookie(response)
    return {"message": "Logged out successfully"}
