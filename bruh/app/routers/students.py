from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from pydantic import BaseModel

from core.config import (
    CONSENT_VERSION,
    FACE_MARGIN_THRESHOLD,
    FACE_MATCH_THRESHOLD,
    FACE_MODEL_PACK,
    JOIN_INTENT_TIMEOUT_MINUTES,
)
from core.security import (
    clear_student_session,
    create_student_session,
    get_student_session_token,
    require_staff,
    require_student,
    verify_google_id_token,
)
from database.database_handler import (
    create_student_account,
    delete_student_profile,
    get_student_by_google_sub,
    get_student_by_id,
    get_student_by_school_id,
    list_students,
    record_face_match_event,
    record_student_consent,
    save_student_face_embedding,
    touch_student_activity,
    touch_student_last_login,
    update_student_department,
)
from services.department_config import DEPARTMENTS, normalize_department
from services import face_service
# Import the module, never the tracker object. queue_service.reset_queue()
# replaces queue_tracker with a fresh QueueTracker, and a name bound here at
# import time would keep pointing at the discarded one for the life of the
# process. That split the system in half after any staff reset: the app armed
# join intent on the dead tracker while the detector asked the live one, so
# every recognized student was written off as a bystander and no number was
# ever issued again. Reaching through the module re-reads the current binding.
from services import queue_service


router = APIRouter()


class GoogleAuthBody(BaseModel):
    id_token: str
    school_id: str | None = None
    # Version of the data-privacy notice the student agreed to in the app.
    consent_version: str | None = None
    department: str | None = None


class DepartmentBody(BaseModel):
    department: str


class ConsentBody(BaseModel):
    consent_version: str


def _consent_required(row: dict) -> bool:
    # Before the privacy migration has run the column is absent; nobody is
    # blocked then, and the requirement applies as soon as it exists.
    if "consent_version" not in row:
        return False
    return row.get("consent_version") != CONSENT_VERSION


def _student_public_profile(row: dict) -> dict:
    return {
        "student_id": row["id"],
        "school_id": row["school_id"],
        "full_name": row.get("full_name"),
        "department": row.get("department"),
        "has_face_embedding": bool(row.get("face_enrolled_at")),
        "consent_required": _consent_required(row),
        "consent_version": CONSENT_VERSION,
    }


def _require_consent(student_id: int) -> None:
    """Block anything that processes the student's face until they have
    agreed to the current data-privacy notice."""
    student = get_student_by_id(student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    if _consent_required(student):
        raise HTTPException(status_code=403, detail="consent_required")


def _check_consent_version(version: str | None) -> None:
    if version != CONSENT_VERSION:
        raise HTTPException(status_code=422, detail="consent_required")


@router.post(
    "/api/students/auth/google",
    summary="Student sign in / sign up via NCF Gbox Google account",
    tags=["Students"],
)
async def student_google_auth(body: GoogleAuthBody):
    """
    Self-service registration + login (per panel revision — no staff-operated
    enrollment). Google is the sole credential authority; no student password
    is ever stored. First sign-in with a given Google account creates the
    account (school_id required that one time to link it to the student's
    school record); every later sign-in with the same account logs in.
    """
    identity = verify_google_id_token(body.id_token)

    existing = get_student_by_google_sub(identity.sub)
    if existing is None:
        school_id = (body.school_id or "").strip()
        if not school_id:
            raise HTTPException(
                status_code=422,
                detail="school_id_required",
            )
        if get_student_by_school_id(school_id) is not None:
            raise HTTPException(
                status_code=409, detail="This school ID is already registered."
            )
        # No account is created until the student has agreed to the
        # data-privacy notice (personal data and face photo).
        _check_consent_version(body.consent_version)

        try:
            department = normalize_department(body.department)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        student_id = create_student_account(
            school_id=school_id,
            gbox_email=identity.email,
            google_sub=identity.sub,
            full_name=identity.name,
            consent_version=body.consent_version,
            department=department,
        )
        if student_id is None:
            raise HTTPException(status_code=500, detail="Failed to create student account")
        existing = get_student_by_id(student_id)
    elif body.consent_version == CONSENT_VERSION and _consent_required(existing):
        record_student_consent(existing["id"], CONSENT_VERSION)
        existing = get_student_by_id(existing["id"]) or existing

    touch_student_last_login(existing["id"])
    token = create_student_session(existing["id"])
    return {"session_token": token, **_student_public_profile(existing)}


@router.patch("/api/students/me/department", summary="Student - update own department", tags=["Students"])
def update_my_department(body: DepartmentBody, student_id: int = Depends(require_student)):
    try:
        department = normalize_department(body.department)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not update_student_department(student_id, department):
        raise HTTPException(status_code=503, detail="Could not update department. Please retry.")
    return {"success": True, "department": department}


@router.post(
    "/api/students/auth/face",
    summary="Student - log in with face alone (no Google sign-in)",
    tags=["Students"],
)
async def student_face_auth(photo: UploadFile = File(...)):
    """
    Quick-login alternative to Google sign-in, for a student who has already
    completed registration (school_id + Gbox account) and captured their
    face at least once. Identifies against every enrolled student (1:N) —
    unlike queue-zone matching this has no "already served today" filter,
    since it's authenticating account access, not linking a queue number.
    Uses the same accept-only-if-clearly-best philosophy as everywhere else
    face matching happens in this system.
    """
    data = await photo.read()
    embedding = face_service.compute_embedding_from_photo(data)
    if embedding is None:
        raise HTTPException(
            status_code=422,
            detail="No confident face detected - retake and retry",
        )

    match = queue_service.embedding_index.match(
        embedding,
        match_threshold=FACE_MATCH_THRESHOLD,
        margin_threshold=FACE_MARGIN_THRESHOLD,
    )
    record_face_match_event(match.student_id, None, match.score, match.margin, match.accepted)
    if not match.accepted:
        raise HTTPException(status_code=401, detail="Face not recognized - try Google sign-in instead")

    student = get_student_by_id(match.student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")

    touch_student_last_login(student["id"])
    token = create_student_session(student["id"])
    return {"session_token": token, **_student_public_profile(student)}


@router.post("/api/students/logout", summary="Student - log out", tags=["Students"])
async def student_logout(request: Request, student_id: int = Depends(require_student)):
    clear_student_session(get_student_session_token(request))
    return {"message": "Logged out"}


@router.post(
    "/api/students/me/consent",
    summary="Student - agree to the data-privacy notice",
    tags=["Students"],
)
async def give_consent(body: ConsentBody, student_id: int = Depends(require_student)):
    """For accounts created before the notice existed, or after its text
    changed: the app shows the notice and records the student's agreement."""
    _check_consent_version(body.consent_version)
    if not record_student_consent(student_id, CONSENT_VERSION):
        raise HTTPException(status_code=500, detail="Failed to record consent")
    student = get_student_by_id(student_id)
    return {"student": _student_public_profile(student)}


@router.delete(
    "/api/students/me",
    summary="Student - delete own account and face data",
    tags=["Students"],
)
async def delete_my_account(request: Request, student_id: int = Depends(require_student)):
    """Withdrawing consent: the student erases their own account, including
    their face data. Queue history keeps its rows with the student unlinked."""
    queue_service.queue_tracker.clear_join_intent(student_id)
    if not delete_student_profile(student_id):
        raise HTTPException(status_code=404, detail="Student not found")
    queue_service.embedding_index.invalidate()
    clear_student_session(get_student_session_token(request))
    return {"message": "Account deleted"}


@router.get("/api/students/me", summary="Student - own profile", tags=["Students"])
async def get_my_profile(student_id: int = Depends(require_student)):
    student = get_student_by_id(student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    return {"student": _student_public_profile(student)}


@router.post(
    "/api/students/me/join",
    summary="Student - ask to be issued a queue number on recognition",
    tags=["Students"],
)
async def join_queue(student_id: int = Depends(require_student)):
    """
    Arms this student for recognition. Being seen by the camera is not by
    itself a request to join the queue — without this, any registered
    student who merely walked past would be issued a number they never
    asked for. The intent lapses after JOIN_INTENT_TIMEOUT_MINUTES and is
    spent as soon as it produces a number.
    """
    _require_consent(student_id)
    touch_student_activity(student_id)
    armed_at = queue_service.queue_tracker.arm_join_intent(student_id)
    return {
        "joined": True,
        "armed_at": armed_at.isoformat(),
        "expires_in_minutes": JOIN_INTENT_TIMEOUT_MINUTES,
    }


@router.delete(
    "/api/students/me/join",
    summary="Student - cancel the request to be issued a queue number",
    tags=["Students"],
)
async def cancel_join_queue(student_id: int = Depends(require_student)):
    """Undo of join_queue(). No-op if nothing was armed."""
    queue_service.queue_tracker.clear_join_intent(student_id)
    return {"joined": False}


@router.get(
    "/api/students/me/queue",
    summary="Student - own active queue entry, if any",
    tags=["Students"],
)
async def get_my_queue_entry(student_id: int = Depends(require_student)):
    """
    Lets a logged-in student see their own live queue status without
    knowing a queue number or access token. Hands back the same
    queue_number + access_token the QR-based ticket-lookup flow uses, so
    the app can drive the existing status screens either way.

    The `state` field names where the student actually is in the flow. The
    three booleans alone could not distinguish "the camera has not seen you"
    from "the camera saw you and wrote you off as a bystander" from "the
    camera can see somebody but has not resolved who": all three produced the
    same empty response, so the app had nothing to show and appeared stuck on
    the join screen while the system was in fact working as designed. Each
    state below is a condition the student can act on.
    """
    tracker = queue_service.queue_tracker
    joined = tracker.has_join_intent(student_id)

    def payload(state: str, **extra) -> dict:
        return {
            "has_active_entry": False,
            "pending_link": False,
            "joined": joined,
            "state": state,
            **extra,
        }

    person = tracker.get_person_by_student_id(student_id)

    if person is None:
        # Recognized earlier, but no join intent was armed at that moment, so
        # nothing was issued. Arming the intent revives the track, but only
        # once the camera produces another face for it — until then the
        # student needs to know why nothing is happening.
        if tracker.get_bystander_by_student_id(student_id) is not None:
            return payload("recognized_not_joined" if not joined else "rejoining")
        if not joined:
            return payload("not_joined")
        # Joined and waiting. Distinguish "nobody is at the camera" from "the
        # camera is looking at someone it has not identified yet", which is
        # usually this student mid-recognition.
        if tracker.has_unidentified_person():
            return payload("identifying")
        return payload("waiting_for_camera")

    # A served/no-show person lingers in active_queue as 'done_pending' until
    # the done-cooldown clears them. Reporting that as still-active makes the
    # app's dashboard poll drag the student straight back into the "session
    # ended" screen they just dismissed, so treat it as finished here.
    if person.status == "done_pending":
        return payload("recently_served")

    if person.queue_number is None:
        return {
            "has_active_entry": False,
            "pending_link": True,
            "joined": joined,
            "state": "identifying",
        }

    return {
        "state": "active",
        "has_active_entry": True,
        "pending_link": False,
        "joined": joined,
        "queue_number": person.queue_number,
        "access_token": person.short_code or person.access_token,
    }


@router.post(
    "/api/students/me/face",
    summary="Student - register or update own face",
    tags=["Students"],
)
async def register_my_face(
    photos: list[UploadFile] = File(...),
    student_id: int = Depends(require_student),
):
    """
    Self-service face capture (Algorithm 4), step 2 of registration. Accepts
    1-5 photos taken at varied angle/lighting; only the averaged embedding is
    stored, never the photos. Can be called again later to re-register (e.g.
    after a poor initial capture).
    """
    _require_consent(student_id)
    if not photos:
        raise HTTPException(status_code=400, detail="At least one enrollment photo is required")

    embeddings = []
    for photo in photos:
        data = await photo.read()
        embedding = face_service.compute_embedding_from_photo(data)
        if embedding is not None:
            embeddings.append(embedding)

    if not embeddings:
        raise HTTPException(
            status_code=422,
            detail="No confident face detected in any submitted photo - retake and retry",
        )

    canonical = face_service.build_enrollment_embedding(embeddings)

    # Reject if this face already belongs to a DIFFERENT enrolled student —
    # stops one person from registering multiple accounts (e.g. for extra
    # queue slots, or to dodge today's done-blacklist). Re-registering your
    # own face (a fresh capture after a poor first attempt) is unaffected,
    # since the check excludes the current student_id. Same accept-only-if-
    # unambiguous logic used everywhere else face matching happens.
    duplicate_check = queue_service.embedding_index.match(
        canonical,
        match_threshold=FACE_MATCH_THRESHOLD,
        margin_threshold=FACE_MARGIN_THRESHOLD,
        exclude_id=student_id,
    )
    if duplicate_check.accepted:
        raise HTTPException(
            status_code=409,
            detail="This face is already registered under another account.",
        )

    saved = save_student_face_embedding(
        student_id=student_id,
        embedding_json=face_service.embedding_to_json(canonical),
        embedding_model_version=FACE_MODEL_PACK,
        num_samples=len(embeddings),
    )
    if not saved:
        raise HTTPException(status_code=500, detail="Failed to save face enrollment")
    queue_service.embedding_index.invalidate()

    return {
        "student_id": student_id,
        "samples_used": len(embeddings),
        "samples_submitted": len(photos),
    }


@router.get("/api/students", summary="Staff - list registered students", tags=["Students"])
async def get_students(username: str = Depends(require_staff)):
    return {"students": list_students()}


@router.get("/api/students/{school_id}", summary="Staff - look up a student", tags=["Students"])
async def get_student(school_id: str, username: str = Depends(require_staff)):
    student = get_student_by_school_id(school_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    return {"student": student}


@router.delete(
    "/api/students/{student_id}",
    summary="Staff - remove a student's registration",
    tags=["Students"],
)
async def remove_student(student_id: int, username: str = Depends(require_staff)):
    """Hard delete, supporting Data Privacy Act erasure requests."""
    if not delete_student_profile(student_id):
        raise HTTPException(status_code=404, detail="Student not found")
    queue_service.embedding_index.invalidate()
    return {"message": "Student profile deleted"}
