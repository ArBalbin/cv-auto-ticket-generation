from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.security import require_staff
from database.database_handler import get_student_by_school_id
from services import retention_service

router = APIRouter()


class RunBody(BaseModel):
    dry_run: bool = False


class SelectionBody(BaseModel):
    student_ids: list[int] = []
    school_ids: list[str] = []


def _resolve(body: SelectionBody) -> tuple[list[int], list[str]]:
    """Accept database IDs, school IDs (e.g. a pasted list of graduates), or
    both. Returns the IDs to act on and the school IDs that were not found."""
    ids = list(dict.fromkeys(body.student_ids))
    missing = []
    for school_id in dict.fromkeys(s.strip() for s in body.school_ids if s.strip()):
        student = get_student_by_school_id(school_id)
        if student is None:
            missing.append(school_id)
        elif student["id"] not in ids:
            ids.append(student["id"])
    if not ids and not missing:
        raise HTTPException(status_code=400, detail="No students selected")
    return ids, missing


@router.get("/api/retention", summary="Staff - data-retention policy and due accounts", tags=["Retention"])
async def get_retention(username: str = Depends(require_staff)):
    return retention_service.overview()


@router.post("/api/retention/run", summary="Staff - apply the retention policy now", tags=["Retention"])
async def run_retention(body: RunBody, username: str = Depends(require_staff)):
    return retention_service.run(dry_run=body.dry_run, actor=username)


@router.post("/api/retention/archive", summary="Staff - archive students (e.g. graduates)", tags=["Retention"])
async def archive_students(body: SelectionBody, username: str = Depends(require_staff)):
    ids, missing = _resolve(body)
    done = retention_service.archive(ids)
    return {"archived": done, "not_found": missing}


@router.post("/api/retention/delete", summary="Staff - permanently delete students", tags=["Retention"])
async def delete_students(body: SelectionBody, username: str = Depends(require_staff)):
    ids, missing = _resolve(body)
    done = retention_service.delete(ids)
    return {"deleted": done, "not_found": missing}
