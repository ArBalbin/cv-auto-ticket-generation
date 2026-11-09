"""
Data-retention policy for student accounts.

Students graduate or stop using the system, and their records, above all their
face data, should not stay in the database indefinitely. The policy has two
stages:

  1. archive: an account with no sign-in or queue request for
     RETENTION_ARCHIVE_AFTER_DAYS is deactivated and its face data erased;
  2. delete: an account archived for RETENTION_DELETE_AFTER_DAYS more is
     deleted. Queue history survives with the student link set to NULL.

A student who signs in during the archive period is restored and enrols their
face again. Staff can apply either stage to anyone, such as a batch of
graduates, from the dashboard. A daily background pass applies the policy
automatically.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime

from core.config import (
    RETENTION_ARCHIVE_AFTER_DAYS,
    RETENTION_DELETE_AFTER_DAYS,
    RETENTION_ENABLED,
    RETENTION_INTERVAL_HOURS,
)
from database import database_handler as db
from services import queue_service

_last_run: dict | None = None
_scheduler_started = False
_run_lock = threading.Lock()


def policy() -> dict:
    return {
        "enabled": RETENTION_ENABLED,
        "archive_after_days": RETENTION_ARCHIVE_AFTER_DAYS,
        "delete_after_days": RETENTION_DELETE_AFTER_DAYS,
        "interval_hours": RETENTION_INTERVAL_HOURS,
    }


def _public(row: dict) -> dict:
    def iso(v):
        # Database times carry no zone; they are on the server's clock (UTC).
        return v.astimezone().isoformat() if isinstance(v, datetime) else v
    return {
        "student_id": row.get("id"),
        "school_id": row.get("school_id"),
        "full_name": row.get("full_name"),
        "last_activity": iso(row.get("last_activity")),
        "archived_at": iso(row.get("archived_at")),
    }


def overview(limit: int = 200) -> dict:
    return {
        "policy": policy(),
        "counts": db.retention_counts(RETENTION_ARCHIVE_AFTER_DAYS, RETENTION_DELETE_AFTER_DAYS),
        "due_for_archive": [_public(r) for r in db.students_due_for_archive(RETENTION_ARCHIVE_AFTER_DAYS, limit)],
        "due_for_deletion": [_public(r) for r in db.students_due_for_deletion(RETENTION_DELETE_AFTER_DAYS, limit)],
        "last_run": _last_run,
    }


def archive(student_ids: list[int]) -> list[int]:
    done = [sid for sid in student_ids if db.archive_student(sid)]
    if done:
        queue_service.embedding_index.invalidate()
    return done


def delete(student_ids: list[int]) -> list[int]:
    done = [sid for sid in student_ids if db.delete_student_profile(sid)]
    if done:
        queue_service.embedding_index.invalidate()
    return done


def run(dry_run: bool = False, actor: str = "scheduler") -> dict:
    """Apply both stages of the policy once and report what was done."""
    global _last_run
    with _run_lock:
        to_archive = db.students_due_for_archive(RETENTION_ARCHIVE_AFTER_DAYS)
        to_delete = db.students_due_for_deletion(RETENTION_DELETE_AFTER_DAYS)
        if dry_run:
            archived, deleted = [], []
        else:
            archived = archive([r["id"] for r in to_archive])
            deleted = delete([r["id"] for r in to_delete])
        report = {
            "ran_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "actor": actor,
            "dry_run": dry_run,
            "due_for_archive": len(to_archive),
            "due_for_deletion": len(to_delete),
            "archived": len(archived),
            "deleted": len(deleted),
        }
        if not dry_run:
            _last_run = report
        print(f"[Retention] {report}")
        return report


def _loop() -> None:
    # Give the database pool and the schema migration time to come up first.
    time.sleep(60)
    while True:
        try:
            run(actor="scheduler")
        except Exception as exc:
            print(f"[Retention] scheduled run failed: {exc}")
        time.sleep(max(1.0, RETENTION_INTERVAL_HOURS) * 3600)


def start_scheduler() -> None:
    global _scheduler_started
    if _scheduler_started or not RETENTION_ENABLED:
        return
    _scheduler_started = True
    threading.Thread(target=_loop, name="retention", daemon=True).start()
