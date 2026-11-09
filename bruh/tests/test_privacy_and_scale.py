"""
Tests for the pre-final panel's three system requirements:
scalability of face matching, consent before registration, and the
data-retention policy. Nothing here touches the live database.

Run:  .venv/Scripts/python.exe -m pytest tests -v
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from core.config import CONSENT_VERSION  # noqa: E402
from routers import retention as retention_router  # noqa: E402
from routers import students  # noqa: E402
from services import face_service, queue_service, retention_service  # noqa: E402
from services.embedding_index import EmbeddingIndex  # noqa: E402


def unit(seed: int, dim: int = 512) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def near(v: np.ndarray, seed: int, noise: float) -> np.ndarray:
    n = np.random.default_rng(seed).normal(size=v.shape).astype(np.float32)
    w = v + noise * n / np.linalg.norm(n)
    return w / np.linalg.norm(w)


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------ scalability
@pytest.mark.parametrize("n", [1, 2, 50, 1000])
def test_index_decides_exactly_like_the_original_matcher(n):
    gallery = {sid: unit(sid) for sid in range(1, n + 1)}
    index = EmbeddingIndex(loader=lambda: [(s, v.tolist()) for s, v in gallery.items()])
    candidates = list(gallery.items())
    probes = [near(gallery[1], 900, 0.3), unit(9999)]          # an enrolled face, a stranger
    if n >= 2:
        probes.append(near((gallery[1] + gallery[2]) / 2, 901, 0.05))  # between two students
    for probe in probes:
        old = face_service.match_student(probe, candidates)
        new = index.match(probe)
        assert (new.student_id, new.accepted) == (old.student_id, old.accepted)
        assert new.score == pytest.approx(old.score, abs=1e-5)
        assert new.margin == pytest.approx(old.margin, abs=1e-5)


def test_index_reloads_only_when_told_to():
    calls, gallery = [], {1: unit(1)}

    def loader():
        calls.append(1)
        return [(s, v.tolist()) for s, v in gallery.items()]

    index = EmbeddingIndex(loader=loader, ttl_seconds=3600)
    probe = near(unit(2), 5, 0.2)
    index.match(probe)
    index.match(probe)
    assert len(calls) == 1, "the database is read once, not on every frame"

    gallery[2] = unit(2)                         # a new student enrols
    assert index.match(probe).student_id is None, "not visible until invalidated"
    index.invalidate()
    assert index.match(probe).student_id == 2
    assert len(calls) == 2


def test_index_can_exclude_the_student_being_checked():
    gallery = {1: unit(1), 2: unit(2)}
    index = EmbeddingIndex(loader=lambda: [(s, v.tolist()) for s, v in gallery.items()])
    probe = near(gallery[1], 7, 0.2)
    assert index.match(probe).student_id == 1
    assert index.match(probe, exclude_id=1).student_id is None


def test_ten_thousand_students_match_quickly():
    rng = np.random.default_rng(0)
    gallery = rng.normal(size=(10_000, 512)).astype(np.float32)
    index = EmbeddingIndex(loader=lambda: enumerate(gallery, start=1))
    target = gallery[4321] / np.linalg.norm(gallery[4321])
    index.match(target)                                      # build once
    import time
    start = time.perf_counter()
    for _ in range(20):
        result = index.match(near(target, 3, 0.2))
    per_match_ms = (time.perf_counter() - start) / 20 * 1000
    assert result.student_id == 4322
    assert per_match_ms < 50, f"{per_match_ms:.1f} ms per match at 10,000 students"


# ------------------------------------------------------------ consent
@pytest.fixture
def account_store(monkeypatch):
    rows: dict[int, dict] = {}
    created: list[dict] = []

    def create(**kw):
        sid = 100 + len(rows)
        rows[sid] = {"id": sid, "school_id": kw["school_id"], "full_name": kw.get("full_name"),
                     "face_enrolled_at": None, "consent_version": kw.get("consent_version")}
        created.append(kw)
        return sid

    def consent(sid, version):
        rows[sid]["consent_version"] = version
        return True

    monkeypatch.setattr(students, "verify_google_id_token",
                        lambda token: SimpleNamespace(sub=token, email=f"{token}@gbox.ncf.edu.ph", name=token))
    monkeypatch.setattr(students, "get_student_by_google_sub",
                        lambda sub: next((r for r in rows.values() if r["full_name"] == sub), None))
    monkeypatch.setattr(students, "get_student_by_school_id",
                        lambda sid: next((r for r in rows.values() if r["school_id"] == sid), None))
    monkeypatch.setattr(students, "get_student_by_id", lambda sid: rows.get(sid))
    monkeypatch.setattr(students, "create_student_account", create)
    monkeypatch.setattr(students, "record_student_consent", consent)
    monkeypatch.setattr(students, "touch_student_last_login", lambda sid: None)
    monkeypatch.setattr(students, "touch_student_activity", lambda sid: None)
    monkeypatch.setattr(students, "create_student_session", lambda sid: f"token-{sid}")
    return SimpleNamespace(rows=rows, created=created)


def sign_up(token, school_id=None, consent=None):
    body = students.GoogleAuthBody(id_token=token, school_id=school_id, consent_version=consent)
    return run(students.student_google_auth(body))


def test_no_account_is_created_without_consent(account_store):
    with pytest.raises(HTTPException) as err:
        sign_up("ana", school_id="2023-0001")
    assert err.value.detail == "consent_required"
    assert account_store.created == []


def test_account_is_created_with_consent_recorded(account_store):
    profile = sign_up("ana", school_id="2023-0001", consent=CONSENT_VERSION)
    assert account_store.created[0]["consent_version"] == CONSENT_VERSION
    assert profile["consent_required"] is False


def test_an_old_account_must_agree_before_queueing_or_enrolling(account_store, monkeypatch):
    account_store.rows[7] = {"id": 7, "school_id": "2020-0007", "full_name": "ben",
                             "face_enrolled_at": None, "consent_version": None}
    profile = sign_up("ben")
    assert profile["consent_required"] is True, "the app is told to show the notice"

    with pytest.raises(HTTPException) as err:
        run(students.join_queue(student_id=7))
    assert err.value.status_code == 403
    with pytest.raises(HTTPException) as err:
        run(students.register_my_face(photos=[], student_id=7))
    assert err.value.status_code == 403

    with pytest.raises(HTTPException):
        run(students.give_consent(students.ConsentBody(consent_version="old"), student_id=7))
    run(students.give_consent(students.ConsentBody(consent_version=CONSENT_VERSION), student_id=7))
    assert run(students.join_queue(student_id=7))["joined"] is True
    queue_service.queue_tracker.clear_join_intent(7)


# ------------------------------------------------------------ retention
@pytest.fixture
def retention_db(monkeypatch):
    state = SimpleNamespace(archived=[], deleted=[])
    db = retention_service.db
    monkeypatch.setattr(db, "students_due_for_archive",
                        lambda days, limit=1000: [{"id": 1, "school_id": "2019-1"}, {"id": 2, "school_id": "2019-2"}])
    monkeypatch.setattr(db, "students_due_for_deletion",
                        lambda days, limit=1000: [{"id": 3, "school_id": "2017-3"}])
    monkeypatch.setattr(db, "archive_student", lambda sid: state.archived.append(sid) or True)
    monkeypatch.setattr(db, "delete_student_profile", lambda sid: state.deleted.append(sid) or True)
    return state


def test_retention_preview_changes_nothing(retention_db):
    report = retention_service.run(dry_run=True, actor="test")
    assert (report["due_for_archive"], report["due_for_deletion"]) == (2, 1)
    assert retention_db.archived == [] and retention_db.deleted == []


def test_retention_archives_then_deletes_and_refreshes_matching(retention_db, monkeypatch):
    index = EmbeddingIndex(loader=lambda: [])
    index.match(unit(1))
    monkeypatch.setattr(queue_service, "embedding_index", index)
    report = retention_service.run(actor="test")
    assert retention_db.archived == [1, 2] and retention_db.deleted == [3]
    assert (report["archived"], report["deleted"]) == (2, 1)
    assert index._dirty, "archived faces must stop matching at once"


def test_staff_can_select_graduates_by_school_id(monkeypatch):
    known = {"2022-0001": {"id": 11}, "2022-0002": {"id": 12}}
    monkeypatch.setattr(retention_router, "get_student_by_school_id", lambda s: known.get(s))
    body = retention_router.SelectionBody(school_ids=["2022-0001", " 2022-0002 ", "9999-0000"])
    ids, missing = retention_router._resolve(body)
    assert ids == [11, 12] and missing == ["9999-0000"]
