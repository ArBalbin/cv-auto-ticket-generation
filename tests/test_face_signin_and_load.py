"""
Tests for the fixes made after the user acceptance test of 16 Nov 2026:

- a face can sign in only to the account last used on the phone, so a
  student whose own face is not registered can no longer land in a
  look-alike classmate's account (a student saw another student's name);
- a student with no registered face is told so, without any photo work;
- turning photos into face signatures no longer freezes the server for
  everyone else while it runs (registrations hung and phones reported
  "Cannot connect" when many students registered at once).

Nothing here touches the live database.

Run:  .venv/Scripts/python.exe -m pytest tests -v
"""
import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from core.config import CONSENT_VERSION  # noqa: E402
from routers import students  # noqa: E402
from services import face_service, queue_service  # noqa: E402
from services.embedding_index import EmbeddingIndex  # noqa: E402


def unit(seed: int, dim: int = 512) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def near(v: np.ndarray, seed: int, noise: float) -> np.ndarray:
    n = np.random.default_rng(seed).normal(size=v.shape).astype(np.float32)
    w = v + noise * n / np.linalg.norm(n)
    return w / np.linalg.norm(w)


class Photo:
    """Stands in for an uploaded file; its bytes name whose face it shows."""

    def __init__(self, face: str):
        self.face = face

    async def read(self) -> bytes:
        return self.face.encode()


def run(coro):
    return asyncio.run(coro)


ANA, BEN = unit(1), unit(2)
FACES = {"ana": near(ANA, 11, 0.3), "ben": near(BEN, 12, 0.3), "cara": unit(3), "blank": None}


@pytest.fixture
def system(monkeypatch):
    rows = {
        1: {"id": 1, "school_id": "2023-0001", "full_name": "Ana", "face_enrolled_at": "2026-11-16",
            "consent_version": CONSENT_VERSION},
        2: {"id": 2, "school_id": "2023-0002", "full_name": "Ben", "face_enrolled_at": "2026-11-16",
            "consent_version": CONSENT_VERSION},
        3: {"id": 3, "school_id": "2023-0003", "full_name": "Cara", "face_enrolled_at": None,
            "consent_version": CONSENT_VERSION},
    }
    state = SimpleNamespace(rows=rows, events=[], embeds=[], saved=[])

    def embed(data: bytes):
        state.embeds.append(data)
        return FACES[data.decode()]

    monkeypatch.setattr(face_service, "compute_embedding_from_photo", embed)
    monkeypatch.setattr(queue_service, "embedding_index",
                        EmbeddingIndex(loader=lambda: [(1, ANA.tolist()), (2, BEN.tolist())]))
    monkeypatch.setattr(students, "get_student_by_id", lambda sid: rows.get(sid))
    monkeypatch.setattr(students, "touch_student_last_login", lambda sid: None)
    monkeypatch.setattr(students, "create_student_session", lambda sid: f"token-{sid}")
    monkeypatch.setattr(students, "record_face_match_event",
                        lambda sid, track, score, margin, accepted: state.events.append((sid, accepted)))
    monkeypatch.setattr(students, "save_student_face_embedding",
                        lambda **kw: state.saved.append(kw) or True)
    # A fresh limiter per test: each test runs its own event loop.
    monkeypatch.setattr(students, "_face_slots", asyncio.Semaphore(1))
    return state


def face_sign_in(face: str, phone_account: int | None):
    return run(students.student_face_auth(photo=Photo(face), expected_student_id=phone_account))


# ------------------------------------------------------------ face sign-in
def test_the_account_owner_signs_in(system):
    profile = face_sign_in("ana", phone_account=1)
    assert profile["session_token"] == "token-1"
    assert profile["full_name"] == "Ana"
    assert system.events == [(1, True)]


def test_a_face_cannot_open_another_students_account(system):
    # Ana's face on a phone last used by Ben: the face is recognised, but it
    # is not the account on this phone, so nobody is signed in.
    with pytest.raises(HTTPException) as err:
        face_sign_in("ana", phone_account=2)
    assert err.value.status_code == 401
    # Ana is told the phone holds another account, not that her face failed.
    assert err.value.detail == students.FACE_SIGNIN_OTHER_ACCOUNT
    assert system.events == [(None, False)]


def test_an_unrecognised_face_is_told_to_try_again(system, monkeypatch):
    monkeypatch.setitem(FACES, "stranger", unit(99))
    with pytest.raises(HTTPException) as err:
        face_sign_in("stranger", phone_account=1)
    assert err.value.detail == students.FACE_SIGNIN_NO_MATCH


def test_without_an_account_on_the_phone_face_sign_in_is_not_attempted(system):
    with pytest.raises(HTTPException) as err:
        face_sign_in("ana", phone_account=None)
    assert err.value.detail == students.FACE_SIGNIN_NEEDS_ACCOUNT
    assert system.embeds == [], "no photo work for a request that cannot succeed"


def test_a_student_without_a_registered_face_is_told_so(system):
    # The UAT case: Cara never registered her face. She is told to use
    # Gbox at once, instead of waiting and seeing "Cannot connect", and her
    # face can no longer be matched to Ana's or Ben's account.
    with pytest.raises(HTTPException) as err:
        face_sign_in("ben", phone_account=3)
    assert err.value.status_code == 401
    assert err.value.detail == students.FACE_SIGNIN_NOT_REGISTERED
    assert system.embeds == []
    assert system.events == []


def test_no_clear_face_gets_a_plain_message(system):
    with pytest.raises(HTTPException) as err:
        face_sign_in("blank", phone_account=1)
    assert err.value.status_code == 422
    assert err.value.detail == students.FACE_NOT_SEEN


# ------------------------------------------------------------ server stays responsive
def test_face_work_does_not_freeze_other_requests(system, monkeypatch):
    window = {}

    def slow_embed(data: bytes):
        window["start"] = time.perf_counter()
        time.sleep(0.6)                       # a photo on the small cloud CPU
        window["end"] = time.perf_counter()
        return FACES[data.decode()]

    monkeypatch.setattr(face_service, "compute_embedding_from_photo", slow_embed)
    ticks = []

    async def other_requests():
        # Stands in for every other student's request on the same server.
        for _ in range(20):
            ticks.append(time.perf_counter())
            await asyncio.sleep(0.05)

    async def scenario():
        await asyncio.gather(
            students.student_face_auth(photo=Photo("ana"), expected_student_id=1),
            other_requests(),
        )

    run(scenario())
    served_meanwhile = [t for t in ticks if window["start"] < t < window["end"]]
    assert len(served_meanwhile) >= 5, "other requests are answered while a face is processed"


def test_registration_still_saves_the_face(system):
    result = run(students.register_my_face(photos=[Photo("cara"), Photo("blank")], student_id=3))
    assert result == {"student_id": 3, "samples_used": 1, "samples_submitted": 2}
    assert system.saved[0]["student_id"] == 3


def test_registration_still_refuses_a_face_already_on_another_account(system):
    with pytest.raises(HTTPException) as err:
        run(students.register_my_face(photos=[Photo("ben")], student_id=3))
    assert err.value.status_code == 409


def test_registration_with_no_clear_face_gets_a_plain_message(system):
    with pytest.raises(HTTPException) as err:
        run(students.register_my_face(photos=[Photo("blank")], student_id=3))
    assert err.value.status_code == 422
    assert err.value.detail == students.FACE_NOT_SEEN
