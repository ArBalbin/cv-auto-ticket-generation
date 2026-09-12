"""
Face-recognition identity validation for registered students
(Algorithm 4 in ALGORITHMS.md). Replaces the retired HSV-histogram
appearance signature / comparison / twin-guard / done-blacklist chain.

Only a fixed-length embedding vector is ever extracted or stored — never a
face image — matching the system's existing "no raw video stored" privacy
stance, extended here to biometric data.

`compute_embedding()` runs in the detector process, where the raw camera
frame is available. `match_student()` runs in the backend, where enrolled
student embeddings live in the database. Neither function ever needs, nor
receives, a face crop or photo.
"""

import threading
from dataclasses import dataclass

import cv2
import numpy as np

from core.config import (
    FACE_MODEL_PACK,
    FACE_MATCH_THRESHOLD,
    FACE_MARGIN_THRESHOLD,
    FACE_MIN_DETECT_CONF,
)

_app = None
_app_lock = threading.Lock()


def _get_app():
    """Lazily load the InsightFace model pack on first use."""
    global _app
    if _app is not None:
        return _app
    with _app_lock:
        if _app is None:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name=FACE_MODEL_PACK, providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(320, 320))
            _app = app
    return _app


def compute_embedding(frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
    """
    Extract a 512-d ArcFace embedding for the face inside a person's bounding
    box, or None if no face is confidently detected. Called from detector.py.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 40 or y2 - y1 < 40:
        return None

    # Crop to the upper portion of the person box before running face
    # detection. The fraction is deliberately generous: 0.45 (the original
    # value) assumes a full-body standing person whose head occupies the top
    # ~20%, but someone stepping close to the camera to be recognized — the
    # intended use case — produces a head-and-shoulders box, and 0.45 then
    # slices through the middle of their face.
    #
    # Measured on a real failing frame (load_testing/bench_face_pipeline.py,
    # 480x360 capture, person bbox 318px tall), detector score by fraction:
    #   0.45 -> 0.572 (REJECTED, below FACE_MIN_DETECT_CONF=0.60)
    #   0.55 -> 0.663    0.65 -> 0.696    0.75 -> 0.722    1.00 -> 0.719
    # Latency was flat (~240-290ms) across all of them, because InsightFace
    # resizes the crop to det_size=(320,320) internally — a tighter crop
    # buys no speed, it only risks cutting the face in half.
    head_y2 = y1 + int((y2 - y1) * 0.75)
    crop = frame[y1:head_y2, x1:x2]
    if crop.size == 0:
        return None

    faces = _get_app().get(crop)
    if not faces:
        return None

    best = max(faces, key=lambda f: f.det_score)
    if best.det_score < FACE_MIN_DETECT_CONF:
        return None

    return best.normed_embedding.astype(np.float32)


def compute_embedding_from_photo(image_bytes: bytes) -> np.ndarray | None:
    """
    Extract a 512-d ArcFace embedding from a standalone enrollment photo
    (the whole image is expected to be framed on one face — unlike
    compute_embedding(), which crops a person bbox out of a wide queue-zone
    frame). Used only by the one-time student enrollment flow.
    """
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        return None

    faces = _get_app().get(frame)
    if not faces:
        return None

    best = max(faces, key=lambda f: f.det_score)
    if best.det_score < FACE_MIN_DETECT_CONF:
        return None

    return best.normed_embedding.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class MatchResult:
    student_id: int | None
    score: float
    margin: float
    accepted: bool


def match_student(
    embedding: np.ndarray,
    candidates: list[tuple[int, np.ndarray]],
    match_threshold: float = FACE_MATCH_THRESHOLD,
    margin_threshold: float = FACE_MARGIN_THRESHOLD,
) -> MatchResult:
    """
    Compare a live embedding against enrolled students eligible for matching
    today (the caller pre-filters out anyone already served/no-show — the
    done-blacklist equivalent). Never guesses: accepts only when the best
    match clears both an absolute threshold and a margin over the runner-up
    (the twin-guard equivalent), otherwise reports "unrecognized."

    Thresholds default to the calibrated config values but can be overridden
    per-call, mirroring how the rest of this codebase lets thresholds be
    tuned at runtime rather than only at import time.
    """
    if not candidates:
        return MatchResult(student_id=None, score=0.0, margin=0.0, accepted=False)

    scored = sorted(
        ((cosine_similarity(embedding, cand_emb), sid) for sid, cand_emb in candidates),
        key=lambda t: t[0],
        reverse=True,
    )
    best_score, best_sid = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -1.0
    margin = best_score - second_score

    accepted = best_score >= match_threshold and margin >= margin_threshold
    return MatchResult(
        student_id=best_sid if accepted else None,
        score=best_score,
        margin=margin,
        accepted=accepted,
    )


def build_enrollment_embedding(sample_embeddings: list[np.ndarray]) -> np.ndarray:
    """Average several enrollment-time samples into one canonical embedding."""
    stacked = np.stack(sample_embeddings).astype(np.float32)
    mean = stacked.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 1e-9 else mean


def embedding_to_json(embedding: np.ndarray) -> list:
    return [float(v) for v in embedding]


def embedding_from_json(data: list) -> np.ndarray:
    return np.asarray(data, dtype=np.float32)
