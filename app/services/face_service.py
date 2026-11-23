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

            # allowed_modules is not an optimization, it is what keeps the
            # deployed server inside its memory limit. Without it FaceAnalysis
            # loads every .onnx in the pack, and buffalo_s ships three this
            # codebase never reads: 1k3d68.onnx (137MB), 2d106det.onnx and
            # genderage.onnx. Only normed_embedding (recognition) and the
            # bbox/det_score (detection) are ever used.
            #
            # Measured resident memory after prepare(), CPU provider:
            #   all five modules       251.9 MB   load 2.01s
            #   detection+recognition   98.9 MB   load 1.22s
            #
            # 153MB on a 512MB host is the difference between answering a face
            # request and being OOM-killed mid-upload — which the proxy turns
            # into a bodyless 502, so the phone reports a bare "Request
            # failed." with nothing in it to diagnose.
            #
            # Embeddings are unaffected: recognition aligns on the detector's
            # own keypoints, not on the landmark models. Verified on three
            # enrolment photos, cosine 1.000000 and max |diff| 0.0 against the
            # unrestricted pack, so existing enrolments stay valid.
            app = FaceAnalysis(
                name=FACE_MODEL_PACK,
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=-1, det_size=(320, 320))
            _app = app
    return _app


def _off_centre(face_cx: float, box: tuple) -> float:
    """How far a face sits from the middle of a person box, as a fraction of
    the box's width: 0 is dead centre, 0.5 is on its edge."""
    width = max(1.0, float(box[2] - box[0]))
    return abs(face_cx - (box[0] + box[2]) / 2.0) / width


def _inside(x: float, y: float, box: tuple) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def choose_own_face(face_boxes: list, bbox: tuple, other_boxes=()) -> int | None:
    """Which of the faces found in a person's box is that person's own.

    `face_boxes` are (x1, y1, x2, y2) in frame coordinates. A face belongs to
    the person in whose box it sits most centrally: your own head is near the
    middle of your box, while a neighbour's face pokes into it near an edge.
    Faces that sit more centrally in someone else's box are theirs and are
    left out; of the rest, the most central one is chosen. Returns its index,
    or None when every face found belongs to somebody else.

    Picking the most confident face instead gave a person the face of whoever
    stood beside them and faced the camera better. On 23 Nov 2026 one track
    matched a student who already had a number 38 times in 85 seconds, so the
    person on that track, who had asked for a number, never got one.
    """
    best, best_off = None, None
    for i, (fx1, fy1, fx2, fy2) in enumerate(face_boxes):
        cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
        mine = _off_centre(cx, bbox)
        if any(_inside(cx, cy, other) and _off_centre(cx, other) < mine for other in other_boxes):
            continue
        if best_off is None or mine < best_off:
            best, best_off = i, mine
    return best


def compute_embedding(frame: np.ndarray, bbox: tuple, other_boxes=()) -> np.ndarray | None:
    """
    Extract a 512-d ArcFace embedding for the face inside a person's bounding
    box, or None if no face of theirs is confidently detected. `other_boxes`
    are the other people in the same frame, so a neighbour's face that falls
    inside this box is not taken for this person's (choose_own_face). Called
    from detector.py.
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
    confident = [f for f in faces if f.det_score >= FACE_MIN_DETECT_CONF]
    if not confident:
        return None

    # Face boxes come back in crop coordinates; the person boxes are in the
    # frame's.
    in_frame = [
        (f.bbox[0] + x1, f.bbox[1] + y1, f.bbox[2] + x1, f.bbox[3] + y1)
        for f in confident
    ]
    own = choose_own_face(in_frame, (x1, y1, x2, y2), other_boxes)
    if own is None:
        return None
    return confident[own].normed_embedding.astype(np.float32)


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
