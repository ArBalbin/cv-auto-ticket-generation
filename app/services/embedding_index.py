"""
In-memory index of enrolled face embeddings for 1:N matching.

Matching used to reload every enrolled student's embedding from MySQL and
compare them one at a time in a Python loop, for each unidentified person on
every frame. That is fine for a handful of students and does not survive a
university: at 10,000 students each frame would read and parse roughly 50 MB
of JSON and run 10,000 Python-level dot products.

The index keeps the embeddings as one L2-normalised float32 matrix (10,000 x
512 is about 20 MB) and scores a live face against all of them with a single
matrix-vector product. It is rebuilt from the database only when something
changes (enrolment, re-enrolment, deletion, archiving) or after a TTL, which
keeps a second backend instance from serving stale data indefinitely.

The decision rule is unchanged: accept only if the best score reaches the
match threshold AND leads the runner-up by the margin.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Iterable

import numpy as np

from core.config import FACE_MARGIN_THRESHOLD, FACE_MATCH_THRESHOLD
from services.face_service import MatchResult

Loader = Callable[[], Iterable[tuple[int, list]]]


class EmbeddingIndex:
    def __init__(self, loader: Loader, ttl_seconds: float = 300.0):
        self._loader = loader
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._ids = np.empty(0, dtype=np.int64)
        self._matrix = np.empty((0, 0), dtype=np.float32)
        self._loaded_at = 0.0
        self._dirty = True

    # ------------------------------------------------------------ loading
    def invalidate(self) -> None:
        """Mark the index stale; the next match rebuilds it."""
        self._dirty = True

    def _rebuild(self) -> None:
        ids, rows = [], []
        for student_id, embedding in self._loader():
            vec = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if vec.ndim != 1 or norm <= 1e-9:
                continue
            ids.append(int(student_id))
            rows.append(vec / norm)
        self._ids = np.asarray(ids, dtype=np.int64)
        self._matrix = np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float32)
        self._loaded_at = time.monotonic()
        self._dirty = False

    def _snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            stale = self._dirty or (time.monotonic() - self._loaded_at) > self._ttl
            if stale:
                self._rebuild()
            return self._ids, self._matrix

    def size(self) -> int:
        return int(self._snapshot()[0].size)

    # ------------------------------------------------------------ matching
    def match(
        self,
        embedding,
        match_threshold: float = FACE_MATCH_THRESHOLD,
        margin_threshold: float = FACE_MARGIN_THRESHOLD,
        exclude_id: int | None = None,
    ) -> MatchResult:
        ids, matrix = self._snapshot()
        if exclude_id is not None and ids.size:
            keep = ids != exclude_id
            ids, matrix = ids[keep], matrix[keep]
        if ids.size == 0:
            return MatchResult(student_id=None, score=0.0, margin=0.0, accepted=False)

        live = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(live)
        if norm <= 1e-9:
            return MatchResult(student_id=None, score=0.0, margin=0.0, accepted=False)
        scores = matrix @ (live / norm)

        if scores.size == 1:
            best_i, second = 0, -1.0
        else:
            top2 = np.argpartition(scores, -2)[-2:]
            best_i, second_i = (top2[1], top2[0]) if scores[top2[1]] >= scores[top2[0]] else (top2[0], top2[1])
            second = float(scores[second_i])
        best = float(scores[best_i])
        margin = best - second
        accepted = best >= match_threshold and margin >= margin_threshold
        return MatchResult(
            student_id=int(ids[best_i]) if accepted else None,
            score=best,
            margin=margin,
            accepted=accepted,
        )
