"""
Micro-benchmark: face_service.match_student() latency vs. enrollment size.

Phase 8 load/pressure testing (panel-required). match_student() does a
brute-force cosine-similarity scan over every eligible enrolled student —
an O(N) cost that matters once enrollment grows. This measures how that
latency scales at N=50/100/300, entirely in-process (no server, no DB).

Run: python load_testing/bench_face_match.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

import numpy as np
from services import face_service

ENROLLMENT_SIZES = [50, 100, 300]
TRIALS_PER_SIZE = 200


def random_embedding() -> np.ndarray:
    v = np.random.normal(0, 1, 512).astype(np.float32)
    return v / np.linalg.norm(v)


def main():
    print(f"{'N enrolled':>12} | {'p50 (ms)':>10} | {'p95 (ms)':>10} | {'p99 (ms)':>10}")
    print("-" * 52)

    for n in ENROLLMENT_SIZES:
        candidates = [(i, random_embedding()) for i in range(n)]
        live = random_embedding()

        durations = []
        for _ in range(TRIALS_PER_SIZE):
            start = time.perf_counter()
            face_service.match_student(live, candidates)
            durations.append((time.perf_counter() - start) * 1000)

        durations.sort()
        p50 = durations[len(durations) // 2]
        p95 = durations[int(len(durations) * 0.95)]
        p99 = durations[int(len(durations) * 0.99)]
        print(f"{n:>12} | {p50:>10.3f} | {p95:>10.3f} | {p99:>10.3f}")


if __name__ == "__main__":
    main()
