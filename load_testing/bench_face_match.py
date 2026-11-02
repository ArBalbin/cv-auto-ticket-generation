"""
Scalability benchmark: identifying one face against N enrolled students.

Compares the two ways the backend has matched a face:

  before  every match reloaded all N embeddings from MySQL as JSON and scored
          them one at a time in a Python loop (face_service.match_student);
  after   the embeddings stay in memory as one normalised matrix and are all
          scored with a single matrix-vector product (EmbeddingIndex).

The database round trip itself is not timed (it depends on the network), only
the JSON parsing it forced on every frame, so the "before" figures are a lower
bound. Everything runs in-process on synthetic embeddings; no server, no DB.

Run:  .venv/Scripts/python.exe load_testing/bench_face_match.py
Writes load_testing/results/face_match_scaling.csv
"""
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "app"))

import numpy as np  # noqa: E402
from services import face_service  # noqa: E402
from services.embedding_index import EmbeddingIndex  # noqa: E402

ENROLLMENT_SIZES = [100, 1_000, 5_000, 10_000]
TRIALS = 30


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))]


def timed(fn, trials):
    out = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        out.append((time.perf_counter() - start) * 1000)
    return out


def main():
    rng = np.random.default_rng(42)
    rows = []
    header = f"{'N':>7} | {'before p50 ms':>13} | {'after p50 ms':>12} | {'speed-up':>8} | {'index MB':>8}"
    print(header)
    print("-" * len(header))
    for n in ENROLLMENT_SIZES:
        gallery = rng.normal(size=(n, 512)).astype(np.float32)
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
        as_json = [json.dumps([float(x) for x in row]) for row in gallery]   # how MySQL returns them
        probe = gallery[n // 2] + rng.normal(scale=0.03, size=512).astype(np.float32)

        def before():
            candidates = [(i, np.asarray(json.loads(s), dtype=np.float32)) for i, s in enumerate(as_json)]
            return face_service.match_student(probe, candidates)

        index = EmbeddingIndex(loader=lambda: enumerate(gallery.tolist()))
        index.match(probe)                                   # built once, as on first use

        def after():
            return index.match(probe)

        assert before().student_id == after().student_id == n // 2
        trials_before = max(5, TRIALS // (n // 1000 + 1))
        b = timed(before, trials_before)
        a = timed(after, TRIALS * 10)
        index_mb = index._matrix.nbytes / 1e6
        row = {
            "enrolled": n,
            "before_p50_ms": round(percentile(b, 0.5), 2),
            "before_p95_ms": round(percentile(b, 0.95), 2),
            "after_p50_ms": round(percentile(a, 0.5), 3),
            "after_p95_ms": round(percentile(a, 0.95), 3),
            "speedup_p50": round(percentile(b, 0.5) / percentile(a, 0.5), 1),
            "index_memory_mb": round(index_mb, 1),
        }
        rows.append(row)
        print(f"{n:>7} | {row['before_p50_ms']:>13} | {row['after_p50_ms']:>12} | "
              f"{row['speedup_p50']:>7}x | {row['index_memory_mb']:>8}")

    out = ROOT / "results" / "face_match_scaling.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
