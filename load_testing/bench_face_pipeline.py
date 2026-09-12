"""
Face-recognition pipeline benchmark: per-stage latency + detection rate.

Complements bench_face_match.py, which times only the cosine-similarity
scan (microseconds — the cheap part). The dominant cost in the live system
is embedding *extraction*: SCRFD face detection + landmark alignment +
ArcFace inference, run per person per frame inside detector.py. That is
what determines whether the detector keeps up with the camera, and how
long a student stands in front of it before being recognized.

This measures three things against the real calibration photo corpus:

  1. Extraction latency  — compute_embedding_from_photo(), p50/p95/p99.
  2. Detection rate      — the share of images yielding a usable embedding
                           at FACE_MIN_DETECT_CONF. A miss is not a wrong
                           answer, it is a *retry*: the live pipeline just
                           waits for the next frame, so a low rate shows up
                           to the student as a slower recognition, not an
                           error.
  3. Match latency       — match_student() at realistic enrollment sizes,
                           so the two stages can be compared honestly.

Run: python load_testing/bench_face_pipeline.py [--photos DIR]
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

import numpy as np  # noqa: E402
from services import face_service  # noqa: E402

DEFAULT_PHOTOS = Path(__file__).parent.parent / "calibration_data" / "faces"
MATCH_SIZES = [50, 100, 300]
MATCH_TRIALS = 200
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def percentiles(values: list[float]) -> tuple[float, float, float]:
    """p50/p95/p99 from an unsorted sample."""
    ordered = sorted(values)
    if not ordered:
        return 0.0, 0.0, 0.0
    last = len(ordered) - 1
    return (
        ordered[min(int(len(ordered) * 0.50), last)],
        ordered[min(int(len(ordered) * 0.95), last)],
        ordered[min(int(len(ordered) * 0.99), last)],
    )


def random_embedding() -> np.ndarray:
    v = np.random.normal(0, 1, 512).astype(np.float32)
    return v / np.linalg.norm(v)


def bench_extraction(photo_dir: Path) -> None:
    images = sorted(
        p for p in photo_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        print(f"No images found under {photo_dir} — skipping extraction benchmark.")
        return

    print(f"\n=== 1. Embedding extraction ({len(images)} images from {photo_dir.name}/) ===")

    # Warm up: the first call builds the ONNX sessions, which would otherwise
    # be charged to whichever image happened to be measured first.
    face_service.compute_embedding_from_photo(images[0].read_bytes())

    durations: list[float] = []
    detected = 0
    for image in images:
        data = image.read_bytes()
        start = time.perf_counter()
        embedding = face_service.compute_embedding_from_photo(data)
        durations.append((time.perf_counter() - start) * 1000)
        if embedding is not None:
            detected += 1

    p50, p95, p99 = percentiles(durations)
    rate = detected / len(images) * 100
    print(f"  detection rate : {detected}/{len(images)}  ({rate:.1f}%)")
    print(f"  latency p50    : {p50:8.1f} ms")
    print(f"  latency p95    : {p95:8.1f} ms")
    print(f"  latency p99    : {p99:8.1f} ms")
    print(f"  latency mean   : {statistics.mean(durations):8.1f} ms")
    print(f"  throughput     : {1000 / statistics.mean(durations):8.1f} extractions/sec")


def bench_matching() -> None:
    print("\n=== 2. Match scan (cosine over enrolled roster) ===")
    print(f"  {'N enrolled':>12} | {'p50 (ms)':>9} | {'p95 (ms)':>9} | {'p99 (ms)':>9}")
    print("  " + "-" * 48)

    for n in MATCH_SIZES:
        candidates = [(i, random_embedding()) for i in range(n)]
        live = random_embedding()

        durations = []
        for _ in range(MATCH_TRIALS):
            start = time.perf_counter()
            face_service.match_student(live, candidates)
            durations.append((time.perf_counter() - start) * 1000)

        p50, p95, p99 = percentiles(durations)
        print(f"  {n:>12} | {p50:>9.3f} | {p95:>9.3f} | {p99:>9.3f}")


def bench_live_path(snapshot: Path, pre_scaled: bool = False, trials: int = 30) -> None:
    """
    The number that actually matters operationally.

    Section 1 measures the *enrollment* path: a whole photo framed on one
    face. The live queue-zone path is different and much cheaper — it crops
    the upper 45% of a YOLO person box out of an already-downscaled frame,
    so the face reaches ArcFace at a fraction of the pixels. Reproducing it
    faithfully (same FRAME_SCALE, same YOLO box, same crop) is the only way
    to claim a real per-frame cost for detector.py.
    """
    import cv2

    if not snapshot.exists():
        print(f"\nNo live snapshot at {snapshot} — skipping live-path benchmark.")
        return

    frame = cv2.imread(str(snapshot))
    if frame is None:
        print(f"\nCould not read {snapshot} — skipping live-path benchmark.")
        return

    import os

    from dotenv import load_dotenv
    from ultralytics import YOLO

    # Mirrors detector.py (which defines these at module scope, so importing
    # it here would start the whole capture loop).
    load_dotenv(Path(__file__).parent.parent / ".env")
    frame_scale = float(os.getenv("FRAME_SCALE", "0.50").split("#")[0].strip())
    model_path = Path(__file__).parent.parent / "Model" / "yolov8n.pt"

    if pre_scaled:
        # Already a frame captured *inside* the detector loop (post-scale),
        # e.g. debug_snapshots/full_frame.png — rescaling it again would
        # understate the real face size.
        small = frame
    else:
        small = (
            cv2.resize(frame, None, fx=frame_scale, fy=frame_scale)
            if frame_scale != 1.0
            else frame
        )
    model = YOLO(str(model_path))
    result = model(small, classes=[0], verbose=False)[0]
    boxes = [tuple(int(v) for v in b) for b in result.boxes.xyxy.tolist()]

    print(f"\n=== 3. Live detector path ({snapshot.name}) ===")
    print(f"  source frame   : {frame.shape[1]}x{frame.shape[0]}")
    scale_note = "pre-scaled capture" if pre_scaled else f"FRAME_SCALE={frame_scale}"
    print(f"  after scale    : {small.shape[1]}x{small.shape[0]}  ({scale_note})")
    if not boxes:
        print("  no person detected in this snapshot — cannot time the live crop.")
        return
    print(f"  persons found  : {len(boxes)}")

    bbox = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    print(f"  person bbox    : {bbox}")

    face_service.compute_embedding(small, bbox)  # warm up

    durations, hits = [], 0
    for _ in range(trials):
        start = time.perf_counter()
        embedding = face_service.compute_embedding(small, bbox)
        durations.append((time.perf_counter() - start) * 1000)
        if embedding is not None:
            hits += 1

    p50, p95, p99 = percentiles(durations)
    print(f"  embedding found: {hits}/{trials}")
    print(f"  latency p50    : {p50:8.1f} ms")
    print(f"  latency p95    : {p95:8.1f} ms")
    print(f"  latency p99    : {p99:8.1f} ms")
    print(f"  latency mean   : {statistics.mean(durations):8.1f} ms")
    print(f"  sustainable    : {1000 / statistics.mean(durations):8.1f} extractions/sec")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", type=Path, default=DEFAULT_PHOTOS)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path(__file__).parent.parent / "calibration_data" / "live_snapshot.jpg",
    )
    parser.add_argument(
        "--pre-scaled",
        action="store_true",
        help="Snapshot was captured inside the detector loop (already downscaled).",
    )
    args = parser.parse_args()

    print("QueuEx face-recognition pipeline benchmark")
    print(f"model pack: {face_service.FACE_MODEL_PACK}  "
          f"min det conf: {face_service.FACE_MIN_DETECT_CONF}")

    bench_extraction(args.photos)
    bench_matching()
    bench_live_path(args.snapshot, pre_scaled=args.pre_scaled)

    print(
        "\nNote: extraction dominates; matching stays sub-millisecond even at\n"
        "300 enrolled. Recognition speed is bounded by how often a frame\n"
        "yields a usable face — not by roster size. Quote section 3 for the\n"
        "live per-frame cost and section 1 for the enrollment-time cost;\n"
        "they are different workloads and should not be conflated."
    )


if __name__ == "__main__":
    main()
