"""
Person-detection accuracy evaluation for the YOLO layer.

WHY COUNTING, NOT mAP
---------------------
The obvious metric for a detector is mAP, but mAP is not what this system
depends on. QueuEx uses YOLO for exactly two things: how many people are in
the queue zone (which drives queue length, density and the wait estimate),
and a person box to crop a face from. Box-regression quality beyond "tight
enough to crop a head" does not affect either. Counting accuracy is therefore
the metric that describes the deployed system, and it has a second practical
advantage: ground truth is a single integer per frame that a human can label
in seconds, where mAP would need every person boxed by hand.

If the panel specifically asks for mAP, the honest answer is that this is
stock YOLOv8n with COCO-pretrained weights, unmodified — so its published
COCO person-class AP applies unchanged, and no claim is being made that this
project improved it. What this script measures is the thing the project can
legitimately claim: how the stock model performs on THIS camera, at THIS
angle and distance, with the deployment's own confidence and size filters
applied.

TWO STAGES
----------
  1. capture — samples frames, runs the exact deployed pipeline, writes an
     annotated image per frame plus a CSV with the predicted count and an
     empty true_count column.

  2. score — you fill in true_count by looking at the annotated images, then
     this computes the error metrics and the figure.

Run:
    python ML/evaluate_yolo_accuracy.py capture --source 0 --frames 120
    python ML/evaluate_yolo_accuracy.py capture --source clip.mp4 --frames 200
    #   ... open ML/yolo_eval/ground_truth.csv, fill true_count ...
    python ML/evaluate_yolo_accuracy.py score

The capture settings below MUST stay identical to app/detector.py. If a
detector default changes and this file is not updated, the numbers stop
describing the deployed system. They are re-read from the same environment
variables for that reason.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
OUT_DIR = Path(__file__).parent / "yolo_eval"
FRAMES_DIR = OUT_DIR / "frames"
CSV_PATH = OUT_DIR / "ground_truth.csv"
FIG_DIR = Path(__file__).parent / "figures"

MODEL_PATH = ROOT / "Model" / "yolov8n.pt"

# Mirrors app/detector.py exactly — same names, same defaults, same env vars.
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "480"))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.50"))
MIN_BBOX_AREA = int(os.getenv("MIN_BBOX_AREA", "1500"))
MAX_BBOX_FRAC = float(os.getenv("MAX_BBOX_FRAC", "0.70"))
FRAME_SCALE = float(os.getenv("FRAME_SCALE", "0.5"))
CAM_WIDTH = int(os.getenv("CAM_WIDTH", "1280"))
CAM_HEIGHT = int(os.getenv("CAM_HEIGHT", "720"))


def detect_people(model, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    """
    The deployed detection path, reproduced: downscale, track, keep class 0
    only, then drop boxes that are too small or implausibly large. The two
    area filters are part of the system's accuracy — a run without them would
    measure a detector this project does not deploy.
    """
    if FRAME_SCALE < 1.0:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (int(w * FRAME_SCALE), int(h * FRAME_SCALE)),
                           interpolation=cv2.INTER_LINEAR)
    else:
        small = frame

    results = model.track(small, imgsz=YOLO_IMGSZ, persist=True,
                          conf=YOLO_CONF, verbose=False)

    h_frame, w_frame = small.shape[:2]
    frame_area = max(1, h_frame * w_frame)

    kept = []
    dropped_small = []
    dropped_large = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            try:
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0]) if box.conf is not None else 0.5
            except Exception:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_BBOX_AREA:
                dropped_small.append((x1, y1, x2, y2, conf))
            elif area > MAX_BBOX_FRAC * frame_area:
                dropped_large.append((x1, y1, x2, y2, conf))
            else:
                kept.append((x1, y1, x2, y2, conf))

    # Filtered boxes are reported, not silently discarded. A person YOLO found
    # with high confidence but the size filter removed is invisible to the
    # system yet looks identical to a miss in the count — and the two have
    # completely different fixes. This was not hypothetical: on a saved test
    # snapshot YOLO scored a person at 0.93 and MAX_BBOX_FRAC=0.70 dropped the
    # box for covering 72% of the frame, which is what a student standing
    # close to the camera looks like.
    return kept, small, {"too_small": dropped_small, "too_large": dropped_large}


def cmd_capture(args) -> None:
    from ultralytics import YOLO

    if not MODEL_PATH.exists():
        sys.exit(f"Model not found: {MODEL_PATH}")

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cap.isOpened():
        sys.exit(f"Could not open source: {args.source}")

    print(f"Model   : {MODEL_PATH.name}")
    print(f"Settings: imgsz={YOLO_IMGSZ} conf={YOLO_CONF} "
          f"min_area={MIN_BBOX_AREA} max_frac={MAX_BBOX_FRAC} scale={FRAME_SCALE}")
    print(f"Sampling every {args.every} frame(s), target {args.frames} samples.\n")

    model = YOLO(str(MODEL_PATH))

    rows = []
    read_idx = 0
    saved = 0
    filtered_small = 0
    filtered_large = 0
    while saved < args.frames:
        ok, frame = cap.read()
        if not ok:
            print("  source ended early")
            break
        read_idx += 1
        if read_idx % args.every:
            continue

        boxes, small, dropped = detect_people(model, frame)
        n_small = len(dropped["too_small"])
        n_large = len(dropped["too_large"])
        filtered_small += n_small
        filtered_large += n_large

        annotated = small.copy()
        for i, (x1, y1, x2, y2, conf) in enumerate(boxes, start=1):
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(annotated, f"{i} {conf:.2f}", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        # Filtered boxes drawn in red so a reviewer labelling ground truth can
        # see a person the system threw away, rather than assuming YOLO missed.
        for x1, y1, x2, y2, conf in dropped["too_large"] + dropped["too_small"]:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(annotated, f"filtered {conf:.2f}", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(annotated, f"YOLO count: {len(boxes)}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        name = f"frame_{saved:04d}.jpg"
        cv2.imwrite(str(FRAMES_DIR / name), annotated)
        rows.append({"frame": name, "predicted_count": len(boxes), "true_count": "",
                     "filtered_too_large": n_large, "filtered_too_small": n_small})
        saved += 1
        flag = ""
        if n_large or n_small:
            flag = f"   [filtered: {n_large} too large, {n_small} too small]"
        print(f"  [{saved}/{args.frames}] {name}  predicted {len(boxes)}{flag}")

    cap.release()

    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "frame", "predicted_count", "true_count",
            "filtered_too_large", "filtered_too_small"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{saved} frames written to {FRAMES_DIR}")
    print(f"Ground-truth sheet: {CSV_PATH}")

    if filtered_large or filtered_small:
        print(f"\n  SIZE FILTER REMOVED {filtered_large + filtered_small} detection(s):")
        print(f"    {filtered_large} for exceeding MAX_BBOX_FRAC={MAX_BBOX_FRAC} "
              f"(person too close to the camera)")
        print(f"    {filtered_small} for falling under MIN_BBOX_AREA={MIN_BBOX_AREA} "
              f"(person too far away)")
        print( "  These are drawn in RED in the saved frames. If any of them is a")
        print( "  real person standing where students actually stand, the filter")
        print( "  is mis-tuned for this camera placement — that person cannot be")
        print( "  detected, recognised, or issued a ticket at all.")
    print("\nNEXT: open each image, count the people you can actually see, and")
    print("put that number in true_count. Count a person as present if a human")
    print("reviewer would say they are in the queue zone — including ones YOLO")
    print("missed entirely, otherwise recall cannot be measured. Then run:")
    print("    python ML/evaluate_yolo_accuracy.py score")


def cmd_score(args) -> None:
    if not CSV_PATH.exists():
        sys.exit(f"{CSV_PATH} not found — run the capture stage first.")

    rows = []
    with CSV_PATH.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row.get("true_count", "").strip():
                continue
            rows.append((row["frame"],
                         int(row["predicted_count"]),
                         int(row["true_count"])))

    if not rows:
        sys.exit("No labelled rows — fill in the true_count column first.")

    pred = np.array([r[1] for r in rows], dtype=float)
    true = np.array([r[2] for r in rows], dtype=float)
    err = pred - true

    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    bias = float(err.mean())
    exact = float((err == 0).mean())
    within1 = float((np.abs(err) <= 1).mean())

    # Counting recall/precision: across the whole set, how many real people
    # were found, and how many detections were spurious. Derived from counts
    # rather than box matching, so it assumes a detection in a frame with
    # fewer predictions than truth is a miss, not a misplaced box. Stated
    # explicitly because it is an approximation, not IoU-matched precision.
    total_true = float(true.sum())
    total_pred = float(pred.sum())
    missed = float(np.clip(true - pred, 0, None).sum())
    spurious = float(np.clip(pred - true, 0, None).sum())
    recall = (total_true - missed) / total_true if total_true else 0.0
    precision = (total_pred - spurious) / total_pred if total_pred else 0.0

    print("\n" + "=" * 60)
    print("QueuEx person-detection accuracy (YOLOv8n, stock COCO weights)")
    print("=" * 60)
    print(f"\nSettings mirrored from detector: imgsz={YOLO_IMGSZ} conf={YOLO_CONF} "
          f"min_area={MIN_BBOX_AREA}")
    print(f"\nFrames labelled       : {len(rows)}")
    print(f"People in ground truth: {int(total_true)}")
    print(f"People detected       : {int(total_pred)}")

    print(f"\nCOUNTING ERROR")
    print(f"  MAE                 : {mae:.3f} people per frame")
    print(f"  RMSE                : {rmse:.3f}")
    print(f"  Mean bias           : {bias:+.3f}  "
          f"({'over' if bias > 0 else 'under'}-counting on average)")
    print(f"  Exact count         : {exact * 100:.1f}% of frames")
    print(f"  Within +/-1         : {within1 * 100:.1f}% of frames")

    print(f"\nDETECTION (count-derived, not IoU-matched)")
    print(f"  Recall              : {recall * 100:.1f}%  "
          f"({int(missed)} people missed)")
    print(f"  Precision           : {precision * 100:.1f}%  "
          f"({int(spurious)} spurious detections)")

    # Does accuracy degrade as the queue gets busier? This is the question a
    # panel will ask about a crowd system, and it is invisible in the average.
    print(f"\nERROR BY CROWD SIZE")
    print(f"  {'people':>8} {'frames':>7} {'MAE':>7} {'bias':>7}")
    for n in sorted(set(true.astype(int))):
        mask = true == n
        print(f"  {n:>8} {int(mask.sum()):>7} "
              f"{np.abs(err[mask]).mean():>7.2f} {err[mask].mean():>+7.2f}")

    _figure(true, pred, err, mae, bias)
    print(f"\nFigure: {FIG_DIR / '09_yolo_counting_accuracy.png'}")
    print()


def _figure(true, pred, err, mae, bias) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 10, "axes.grid": True,
        "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    hi = max(true.max(), pred.max()) + 1
    jitter = (np.random.default_rng(0).random(len(true)) - 0.5) * 0.18
    ax1.plot([0, hi], [0, hi], "--", color="#adb5bd", linewidth=1.5,
             label="Perfect count")
    ax1.scatter(true + jitter, pred + jitter, s=30, alpha=0.6, color="#457b9d",
                edgecolor="none", label="Frame")
    ax1.set_xlabel("People actually present")
    ax1.set_ylabel("People detected")
    ax1.set_title(f"Counting accuracy (MAE {mae:.2f})")
    ax1.legend(frameon=False, fontsize=8)

    bins = np.arange(err.min() - 0.5, err.max() + 1.5, 1)
    ax2.hist(err, bins=bins, color="#2a9d8f", edgecolor="white")
    ax2.axvline(0, color="#1d3557", linestyle="--", linewidth=2)
    ax2.set_xlabel("Detected − actual")
    ax2.set_ylabel("Frames")
    ax2.set_title(f"Count error (bias {bias:+.2f})")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_yolo_counting_accuracy.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="Sample frames and run the deployed pipeline.")
    cap.add_argument("--source", default="0",
                     help="Webcam index (0) or a video file path.")
    cap.add_argument("--frames", type=int, default=100,
                     help="How many sampled frames to save.")
    cap.add_argument("--every", type=int, default=15,
                     help="Sample one frame out of every N read (default 15).")
    cap.set_defaults(func=cmd_capture)

    sco = sub.add_parser("score", help="Score the filled-in ground-truth sheet.")
    sco.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
