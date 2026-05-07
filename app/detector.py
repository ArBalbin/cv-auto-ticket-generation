#!/usr/bin/env python3

import sys
import time
import signal
import threading
import queue as _queue
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

app_root = Path(__file__).parent
project_root = app_root.parent
for import_path in (str(app_root), str(project_root)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from dotenv import load_dotenv
load_dotenv()

import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

print("[Detector] Loading OpenCV...", flush=True)
import cv2
print("[Detector] Loading HTTP client...", flush=True)
import requests

print("[Detector] Loading Ultralytics/YOLO...", flush=True)
from ultralytics import YOLO
from services.prediction_service import PredictionService
print("[Detector] Detector libraries loaded.", flush=True)

# CONFIGURATION
def _env(key: str, default: str) -> str:
    """Read env var and strip inline shell comments (# ...) from the value."""
    val = os.getenv(key, default)
    return val.split("#")[0].strip()

API_BASE_URL  = _env("API_BASE_URL",  "http://localhost:5000")
CAM_TOKEN     = os.getenv("CAM_TOKEN",     "detector-secret-token")
MODEL_PATH    = str(project_root / "Model" / "yolov8n.pt")

CAMERA_INDEX_RAW = os.getenv("CAMERA_INDEX", "auto").split("#")[0].strip()
CAMERA_INDEX = None if CAMERA_INDEX_RAW.lower() in {"", "auto", "-1"} else int(CAMERA_INDEX_RAW)
CAMERA_SCAN_LIMIT = int(os.getenv("CAMERA_SCAN_LIMIT", "5"))
CAMERA_FALLBACK_SCAN = os.getenv("CAMERA_FALLBACK_SCAN", "1").strip() != "0"
CAMERA_FPS = float(os.getenv("CAMERA_FPS", "30"))
CAMERA_FOURCC = os.getenv("CAMERA_FOURCC", "MJPG").split("#")[0].strip().upper()
CAMERA_READ_FAIL_LIMIT = int(os.getenv("CAMERA_READ_FAIL_LIMIT", "12"))
CAMERA_READ_RETRY_SLEEP = float(os.getenv("CAMERA_READ_RETRY_SLEEP", "0.03"))
CAMERA_BACKENDS = [
    name.strip().upper()
    for name in os.getenv("CAMERA_BACKENDS", "DSHOW,MSMF,ANY").split(",")
    if name.strip()
]

YOLO_EVERY    = int(os.getenv("YOLO_EVERY",    "5"))   # YOLO every Nth frame
PUSH_EVERY    = int(os.getenv("PUSH_EVERY",    "5"))   # push every Nth YOLO frame
API_PUSH_FPS  = float(os.getenv("API_PUSH_FPS", "5.0")) # max queue-state pushes/sec
JPEG_QUALITY  = int(os.getenv("JPEG_QUALITY",  "70"))  # lower = faster encode
SNAPSHOT_FPS  = float(os.getenv("SNAPSHOT_FPS", "5.0")) # dashboard video upload fps
SNAPSHOT_UPLOAD_ENABLED = os.getenv("SNAPSHOT_UPLOAD_ENABLED", "1").strip() != "0"

# Downscale factor applied to frame before YOLO + overlay (1.0 = no scaling)
FRAME_SCALE   = float(os.getenv("FRAME_SCALE", "0.50"))

# Dashboard/video snapshot scale. Keep YOLO small but stream a clearer image.
SNAPSHOT_SCALE = float(os.getenv("SNAPSHOT_SCALE", "1.0"))

# YOLO inference image size - smaller = faster, less accurate
YOLO_IMGSZ    = int(os.getenv("YOLO_IMGSZ", "320"))

# Camera capture resolution
CAM_WIDTH     = int(os.getenv("CAM_WIDTH",  "1280"))
CAM_HEIGHT    = int(os.getenv("CAM_HEIGHT", "720"))

CAM_RETRY_SLEEP = float(os.getenv("CAM_RETRY_SLEEP", "2.0"))
CAM_MAX_RETRIES = int(os.getenv("CAM_MAX_RETRIES",   "30"))

QUEUE_CONFIG = {
    "avg_service_time": float(os.getenv("AVG_SERVICE_TIME", "3.0")),
    "num_counters":     int(os.getenv("NUM_COUNTERS",       "3")),
}

NMS_IOU_THRESH = float(os.getenv("NMS_IOU_THRESH", "0.40"))
NMS_DIST_FRAC  = float(os.getenv("NMS_DIST_FRAC",  "0.25"))
PUSH_TIMEOUT   = float(os.getenv("PUSH_TIMEOUT",   "5.0"))
SNAPSHOT_TIMEOUT = float(os.getenv("SNAPSHOT_TIMEOUT", str(min(PUSH_TIMEOUT, 1.0))))
SNAPSHOT_FAILURE_DISABLE_AFTER = int(os.getenv("SNAPSHOT_FAILURE_DISABLE_AFTER", "3"))
SNAPSHOT_FAILURE_BACKOFF_SECONDS = float(os.getenv("SNAPSHOT_FAILURE_BACKOFF_SECONDS", "15"))
YOLO_CONF      = float(os.getenv("YOLO_CONF",      "0.50"))
MIN_BBOX_AREA  = int(os.getenv("MIN_BBOX_AREA",    "2000"))
MAX_BBOX_FRAC  = float(os.getenv("MAX_BBOX_FRAC",  "0.70"))
DETECTION_DEBUG_EVERY = int(os.getenv("DETECTION_DEBUG_EVERY", "30"))


CAMERA_BACKEND_CODES = {
    "DSHOW": cv2.CAP_DSHOW,
    "MSMF": cv2.CAP_MSMF,
    "ANY": cv2.CAP_ANY,
}



# SHUTDOWN FLAG
_shutdown = threading.Event()


def _handle_signal(sig, frame):
    print(f"\n[Detector] Signal {sig} received - shutting down")
    _shutdown.set()


signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)



# SHARED STATE
_state_lock   = threading.Lock()
_shared_state = {
    # crowd stats
    "count":                0,
    "avg_density":          0.0,
    "max_density":          0.0,
    "timestamp":            time.time(),
    "snapshot_jpg":         None,
    "snapshot_seq":         0,
    "tracked_persons":      [],
    # prediction fields from PredictionService - forwarded to api.py
    "queue_length":         0,
    "active_counters":      QUEUE_CONFIG["num_counters"],
    "estimated_wait_time":  0.0,
    "arrival_rate":         0.0,
    "system_utilization":   0.0,
    "predicted_wait_5min":  0.0,
    "predicted_wait_15min": 0.0,
    "predicted_wait_30min": 0.0,
    # queue_state returned by api.py after processing - used for overlay drawing
    "api_queue_state":      {},
    # done_pending returned by api.py - drawn as red "DONE - EXIT PLEASE" boxes
    "done_pending":         [],
}

# PredictionService writes into current_data; values are copied to
# _shared_state at push time. Kept separate so the predictor can update
# freely without holding _state_lock.
_data_lock   = threading.Lock()
current_data = {
    "count":                0,
    "avg_density":          0.0,
    "max_density":          0.0,
    "queue_length":         0,
    "active_counters":      QUEUE_CONFIG["num_counters"],
    "estimated_wait_time":  0.0,
    "predicted_wait_5min":  0.0,
    "predicted_wait_15min": 0.0,
    "predicted_wait_30min": 0.0,
    "system_utilization":   0.0,
    "arrival_rate":         0.0,
    "service_rate":         1.0 / QUEUE_CONFIG["avg_service_time"],
    "timestamp":            time.time(),
}



# NMS HELPER
def _nms_is_duplicate(bbox: tuple, kept: list) -> bool:
    cx   = (bbox[0] + bbox[2]) / 2
    cy   = (bbox[1] + bbox[3]) / 2
    diag = ((bbox[2]-bbox[0])**2 + (bbox[3]-bbox[1])**2) ** 0.5
    for kb in kept:
        kcx  = (kb[0]+kb[2]) / 2
        kcy  = (kb[1]+kb[3]) / 2
        kd   = ((kb[2]-kb[0])**2 + (kb[3]-kb[1])**2) ** 0.5
        dist = ((cx-kcx)**2 + (cy-kcy)**2) ** 0.5
        ix1, iy1 = max(bbox[0], kb[0]), max(bbox[1], kb[1])
        ix2, iy2 = min(bbox[2], kb[2]), min(bbox[3], kb[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        union = ((bbox[2]-bbox[0])*(bbox[3]-bbox[1]) +
                 (kb[2]-kb[0])*(kb[3]-kb[1]) - inter)
        iou = inter / union if union > 0 else 0.0
        if iou > NMS_IOU_THRESH or dist < NMS_DIST_FRAC*(diag+kd)/2:
            return True
    return False


def _run_nms(raw: list) -> tuple[list, list]:
    kept, persons, tracked_persons = [], [], []
    for det in sorted(raw, key=lambda d: d["conf"], reverse=True):
        bbox = det["bbox"]
        if not _nms_is_duplicate(bbox, kept):
            kept.append(bbox)
            persons.append(bbox)
            tracked_persons.append({"track_id": det["track_id"], "bbox": bbox,
                                    "conf": det["conf"]})
    return persons, tracked_persons



# APPEARANCE EXTRACTION
def _extract_appearance(frame, bbox) -> list | None:
    if frame is None:
        return None
    x1, y1, x2, y2 = bbox
    h_f, w_f = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_f-1, x2), min(h_f-1, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return None
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().tolist()



# HTTP PUSH HELPER
_session     = requests.Session()
_push_errors = 0


def _apply_backend_config(config: dict) -> None:
    if not isinstance(config, dict):
        return

    changed = False

    counters_raw = config.get("active_counters", config.get("num_counters"))
    if counters_raw is not None:
        try:
            counters = max(1, int(counters_raw))
            changed = changed or counters != QUEUE_CONFIG["num_counters"]
            QUEUE_CONFIG["num_counters"] = counters
        except (TypeError, ValueError):
            pass

    avg_raw = config.get("avg_service_time", config.get("avg_service_time_min"))
    if avg_raw is not None:
        try:
            avg_service_time = max(0.1, float(avg_raw))
            changed = changed or avg_service_time != QUEUE_CONFIG["avg_service_time"]
            QUEUE_CONFIG["avg_service_time"] = avg_service_time
        except (TypeError, ValueError):
            pass

    service_rate = 1.0 / max(0.1, QUEUE_CONFIG["avg_service_time"])
    with _data_lock:
        current_data["active_counters"] = QUEUE_CONFIG["num_counters"]
        current_data["service_rate"] = service_rate

    with _state_lock:
        _shared_state["active_counters"] = QUEUE_CONFIG["num_counters"]

    if changed:
        print(
            "[Detector] Synced queue config from API: "
            f"{QUEUE_CONFIG['num_counters']} counter(s), "
            f"{QUEUE_CONFIG['avg_service_time']:.1f} min service time"
        )


def _push_to_api(payload: dict) -> None:
    global _push_errors
    headers = {"X-CAM-TOKEN": CAM_TOKEN}

    try:
        resp = _session.post(
            f"{API_BASE_URL}/yolo/push-frame",
            json    = payload,
            headers = headers,
            timeout = PUSH_TIMEOUT,
        )
        data         = resp.json()
        queue_state  = data.get("queue_state",  {})
        done_pending = data.get("done_pending", [])
        _apply_backend_config(data.get("config", {}))
        _push_errors = 0

        with _state_lock:
            _shared_state["api_queue_state"] = queue_state
            _shared_state["done_pending"]    = done_pending

    except requests.RequestException as e:
        _push_errors += 1
        if _push_errors <= 5 or _push_errors % 30 == 0:
                    print(f"[Detector] WARNING push-frame failed ({_push_errors}x): {e}")


def _snapshot_upload_worker() -> None:
    """Upload the latest JPEG at a fixed rate, independent of YOLO cadence."""
    interval = 1.0 / max(0.1, SNAPSHOT_FPS)
    headers  = {"X-CAM-TOKEN": CAM_TOKEN, "Content-Type": "image/jpeg"}
    last_seq = -1
    errors   = 0
    backoff_until = 0.0

    session = requests.Session()
    try:
        while not _shutdown.is_set():
            started = time.time()
            now = started
            if now < backoff_until:
                _shutdown.wait(min(interval, backoff_until - now))
                continue

            with _state_lock:
                snap_jpg = _shared_state.get("snapshot_jpg")
                seq      = _shared_state.get("snapshot_seq", 0)

            if snap_jpg is not None and seq != last_seq:
                try:
                    session.post(
                        f"{API_BASE_URL}/yolo/update",
                        data    = snap_jpg,
                        headers = headers,
                        timeout = SNAPSHOT_TIMEOUT,
                    )
                    last_seq = seq
                    errors   = 0
                except requests.RequestException as e:
                    last_seq = seq
                    errors += 1
                    if errors <= 5 or errors % 30 == 0:
                        print(f"[Detector] WARNING snapshot push failed ({errors}x): {e}")
                    if errors >= SNAPSHOT_FAILURE_DISABLE_AFTER:
                        backoff_until = time.time() + SNAPSHOT_FAILURE_BACKOFF_SECONDS
                    else:
                        backoff_until = time.time() + min(3.0, max(0.5, errors * 0.5))

            delay = interval - (time.time() - started)
            if delay > 0:
                _shutdown.wait(delay)
    finally:
        session.close()



# CAMERA HELPER
def _camera_index_label(index: int | None) -> str:
    return "auto" if index is None else str(index)


def _candidate_camera_indexes(index: int | None) -> list[int]:
    seeds = range(max(1, CAMERA_SCAN_LIMIT))
    if index is None:
        candidates = list(seeds)
    elif CAMERA_FALLBACK_SCAN:
        candidates = [index, *seeds]
    else:
        candidates = [index]

    unique = []
    for item in candidates:
        if item >= 0 and item not in unique:
            unique.append(item)
    return unique


def _candidate_camera_resolutions() -> list[tuple[int, int]]:
    candidates = []
    if CAM_WIDTH > 0 and CAM_HEIGHT > 0:
        candidates.append((CAM_WIDTH, CAM_HEIGHT))
    candidates.extend([(640, 480), (320, 240)])

    unique = []
    for item in candidates:
        if item not in unique:
            unique.append(item)
    return unique


def _camera_backend_options() -> list[tuple[int, str]]:
    options = []
    for name in CAMERA_BACKENDS:
        backend = CAMERA_BACKEND_CODES.get(name)
        if backend is not None:
            options.append((backend, name))
    return options or [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_ANY, "ANY")]


def _has_readable_frame(cap: cv2.VideoCapture, warmup_reads: int = 12) -> bool:
    time.sleep(0.4)
    for _ in range(warmup_reads):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return True
        time.sleep(0.05)
    return False


def _open_camera(index: int | None) -> cv2.VideoCapture | None:
    attempt = 0
    while not _shutdown.is_set():
        for camera_index in _candidate_camera_indexes(index):
            for backend, label in _camera_backend_options():
                for width, height in _candidate_camera_resolutions():
                    cap = cv2.VideoCapture(camera_index, backend)
                    if not cap.isOpened():
                        cap.release()
                        continue

                    print(
                        f"[Detector] Trying camera {camera_index} via {label} "
                        f"at {width}x{height} (attempt {attempt + 1})"
                    )

                    # Force MJPEG BEFORE setting resolution - cuts USB bandwidth
                    # and avoids many USB camera read failures on Windows.
                    if CAMERA_FOURCC:
                        cap.set(
                            cv2.CAP_PROP_FOURCC,
                            cv2.VideoWriter_fourcc(*CAMERA_FOURCC[:4]),
                        )
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    if CAMERA_FPS > 0:
                        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    if not _has_readable_frame(cap):
                        print(
                            f"[Detector] Camera {camera_index} via {label} "
                            f"opened but returned no frames"
                        )
                        cap.release()
                        continue

                    actual_fps = cap.get(cv2.CAP_PROP_FPS)
                    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    actual_cc = int(cap.get(cv2.CAP_PROP_FOURCC))
                    fourcc_str = "".join(
                        chr((actual_cc >> 8 * i) & 0xFF) for i in range(4)
                    ).strip()
                    print(
                        f"[Detector] OK camera {camera_index} via {label}: "
                        f"{int(actual_w)}x{int(actual_h)} @ {actual_fps:.0f}fps "
                        f"| codec={fourcc_str or 'unknown'}"
                    )
                    return cap

        attempt += 1
        if CAM_MAX_RETRIES and attempt >= CAM_MAX_RETRIES:
            print(
                f"[Detector] ERROR camera {_camera_index_label(index)} unavailable "
                f"after {attempt} attempt(s)"
            )
            return None
        sleep = min(CAM_RETRY_SLEEP * (2 ** min(attempt, 5)), 30.0)
        print(
            f"[Detector] WARNING camera {_camera_index_label(index)} not ready - "
            f"retry in {sleep:.1f}s"
        )
        time.sleep(sleep)
    return None



# OVERLAY DRAWING
def _draw_queue_overlay(
    frame,
    queue_state: dict,
    done_pending: list,
    scale: float = 1.0,
    bbox_scale_x: float = 1.0,
    bbox_scale_y: float = 1.0,
) -> None:
    if not queue_state and not done_pending:
        return

    h, w = frame.shape[:2]
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    fs_large = max(0.28, 0.38 * scale)
    fs_label = max(0.30, 0.42 * scale)
    fs_info  = max(0.22, 0.30 * scale)
    fs_warn  = max(0.24, 0.32 * scale)

    zone_x1 = 10
    zone_y1 = 10
    zone_x2 = w - 10
    zone_y2 = h - 10

    cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 255, 255), 1)
    lbl = "QUEUE ZONE"
    (lw, lh), _ = cv2.getTextSize(lbl, FONT, fs_large, 1)
    lx, ly = zone_x1 + 6, zone_y1 + lh + 8
    cv2.rectangle(frame, (lx-2, ly-lh-4), (lx+lw+2, ly+4), (0, 0, 0), -1)
    cv2.putText(frame, lbl, (lx, ly), FONT, fs_large, (0, 255, 255), 1)

    for i, alert in enumerate(queue_state.get("noshow_alerts", [])):
        color    = (0, 0, 255) if alert["status"] == "critical" else (0, 165, 255)
        warn_txt = (f"{alert['queue_number']} NO-SHOW "
                    f"Bumping in {alert['seconds_remaining']}s")
        (aw, ah), _ = cv2.getTextSize(warn_txt, FONT, fs_warn, 1)
        ay = h - 10 - i * int(ah * 1.8)
        cv2.rectangle(frame, (8, ay-ah-4), (aw+16, ay+4), (0, 0, 0), -1)
        cv2.putText(frame, warn_txt, (12, ay), FONT, fs_warn, color, 1)

    for person in queue_state.get("active_queue", []):
        bbox = person.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * bbox_scale_x); y1 = int(y1 * bbox_scale_y)
        x2 = int(x2 * bbox_scale_x); y2 = int(y2 * bbox_scale_y)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w-1, x2); y2 = min(h-1, y2)

        status   = person.get("status", "waiting")
        pos      = person.get("position_in_line", 0)
        label    = person.get("queue_label", "Q???")
        wait_str = person.get("wait_time", "0s")

        if status == "missing":
            box_color  = (128, 128, 128)
            info_text  = f"#{pos} | {wait_str} | MISSING"
            text_color = (200, 200, 200)
            thickness  = 1
        elif pos == 1:
            box_color  = (0, 255, 0)
            info_text  = f"#1 NEXT | {wait_str}"
            text_color = (0, 255, 0)
            thickness  = 1
        else:
            box_color  = (0, 165, 255)
            info_text  = f"#{pos} | {wait_str}"
            text_color = (0, 165, 255)
            thickness  = 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
        (tw, th), _ = cv2.getTextSize(label, FONT, fs_label, 1)
        by1 = max(0, y1 - th - 6)
        cv2.rectangle(frame, (x1, by1), (x1+tw+6, y1), box_color, -1)
        cv2.putText(frame, label, (x1+3, y1-3), FONT, fs_label, (255, 255, 255), 1)
        (iw, ih), _ = cv2.getTextSize(info_text, FONT, fs_info, 1)
        info_y = min(h-4, y2+ih+4)
        cv2.rectangle(frame, (x1, info_y-ih-2), (x1+iw+4, info_y+2), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (x1+2, info_y), FONT, fs_info, text_color, 1)

    for person in done_pending:
        bbox = person.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * bbox_scale_x); y1 = int(y1 * bbox_scale_y)
        x2 = int(x2 * bbox_scale_x); y2 = int(y2 * bbox_scale_y)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w-1, x2); y2 = min(h-1, y2)

        label     = person.get("queue_label", "Q???")
        box_color = (0, 0, 255)
        info_text = "DONE-EXIT"
        thickness = 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
        (tw, th), _ = cv2.getTextSize(label, FONT, fs_label, 1)
        by1 = max(0, y1 - th - 6)
        cv2.rectangle(frame, (x1, by1), (x1+tw+6, y1), box_color, -1)
        cv2.putText(frame, label, (x1+3, y1-3), FONT, fs_label, (255, 255, 255), 1)
        (iw, ih), _ = cv2.getTextSize(info_text, FONT, fs_info, 1)
        info_y = min(h-4, y2+ih+4)
        cv2.rectangle(frame, (x1, info_y-ih-2), (x1+iw+4, info_y+2), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (x1+2, info_y), FONT, fs_info, box_color, 1)

    waiting = queue_state.get("queue_count", 0)
    summary = f"Queue: {waiting} waiting"
    (sw, sh), _ = cv2.getTextSize(summary, FONT, fs_large, 1)
    sx = w - sw - 8
    cv2.rectangle(frame, (sx-4, 4), (sx+sw+4, sh+10), (0, 0, 0), -1)
    cv2.putText(frame, summary, (sx, sh+7), FONT, fs_large, (0, 255, 255), 1)



# MAIN DETECTOR LOOP
def run() -> None:
    print("[Detector] Starting...")
    print(f"[Detector] Camera index : {_camera_index_label(CAMERA_INDEX)}")
    print(f"[Detector] Backends     : {', '.join(CAMERA_BACKENDS)}")
    print(f"[Detector] Capture res  : {CAM_WIDTH}x{CAM_HEIGHT}")
    print(f"[Detector] Camera FPS   : {CAMERA_FPS:g}")
    print(f"[Detector] Camera codec : {CAMERA_FOURCC or 'default'}")
    print(f"[Detector] Frame scale  : {FRAME_SCALE}")
    print(f"[Detector] Video scale  : {SNAPSHOT_SCALE}")
    print(f"[Detector] YOLO imgsz   : {YOLO_IMGSZ}")
    print(f"[Detector] YOLO every   : {YOLO_EVERY} frames")
    print(f"[Detector] Push every   : {PUSH_EVERY} YOLO frames")
    print(f"[Detector] API push fps : {API_PUSH_FPS:.1f}")
    print(f"[Detector] Snapshot fps : {SNAPSHOT_FPS:.1f}")
    print(f"[Detector] Snapshot push: {'on' if SNAPSHOT_UPLOAD_ENABLED else 'off'}")

    try:
        model = YOLO(MODEL_PATH)
        print(f"[Detector] OK YOLO loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[Detector] ERROR YOLO load failed: {e}")
        return

    cap = _open_camera(CAMERA_INDEX)
    if cap is None:
        return

    predictor  = PredictionService(QUEUE_CONFIG)

    _fallback: dict = {}
    _next_id        = [1000]

    def _fallback_id(cx: int, cy: int, bucket: int = 80) -> int:
        key = (cx // bucket, cy // bucket)
        if key not in _fallback:
            _fallback[key] = _next_id[0]
            _next_id[0] += 1
        return _fallback[key]

    # Queue: capture loop feeds frames; YOLO thread drains them.
    # maxsize=1 means if YOLO is busy the capture loop drops the frame and
    # keeps reading - the camera never stalls waiting for inference.
    _yolo_queue: _queue.Queue = _queue.Queue(maxsize=1)
    _api_push_queue: _queue.Queue = _queue.Queue(maxsize=1)
    _snapshot_encode_queue: _queue.Queue = _queue.Queue(maxsize=1)

    # Latest YOLO results - written by YOLO thread, read by capture loop.
    _result_lock = threading.Lock()
    _yolo_cache  = {
        "count": 0, "avg_d": 0.0, "max_d": 0.0,
        "persons": [], "tracked_persons": [],
    }

    yolo_frame_idx = [0]   # list so the closure can mutate it

    def _queue_latest(target_queue: _queue.Queue, item) -> None:
        try:
            target_queue.put_nowait(item)
            return
        except _queue.Full:
            pass

        try:
            target_queue.get_nowait()
        except _queue.Empty:
            pass

        try:
            target_queue.put_nowait(item)
        except _queue.Full:
            pass

    # YOLO worker thread 
    def _yolo_worker() -> None:
        while not _shutdown.is_set():
            try:
                small = _yolo_queue.get(timeout=0.5)
            except _queue.Empty:
                continue

            yolo_frame_idx[0] += 1

            try:
                results = model.track(
                    small, imgsz=YOLO_IMGSZ, persist=True,
                    conf=YOLO_CONF, verbose=False,
                )
            except Exception as e:
                print(f"[Detector] WARNING YOLO error: {e}")
                continue

            raw: list = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    try:
                        if int(box.cls[0]) != 0:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        tid  = (int(box.id[0]) if box.id is not None
                                else _fallback_id((x1+x2)//2, (y1+y2)//2))
                        conf = float(box.conf[0]) if box.conf is not None else 0.5
                        raw.append({"track_id": tid,
                                    "bbox": (x1, y1, x2, y2),
                                    "conf": conf})
                    except Exception:
                        continue

            h_frame, w_frame = small.shape[:2]
            frame_area = max(1, h_frame * w_frame)
            raw_before_area = len(raw)
            kept = []
            rejected_small = 0
            rejected_large = 0
            for d in raw:
                area = ((d["bbox"][2]-d["bbox"][0]) *
                        (d["bbox"][3]-d["bbox"][1]))
                if area < MIN_BBOX_AREA:
                    rejected_small += 1
                    continue
                if area > MAX_BBOX_FRAC * frame_area:
                    rejected_large += 1
                    continue
                kept.append(d)
            raw = kept

            persons, tracked_persons = _run_nms(raw)
            if (DETECTION_DEBUG_EVERY > 0 and
                    yolo_frame_idx[0] % DETECTION_DEBUG_EVERY == 0):
                print("[Detector] detections "
                      f"raw={raw_before_area} kept={len(raw)} "
                      f"nms={len(tracked_persons)} "
                      f"small_reject={rejected_small} "
                      f"large_reject={rejected_large} "
                      f"min_area={MIN_BBOX_AREA} imgsz={YOLO_IMGSZ} "
                      f"scale={FRAME_SCALE:.2f}")
            avg_d, max_d = predictor.calculate_density(persons, w_frame, h_frame)

            for tp in tracked_persons:
                tp["appearance"] = _extract_appearance(small, tuple(tp["bbox"]))

            with _data_lock:
                current_data.update({
                    "count":           len(persons),
                    "avg_density":     avg_d,
                    "max_density":     max_d,
                    "active_counters": QUEUE_CONFIG["num_counters"],
                    "timestamp":       time.time(),
                })
            predictor.update(len(persons), current_data, _data_lock)

            with _data_lock:
                ew  = current_data.get("estimated_wait_time",  0.0)
                ar  = current_data.get("arrival_rate",          0.0)
                su  = current_data.get("system_utilization",    0.0)
                p5  = current_data.get("predicted_wait_5min",   0.0)
                p15 = current_data.get("predicted_wait_15min",  0.0)
                p30 = current_data.get("predicted_wait_30min",  0.0)
                ql  = current_data.get("queue_length",          0)

            with _state_lock:
                _shared_state.update({
                    "yolo_frame_idx":       yolo_frame_idx[0],
                    "count":                len(persons),
                    "avg_density":          avg_d,
                    "max_density":          max_d,
                    "active_counters":      QUEUE_CONFIG["num_counters"],
                    "estimated_wait_time":  ew,
                    "arrival_rate":         ar,
                    "system_utilization":   su,
                    "predicted_wait_5min":  p5,
                    "predicted_wait_15min": p15,
                    "predicted_wait_30min": p30,
                    "queue_length":         ql,
                    "timestamp":            time.time(),
                    "tracked_persons": [
                        {"track_id":   p["track_id"],
                         "bbox":       list(p["bbox"]),
                         "conf":       p.get("conf", 0.5),
                         "appearance": p.get("appearance")}
                        for p in tracked_persons
                    ],
                })

            # Cache latest results for overlay drawing in the capture loop
            with _result_lock:
                _yolo_cache.update({
                    "count":           len(persons),
                    "avg_d":           avg_d,
                    "max_d":           max_d,
                    "persons":         persons,
                    "tracked_persons": tracked_persons,
                })

            # Push on every Nth YOLO frame
            if yolo_frame_idx[0] % PUSH_EVERY == 0:
                with _state_lock:
                    payload = {
                        "count":                _shared_state["count"],
                        "avg_density":          _shared_state["avg_density"],
                        "max_density":          _shared_state["max_density"],
                        "active_counters":      _shared_state["active_counters"],
                        "estimated_wait_time":  _shared_state["estimated_wait_time"],
                        "arrival_rate":         _shared_state.get("arrival_rate",         0.0),
                        "system_utilization":   _shared_state.get("system_utilization",   0.0),
                        "predicted_wait_5min":  _shared_state.get("predicted_wait_5min",  0.0),
                        "predicted_wait_15min": _shared_state.get("predicted_wait_15min", 0.0),
                        "predicted_wait_30min": _shared_state.get("predicted_wait_30min", 0.0),
                        "queue_length":         _shared_state.get("queue_length",         0),
                        "timestamp":            _shared_state["timestamp"],
                        "yolo_frame_idx":       _shared_state.get("yolo_frame_idx", 0),
                        "tracked_persons":      _shared_state["tracked_persons"],
                    }
                _queue_latest(_api_push_queue, payload)

    def _api_push_worker() -> None:
        interval = 1.0 / max(0.1, API_PUSH_FPS)

        while not _shutdown.is_set():
            started = time.time()
            try:
                payload = _api_push_queue.get(timeout=0.5)
            except _queue.Empty:
                continue

            _push_to_api(payload)

            delay = interval - (time.time() - started)
            if delay > 0:
                _shutdown.wait(delay)

    def _snapshot_encode_worker() -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX

        while not _shutdown.is_set():
            try:
                frame = _snapshot_encode_queue.get(timeout=0.5)
            except _queue.Empty:
                continue

            if frame is None or frame.size == 0:
                continue

            if SNAPSHOT_SCALE <= 0:
                h_orig, w_orig = frame.shape[:2]
                overlay = cv2.resize(
                    frame,
                    (
                        int(w_orig * FRAME_SCALE),
                        int(h_orig * FRAME_SCALE),
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )
                overlay_scale_x = 1.0
                overlay_scale_y = 1.0
                overlay_text_scale = FRAME_SCALE
            elif abs(SNAPSHOT_SCALE - 1.0) < 0.01:
                overlay = frame.copy()
                overlay_scale_x = 1.0 / max(0.01, FRAME_SCALE)
                overlay_scale_y = 1.0 / max(0.01, FRAME_SCALE)
                overlay_text_scale = 1.0
            else:
                h_orig, w_orig = frame.shape[:2]
                overlay = cv2.resize(
                    frame,
                    (
                        int(w_orig * SNAPSHOT_SCALE),
                        int(h_orig * SNAPSHOT_SCALE),
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )
                overlay_scale_x = SNAPSHOT_SCALE / max(0.01, FRAME_SCALE)
                overlay_scale_y = SNAPSHOT_SCALE / max(0.01, FRAME_SCALE)
                overlay_text_scale = SNAPSHOT_SCALE

            with _state_lock:
                api_queue_state = _shared_state.get("api_queue_state", {})
                done_pending    = _shared_state.get("done_pending",    [])
                ew_display      = _shared_state.get("estimated_wait_time", 0.0)
            with _result_lock:
                last_count = _yolo_cache["count"]
                last_max_d = _yolo_cache["max_d"]

            _draw_queue_overlay(
                overlay,
                api_queue_state,
                done_pending,
                scale=overlay_text_scale,
                bbox_scale_x=overlay_scale_x,
                bbox_scale_y=overlay_scale_y,
            )

            hud_fs = max(0.28, 0.50 * overlay_text_scale)
            text = (
                f"Count:{last_count} Den:{last_max_d:.1f} "
                f"Wait:{ew_display:.0f}m"
            )
            (tw, th), _ = cv2.getTextSize(text, font, hud_fs, 1)
            cv2.rectangle(overlay, (4, 4), (tw + 10, th + 10), (0, 0, 0), -1)
            cv2.putText(
                overlay,
                text,
                (6, th + 6),
                font,
                hud_fs,
                (0, 255, 0),
                1,
            )

            ok_jpg, buf = cv2.imencode(
                ".jpg",
                overlay,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            if ok_jpg:
                with _state_lock:
                    _shared_state["snapshot_jpg"] = buf.tobytes()
                    _shared_state["snapshot_seq"] += 1

    yolo_thread = threading.Thread(target=_yolo_worker,
                                   name="YOLOWorker", daemon=True)
    yolo_thread.start()

    api_push_thread = threading.Thread(
        target=_api_push_worker,
        name="APIPushWorker",
        daemon=True,
    )
    api_push_thread.start()

    snapshot_thread: threading.Thread | None = None
    snapshot_encode_thread: threading.Thread | None = None
    if SNAPSHOT_UPLOAD_ENABLED:
        snapshot_encode_thread = threading.Thread(
            target=_snapshot_encode_worker,
            name="SnapshotEncoder",
            daemon=True,
        )
        snapshot_encode_thread.start()
        snapshot_thread = threading.Thread(target=_snapshot_upload_worker,
                                           name="SnapshotUploader", daemon=True)
        snapshot_thread.start()

    # Capture loop - never blocks on YOLO.
    frame_idx   = 0
    _fps_t0     = time.time()
    _fps_frames = 0
    snapshot_interval = 1.0 / max(0.1, SNAPSHOT_FPS)
    next_snapshot_at  = 0.0
    read_failures = 0

    print("[Detector] Loop started - press Ctrl-C to stop")

    try:
        while not _shutdown.is_set():
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                read_failures += 1
                if read_failures < CAMERA_READ_FAIL_LIMIT:
                    time.sleep(CAMERA_READ_RETRY_SLEEP)
                    continue
                print(
                    "[Detector] WARNING camera returned no frames "
                    f"({read_failures} reads) - reconnecting..."
                )
                cap.release()
                cap = _open_camera(CAMERA_INDEX)
                if cap is None:
                    break
                read_failures = 0
                continue
            read_failures = 0

            frame_idx   += 1
            _fps_frames += 1

            elapsed = time.time() - _fps_t0
            if elapsed >= 5.0:
                with _result_lock:
                    cnt = _yolo_cache["count"]
                print(f"[Detector] {_fps_frames/elapsed:.1f} fps | "
                      f"count={cnt} | push_errors={_push_errors}")
                _fps_t0     = time.time()
                _fps_frames = 0

            # Downscale for YOLO only. Dashboard encoding happens in a
            # separate worker from the original frame.
            if FRAME_SCALE < 1.0:
                h_orig, w_orig = frame.shape[:2]
                small = cv2.resize(
                    frame,
                    (int(w_orig * FRAME_SCALE), int(h_orig * FRAME_SCALE)),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                small = frame

            # Feed YOLO thread every YOLO_EVERY frames.
            # put_nowait() drops the frame if YOLO is still busy - this is
            # intentional; we never want the camera read to stall.
            if frame_idx % YOLO_EVERY == 0:
                yolo_frame = small.copy()
                try:
                    _yolo_queue.put_nowait(yolo_frame)
                except _queue.Full:
                    try:
                        _yolo_queue.get_nowait()
                    except _queue.Empty:
                        pass
                    try:
                        _yolo_queue.put_nowait(yolo_frame)
                    except _queue.Full:
                        pass

            # Queue only the newest frame for dashboard video. Drawing and
            # JPEG encoding happen outside the camera read loop.
            now = time.time()
            if SNAPSHOT_UPLOAD_ENABLED and now >= next_snapshot_at:
                next_snapshot_at = now + snapshot_interval
                _queue_latest(_snapshot_encode_queue, frame.copy())

    finally:
        _shutdown.set()
        print("[Detector] Releasing camera...")
        cap.release()
        _session.close()
        if snapshot_thread is not None:
            snapshot_thread.join(timeout=2.0)
        if snapshot_encode_thread is not None:
            snapshot_encode_thread.join(timeout=2.0)
        api_push_thread.join(timeout=2.0)
        yolo_thread.join(timeout=2.0)
        print("[Detector] Stopped.")


if __name__ == "__main__":
    run()
