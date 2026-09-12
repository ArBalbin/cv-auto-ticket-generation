# QueuEx

QueuEx is a computer-vision queue management prototype for campus service
offices. A camera watches the queue area; YOLOv8n detects and tracks people,
ArcFace face recognition identifies enrolled students, and the system issues
each recognised student a queue number automatically — no kiosk button, no
staff typing, no manual input.

Students enrol once through a mobile app (institutional Google sign-in, then a
guided face capture). On a later visit they tap **Join the Queue**, walk into
the queue area, and the camera does the rest.

Last updated: September 12, 2026

> **This repository is the backend.** The system is three separate repositories:
>
> | Part | Repo | Stack |
> |---|---|---|
> | **Backend + detector** (this one) | `Crowd_Monitoring` | FastAPI, YOLOv8n, InsightFace, MySQL |
> | **Student mobile app** | `queueflow_mobile` | Flutter |
> | **Staff dashboard** | `CV-frontend` | React + Vite + TypeScript |
>
> The backend serves **JSON only**. It has no HTML pages — the staff dashboard
> is the separate React app, and the student interface is the Flutter app.

## How a ticket gets issued

1. A student signs in to the mobile app with their Gbox account and captures
   their face. Only a 512-number embedding is stored — **never a photograph**.
2. Later, at the office, they tap **Join the Queue** in the app. This arms
   their intent for a short window; recognition alone issues nothing, so a
   student merely walking past the camera is never charged a ticket.
3. The detector confirms a person is genuinely present in the queue zone
   (rejecting static objects and momentary ghost tracks), then extracts a face
   embedding from the person crop.
4. The embedding is matched against enrolled students. A match is accepted only
   if it clears **both** an absolute similarity threshold and a margin over the
   runner-up. Anything ambiguous is refused and escalated to staff rather than
   guessed.
5. On an accepted match the system mints a queue number, generates a ticket
   (PDF now, thermal printer at deployment) carrying the number, the student's
   name, a QR code and a short access code, and pushes live status to the app.
6. Staff mark the number done from the dashboard.

Walk-ins without an enrolled face are handled by staff typing the number
manually — the system never refuses service to someone who has not enrolled.

## What is measured, not claimed

Accuracy figures come from experiments that reproduce with a named script. See
`ACCURACY.md` — it is split into **Part A (measured)** and **Part B
(estimated)**, and the two must not be mixed.

| Result | Value | Script |
|---|---|---|
| Face recognition ROC AUC | 0.999907 | `ML/evaluate_face_accuracy.py` |
| Equal Error Rate | 0.42 % | same |
| Rank-1 identification | 100 % (45/45) | same |
| Wrong identities issued | **0** | same |
| Unenrolled strangers refused | **45/45 (100 %)** | same |
| API under 50 concurrent users | p50 12 ms, p99 85 ms, 0 failures | `load_testing/run_load_test.py` |

Person-detection accuracy on the deployment camera is **not yet measured** —
the harness (`ML/evaluate_yolo_accuracy.py`) is built and verified, but needs
ground-truth labelling. `ACCURACY.md` §A4 says so plainly rather than quoting
COCO figures as if they described this deployment.

## Repository structure

```text
Crowd_Monitoring/
|-- app/
|   |-- main.py                    # FastAPI app entry point
|   |-- detector.py                # Camera + YOLOv8n detector process
|   |-- state.py                   # Runtime state and snapshot helpers
|   |-- core/
|   |   |-- config.py              # Environment/config validation
|   |   |-- database.py            # Engine/session helpers
|   |   `-- security.py            # Staff auth, sessions, Google ID tokens
|   |-- database/
|   |   `-- database_handler.py    # MySQL pool and persistence helpers
|   |-- routers/
|   |   |-- auth.py                # /api/auth/*        staff login
|   |   |-- crowd.py               # /api/stats, snapshot, history, video
|   |   |-- detector_api.py        # /yolo/push-frame, /yolo/update
|   |   |-- health.py              # /health, /
|   |   |-- queue.py               # /api/queue/*       queue operations
|   |   `-- students.py            # /api/students/*    enrolment, join gate
|   `-- services/
|       |-- cache_service.py       # Optional Redis cache
|       |-- face_service.py        # ArcFace embeddings + match decision rule
|       |-- object_storage_service.py
|       |-- prediction_service.py  # M/M/c + trend/Holt/mean-reversion forecasts
|       |-- queue_service.py       # Queue business logic
|       |-- queue_tracker.py       # Presence, tracking, re-entry, minting
|       |-- ticket_printer.py      # PDF ticket, QR, JWT, short code
|       `-- ticket_service.py      # Background ticket worker
|-- ML/
|   |-- calibrate_face_recognition.py   # Threshold calibration
|   |-- evaluate_face_accuracy.py       # FAR/FRR/EER/ROC + open-set test
|   |-- evaluate_yolo_accuracy.py       # Person-count accuracy harness
|   |-- generate_accuracy_charts.py     # Figures for the evaluation chapter
|   |-- report_recognition_metrics.py   # Live accuracy from the database
|   `-- figures/                        # Generated evaluation figures
|-- load_testing/                  # Locust suite + results
|-- database_sql/                  # Schemas and migrations
|-- Model/                         # YOLO weights (yolov8n.pt)
|-- calibration_data/              # Face photos — gitignored, never committed
`-- requirements.txt
```

## Requirements

- Python 3.12
- MySQL database
- Camera connected to the detector machine
- YOLOv8n weights at `Model/yolov8n.pt`
- Optional Redis instance for cloud cache
- Optional S3-compatible object storage for ticket PDFs

InsightFace model weights (`buffalo_s`) download automatically on first run.

```bash
pip install -r requirements.txt
```

## Environment setup

Copy `.env.example` to `.env` and fill in:

- `APP_ENV`
- `JWT_SECRET_KEY`, `CAM_TOKEN`
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USERNAME`, `DB_PASSWORD`
- `GOOGLE_OAUTH_CLIENT_ID`, `GBOX_ALLOWED_DOMAIN` — required for student sign-in
- `STAFF_REGISTRATION_ENABLED`, `STAFF_REGISTRATION_CODE`
- Camera and YOLO tuning values as needed

The face thresholds are **not** in `.env.example`, because their defaults are
calibrated values rather than site settings and should not be changed casually.
They are still overridable when a deployment genuinely needs it:
`FACE_MATCH_THRESHOLD` (0.30), `FACE_MARGIN_THRESHOLD` (0.15),
`FACE_MIN_DETECT_CONF` (0.60). Read `ACCURACY.md` §A1 before touching them —
lowering the margin below 0.15 is measured to let strangers through.

**`PORTAL_BASE_URL` usually needs no value.** It is baked into every printed
ticket's QR code, and left at `localhost` the QR would resolve to the student's
own phone instead of the server. In development the backend now detects the
machine's LAN address automatically and prints it at startup:

```
[API] Ticket QR codes will point to: http://10.23.83.12:5000
```

Set it explicitly only to pin a specific address or a public domain. Production
requires an explicit public URL and refuses to start on a loopback address.

For production/cloud setup, start from `.env.production.example`.

## Database setup

Fresh database:

```text
database_sql/schema_cloud_ready.sql
```

Clean Aiven reset:

```text
database_sql/aiven_clean_full_schema.sql
```

Then apply the migrations in `database_sql/` in filename order. See
`DATABASE_RELATIONSHIPS_DOCUMENTATION.md` for what each one adds.

## Running locally

Backend:

```bash
python app/main.py
```

Detector, in a separate terminal on the camera machine:

```bash
python app/detector.py
```

> Use `python -u` when piping either process's output to a file or another
> tool. Python buffers stdout when it is not a terminal, which makes a running
> process look hung — this cost two wrong diagnoses during development.

Useful endpoints:

- Interactive API docs: `http://localhost:5000/docs`
- Health check: `http://localhost:5000/health`
- Queue list: `http://localhost:5000/api/queue/list`
- Public display board data: `http://localhost:5000/api/queue/display`
- Prediction: `http://localhost:5000/api/queue/prediction`

The staff dashboard and student app are separate applications; point them at
this backend's address.

## Evaluation and testing

```bash
python ML/evaluate_face_accuracy.py          # biometric accuracy + figures
python ML/generate_accuracy_charts.py        # calibration figures
python ML/report_recognition_metrics.py      # live accuracy from the database
python load_testing/run_load_test.py         # API load test
python ML/evaluate_yolo_accuracy.py capture  # person-detection ground truth
```

The load test seeds a real ticket, runs Locust headless, writes CSVs, and
removes the seeded ticket afterwards. It targets `127.0.0.1` by default, not
`localhost` — on Windows the latter resolves to IPv6 first and adds a spurious
~2 s to every new connection, which earlier runs mistook for server latency.

## Cloud deployment

The FastAPI backend deploys to a cloud host; the detector must still run on the
machine physically connected to the camera, with `API_BASE_URL` pointed at the
hosted backend.

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT --proxy-headers
```

See `CLOUD_DEPLOYMENT_CHECKLIST.md` and `CLOUD_CACHE_SETUP.md`.

## Documentation

| File | Contents |
|---|---|
| `ALGORITHMS.md` | The nine algorithms, with the thresholds actually deployed |
| `ACCURACY.md` | Measured results (Part A) and estimated behaviour (Part B) |
| `FACE_RECOGNITION_CALIBRATION.md` | How the thresholds were chosen |
| `DATABASE_RELATIONSHIPS_DOCUMENTATION.md` | All eight tables and their relationships |
| `SYSTEM_ARCHITECTURE_DOCUMENTATION.md` | Component architecture |
| `SYSTEM_FLOW_DOCUMENTATION.md` | End-to-end runtime flow |
| `SYSTEM_DATAFLOW_DOCUMENTATION.md` | Data movement and storage |
| `CURRENT_PROGRESS.md` | Implementation status |
| `PITFALLS.md` | Real bugs and design errors hit during development |

## Privacy

- Face **embeddings** are stored; face **photographs** are not. Enrolment
  images exist only in the phone's memory during capture.
- An embedding is one-way: it supports comparison, but the original face cannot
  be reconstructed from it.
- `calibration_data/` holds real people's photos for threshold calibration and
  is gitignored. It must never be committed.
- Students choose to be recognised each visit through the Join the Queue gate.

## Team

| Name | GitHub |
|---|---|
| Archie Balbin | [@ArBalbin](https://github.com/ArBalbin) |
