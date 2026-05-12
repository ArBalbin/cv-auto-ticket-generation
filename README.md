# QueueFlow

QueueFlow is a computer-vision-based queue monitoring and automated queue ticket
prototype for campus service offices. It uses a local camera and YOLOv8n person
detection to identify people entering a queue zone, assigns queue numbers,
generates QR-enabled queue tickets, stores queue records in MySQL, and exposes
staff dashboard and student queue-status APIs through a FastAPI backend.

The target deployment uses a thermal printer for physical queue tickets. The
current prototype generates PDF tickets with the same queue number, short code,
QR link, and JWT-backed validation flow so the ticket workflow can be demonstrated
without printer hardware.

## Current Progress

Last updated: May 12, 2026

Implemented:

- FastAPI backend with dashboard routes, auth routes, detector upload routes,
  queue APIs, crowd APIs, and health check.
- Separate OpenCV + YOLOv8n detector process that reads a camera feed, performs
  person detection/tracking, sends metadata to the backend, and uploads annotated
  snapshots for the dashboard.
- Queue tracker with queue-zone filtering, candidate confirmation, duplicate
  suppression, hybrid re-entry matching, no-show handling, manual force-new
  override, and staff done/reset actions.
- Ticket worker that generates PDF queue tickets with QR codes, short codes, JWT
  tokens, and MySQL persistence.
- Wait-time prediction using an M/M/c baseline, short trend projection, Holt's
  double exponential smoothing, and growth-ratio mean reversion for current,
  5-minute, 15-minute, and 30-minute estimates.
- MySQL schemas for users, queue records, queue events, counter configuration
  history, and crowd snapshots.
- Optional Redis cache for live state/snapshot mirroring in cloud deployments.
- Optional S3-compatible object storage support for generated ticket PDFs.
- Render/cloud deployment files and production environment examples.

In progress / demo-limited:

- Thermal printer integration is planned for final deployment; PDF ticket
  generation is the current demo fallback.
- Detector still runs on the camera-connected local machine even when the
  FastAPI backend is cloud-hosted.
- Mobile app integration is documented as a consuming client for QR/status lookup;
  this repository contains the backend, detector, dashboard, and ticket services.
- Live testing depends on available camera position, queue-zone calibration, and
  local service-office conditions.

## Repository Structure

```text
Crowd_Monitoring/
|-- app/
|   |-- main.py                    # FastAPI app entry point
|   |-- detector.py                # Camera + YOLOv8n detector process
|   |-- state.py                   # Runtime state and snapshot helpers
|   |-- core/
|   |   |-- config.py              # Environment/config validation
|   |   `-- security.py            # Auth, sessions, staff registration
|   |-- database/
|   |   `-- database_handler.py    # MySQL pool and persistence helpers
|   |-- routers/
|   |   |-- auth.py                # /api/auth/*
|   |   |-- crowd.py               # /api/stats, snapshot, history, video
|   |   |-- detector_api.py        # /yolo/push-frame, /yolo/update
|   |   |-- health.py              # /health
|   |   |-- pages.py               # HTML dashboard/login routes
|   |   `-- queue.py               # /api/queue/*
|   |-- services/
|   |   |-- cache_service.py       # Optional Redis cache
|   |   |-- object_storage_service.py
|   |   |-- prediction_service.py  # M/M/c + trend/Holt/mean-reversion forecasts
|   |   |-- queue_service.py       # Queue business logic
|   |   |-- queue_tracker.py       # Queue number tracking and re-entry matching
|   |   |-- ticket_printer.py      # PDF ticket, QR, JWT, short code
|   |   `-- ticket_service.py      # Background ticket worker
|   `-- templates/                 # Dashboard/login/register pages
|-- database_sql/                  # Local/cloud database schemas
|-- ML/                            # Training-related files
|-- Model/                         # Place YOLO weights here
|-- CURRENT_PROGRESS.md            # Current implementation progress
|-- ALGORITHMS.md                  # Algorithm reference
|-- SYSTEM_ARCHITECTURE_DOCUMENTATION.md
|-- SYSTEM_DATAFLOW_DOCUMENTATION.md
|-- SYSTEM_FLOW_DOCUMENTATION.md
|-- DATABASE_RELATIONSHIPS_DOCUMENTATION.md
|-- CLOUD_DEPLOYMENT_CHECKLIST.md
|-- CLOUD_CACHE_SETUP.md
|-- Dockerfile
|-- Procfile
|-- render.yaml
`-- requirements.txt
```

## Requirements

- Python 3.12
- MySQL database
- Camera connected to the detector machine
- YOLOv8n weights at `Model/yolov8n.pt`
- Optional Redis instance for cloud cache
- Optional S3-compatible object storage for ticket PDFs
- Thermal printer for final deployment, or PDF output for prototype demo

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup

Copy `.env.example` to `.env`, then fill in values for:

- `APP_ENV`
- `PORTAL_BASE_URL`
- `CAM_TOKEN`
- `JWT_SECRET_KEY`
- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`
- `STAFF_REGISTRATION_ENABLED`
- `STAFF_REGISTRATION_CODE`
- camera and YOLO tuning values as needed

For production/cloud setup, use `.env.production.example` as the starting point.

## Database Setup

For a fresh cloud database, import:

```text
database_sql/schema_cloud_ready.sql
```

For a clean Aiven reset that drops and recreates QueueFlow tables, use:

```text
database_sql/aiven_clean_full_schema.sql
```

## Running Locally

Start the FastAPI backend:

```bash
python app/main.py
```

Or with Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 5000
```

Start the detector in a separate terminal on the camera-connected machine:

```bash
python app/detector.py
```

Common local URLs:

- Login: `http://localhost:5000/login`
- Queue dashboard: `http://localhost:5000/dashboard/queueflow`
- Queue analytics: `http://localhost:5000/dashboard/queue-analytics`
- Computer vision dashboard: `http://localhost:5000/dashboard/computer-vision`
- Health check: `http://localhost:5000/health`
- Queue list API: `http://localhost:5000/api/queue/list`
- Queue prediction API: `http://localhost:5000/api/queue/prediction`

## Cloud Deployment

Cloud deployment is supported for the FastAPI backend. The detector should still
run on the local machine connected to the camera, with `API_BASE_URL` pointed to
the hosted backend URL.

Minimum cloud start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT --proxy-headers
```

Use these docs for deployment:

- `CLOUD_DEPLOYMENT_CHECKLIST.md`
- `CLOUD_CACHE_SETUP.md`

## Important Documentation

- `CURRENT_PROGRESS.md` - current progress, completed work, remaining work
- `SYSTEM_ARCHITECTURE_DOCUMENTATION.md` - component architecture
- `SYSTEM_FLOW_DOCUMENTATION.md` - end-to-end runtime flow
- `SYSTEM_DATAFLOW_DOCUMENTATION.md` - data movement and storage
- `DATABASE_RELATIONSHIPS_DOCUMENTATION.md` - database entities and relations
- `ALGORITHMS.md` - detection, queue tracking, re-ID, no-show, prediction

## Team

| Name | GitHub |
|---|---|
| Archie Balbin | [@ArBalbin](https://github.com/ArBalbin) |
