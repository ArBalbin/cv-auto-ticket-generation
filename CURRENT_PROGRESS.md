# QueueFlow Current Progress

Last updated: May 12, 2026

This file summarizes the current implementation status of QueueFlow. It is meant
to give advisers, panelists, and developers a quick view of what already works,
what is demo-only, and what still needs final deployment work.

## Project Status

QueueFlow is currently a working prototype with the core backend, detector,
queue tracking, ticket generation, dashboard, database persistence, and
prediction features implemented. The system is aligned with a YOLOv8n-based
computer-vision queue workflow.

The prototype can:

- Run a FastAPI backend.
- Receive YOLO detection metadata from a separate detector process.
- Track queue entrants inside a configured queue zone.
- Assign queue numbers automatically.
- Generate PDF queue tickets with QR codes and short codes.
- Persist ticket and queue records in MySQL.
- Show live queue and crowd data to staff through dashboard routes.
- Provide status lookup APIs for ticket holders.
- Estimate wait times for current, 5-minute, 15-minute, and 30-minute horizons.

## Completed Components

### Backend API

Implemented in `app/main.py` and `app/routers/`.

- FastAPI application setup
- CORS and trusted-host configuration
- Error handling for API and dashboard routes
- Auth router for staff login, registration, profile, and logout
- Detector router for camera metadata and snapshot uploads
- Queue router for queue list, status, prediction, analytics, zone, done, reset,
  no-show configuration, force-new, and counter adjustment
- Crowd router for current stats, snapshot, history, and MJPEG stream
- Health router for deployment smoke tests

### Computer Vision Detector

Implemented in `app/detector.py`.

- OpenCV camera capture
- Camera backend scanning and retry handling
- YOLOv8n model loading from `Model/yolov8n.pt`
- Configurable camera resolution, frame scaling, YOLO interval, confidence, and
  push rate
- NMS-style duplicate filtering before backend upload
- Appearance feature extraction using HSV histograms
- Annotated snapshot upload for dashboard view
- Metadata upload to `/yolo/push-frame`
- Snapshot upload to `/yolo/update`

### Queue Tracking

Implemented in `app/services/queue_tracker.py` and `app/services/queue_service.py`.

- Queue zone membership checking
- Candidate accumulation before assigning a queue number
- Static/low-motion rejection
- Bounding-box smoothing
- Duplicate person suppression
- Hybrid re-entry matching using spatial proximity, recency, and appearance
  similarity
- Done blacklist to reduce accidental reassignment of already served persons
- No-show countdown and no-show status update
- Manual force-new override for missed/twin cases
- Staff done and reset operations
- Counter adjustment and service-time refresh support

### Ticket Generation

Implemented in `app/services/ticket_service.py` and `app/services/ticket_printer.py`.

- Background ticket worker
- PDF queue ticket generation
- QR code generation
- Short-code generation
- JWT token generation and validation
- Queue status URL generation using `PORTAL_BASE_URL`
- MySQL persistence for queue records
- Ticket cleanup on reset, duplicate removal, no-show, or invalid tracker state
- Optional S3-compatible ticket upload support through object storage service

Current limitation:

- Thermal printer output is not yet wired to hardware. PDF generation is the
  active demonstration fallback.

### Prediction

Implemented in `app/services/prediction_service.py`.

- Crowd density calculation
- Arrival-rate estimation from recent count history
- M/M/c wait-time baseline
- Short-term trend projection
- Holt's double exponential smoothing for medium-term forecasting
- Growth-ratio plus mean-reversion logic for longer forecast stability
- Current, 5-minute, 15-minute, and 30-minute wait-time fields

### Database

Implemented through `app/database/database_handler.py` and `database_sql/`.

Main tables:

- `users`
- `queue_records`
- `queue_events`
- `counter_config_history`
- `crowd_snapshots`

Database-supported features:

- Staff account storage
- Ticket records with queue number, short code, JWT, PDF path, status, and expiry
- Queue event audit trail
- Counter configuration history
- Crowd snapshot history
- Average service-time measurement from served records

### Cloud Readiness

Implemented through `Dockerfile`, `Procfile`, `render.yaml`,
`.env.production.example`, and cloud schema files.

- FastAPI backend can be hosted on Render or similar platforms.
- `PORTAL_BASE_URL` controls QR/status links.
- `APP_ENV=production` validates critical deployment values.
- Redis cache is optional.
- Object storage for PDFs is optional.
- Detector remains local because it requires direct camera access.

## Demo Workflow Available Now

1. Start the FastAPI backend.
2. Start `app/detector.py` on the camera-connected machine.
3. Log in to the dashboard.
4. Configure or confirm the queue zone.
5. Let a person enter the queue zone.
6. System confirms the candidate and assigns a queue number.
7. Ticket worker generates a PDF ticket with QR and short code.
8. Queue record is saved to MySQL.
9. Staff can monitor the queue and predictions from the dashboard.
10. Student can check queue status using QR URL or short code.
11. Staff can mark the queue number done or no-show.

## Known Limitations

- Thermal printer hardware integration is pending.
- The generated ticket is currently a PDF demo substitute.
- The detector requires local camera access and does not run inside the cloud
  backend.
- Forecast accuracy still depends on real service-office testing and calibration.
- Queue-zone coordinates must be calibrated per camera placement.
- Some older documentation and comments may mention external mobile app behavior;
  this repository mainly contains backend, detector, dashboard, and ticket logic.
- Some references in older research drafts were replaced in the manuscript, but
  system docs should continue to cite only implementation behavior.

## Recommended Next Work

1. Connect and test actual thermal printer output.
2. Run live queue trials in the target service area.
3. Calibrate queue-zone coordinates and YOLO thresholds using real camera angles.
4. Validate wait-time predictions against actual served/no-show records.
5. Finalize the student-facing mobile or web status interface.
6. Add automated tests for queue tracker edge cases.
7. Add screenshots of the dashboard and sample PDF ticket to the README.
8. Review all cloud environment variables before production deployment.

## Documentation Map

- `README.md` - setup, run instructions, and project overview
- `SYSTEM_ARCHITECTURE_DOCUMENTATION.md` - architecture and component layout
- `SYSTEM_FLOW_DOCUMENTATION.md` - runtime system flow
- `SYSTEM_DATAFLOW_DOCUMENTATION.md` - data movement and storage
- `DATABASE_RELATIONSHIPS_DOCUMENTATION.md` - database entities and relations
- `ALGORITHMS.md` - detection, tracking, re-identification, and prediction logic
- `CLOUD_DEPLOYMENT_CHECKLIST.md` - deployment checklist
- `CLOUD_CACHE_SETUP.md` - Redis cache setup

