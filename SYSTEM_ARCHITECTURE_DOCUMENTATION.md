# QueueFlow System Architecture Documentation

## 1. Architecture Overview

QueueFlow is designed as a modular computer-vision queue monitoring system.
It separates camera detection, backend API services, queue business logic,
ticket generation, data persistence, live cache, dashboard pages, and mobile
queue tracking.

High-level architecture:

```text
Mobile App
Dashboard Browser
      |
      v
FastAPI Backend
      |
      +--> Queue Service
      |       |
      |       v
      |   Queue Tracker
      |
      +--> Ticket Service
      |       |
      |       v
      |   Ticket Printer / PDF Generator
      |
      +--> MySQL Database
      |
      +--> Redis Cache, optional

Detector Process
      |
      +--> Camera
      +--> YOLO Model
      +--> FastAPI Backend
```

## 2. Main Architectural Layers

QueueFlow can be understood in six layers:

```text
1. Presentation Layer
2. API Layer
3. Computer Vision Layer
4. Business Logic Layer
5. Persistence Layer
6. Cache / Runtime State Layer
```

## 3. Presentation Layer

The presentation layer contains the user interfaces.

### 3.1 Web Dashboard

Files:

```text
app/templates/queue_analytics.html
app/templates/profile.html
```

Routes:

```text
/dashboard/computer-vision
/dashboard/queueflow
/dashboard/queue-analytics
/dashboard/profile
```

Responsibilities:

- Shows queue analytics.
- Shows live queue statistics.
- Shows wait-time forecast bars.
- Shows wait band histogram.
- Shows active queue and recent completed queue records.
- Displays live camera snapshot or video stream.

### 3.2 Mobile App

Project:

```text
C:\Users\Lenovo\Desktop\queueflow_mobile
```

Responsibilities:

- Scans ticket QR code.
- Extracts queue number and token.
- Calls the backend queue status API.
- Displays the user's queue position and status.
- Lets the user navigate between queue, history, profile, and help screens.

## 4. API Layer

The API layer is implemented using FastAPI.

Main file:

```text
app/main.py
```

Responsibilities:

- Creates the FastAPI app.
- Registers routers.
- Sets CORS middleware.
- Starts the backend server.
- Warms up the database connection pool.
- Provides error handling.

Routers:

```text
app/routers/pages.py
app/routers/auth.py
app/routers/detector_api.py
app/routers/crowd.py
app/routers/queue.py
app/routers/health.py
```

### 4.1 Pages Router

File:

```text
app/routers/pages.py
```

Responsibilities:

- Serves login page.
- Serves dashboard pages.
- Serves live MJPEG video stream.

Important endpoints:

```text
GET /
GET /login
POST /login
GET /logout
GET /dashboard/computer-vision
GET /dashboard/queueflow
GET /dashboard/queue-analytics
GET /dashboard/profile
GET /video
GET /api/crowd/video
```

### 4.2 Detector API Router

File:

```text
app/routers/detector_api.py
```

Responsibilities:

- Receives YOLO detection metadata.
- Receives annotated camera snapshots.
- Updates backend state.
- Sends cleaned queue state back to detector.

Important endpoints:

```text
POST /yolo/push-frame
POST /yolo/update
```

### 4.3 Queue Router

File:

```text
app/routers/queue.py
```

Responsibilities:

- Provides live queue state.
- Provides mobile ticket status lookup.
- Provides wait-time prediction.
- Provides staff queue actions.

Important endpoints:

```text
GET /api/queue/list
GET /api/queue/status
GET /api/queue/prediction
GET /api/queue/analytics
GET /api/queue/zone
POST /api/queue/done
POST /api/queue/reset
POST /api/queue/zone
POST /api/queue/adjust_counters
```

### 4.4 Crowd Router

File:

```text
app/routers/crowd.py
```

Responsibilities:

- Provides crowd statistics.
- Provides latest snapshot image.
- Provides recent crowd count history.

Important endpoints:

```text
GET /api/stats
GET /api/crowd/data
GET /api/snapshot
GET /api/history
GET /api/data
```

### 4.5 Health Router

File:

```text
app/routers/health.py
```

Responsibilities:

- Reports backend status.
- Reports database availability.
- Reports Redis cache availability.
- Reports whether a snapshot exists.

Endpoint:

```text
GET /health
```

## 5. Computer Vision Layer

Main file:

```text
app/detector.py
```

Responsibilities:

- Opens the camera.
- Loads YOLO model.
- Detects people in camera frames.
- Tracks people using YOLO track IDs.
- Computes crowd density.
- Computes prediction metrics.
- Draws queue overlays.
- Sends metadata and snapshots to the backend.

Key dependencies:

```text
OpenCV
Ultralytics YOLO
requests
PredictionService
```

The detector is intentionally separate from the FastAPI backend. This prevents
camera processing from blocking dashboard/API requests.

Detector outputs:

```text
detection metadata
tracked people
prediction metrics
annotated JPEG snapshots
```

## 6. Business Logic Layer

The business logic layer turns raw detections into queue operations.

### 6.1 Queue Service

File:

```text
app/services/queue_service.py
```

Responsibilities:

- Filters detector results.
- Removes weak detections.
- Remaps unstable track IDs.
- Sends detections to the queue tracker.
- Builds queue prediction response.
- Builds queue analytics response.
- Handles staff queue actions.
- Triggers ticket generation for new queue people.

### 6.2 Queue Tracker

File:

```text
app/services/queue_tracker.py
```

Responsibilities:

- Maintains active queue in memory.
- Assigns queue numbers.
- Tracks person status.
- Handles missing people.
- Handles no-show countdowns.
- Maintains completed queue records.
- Matches returning people using spatial and appearance logic.

Runtime data:

```text
active_queue
completed_queue
noshow_alerts
appearance_rejections
```

Important note:

The active queue tracker is currently process memory. In cloud production,
running multiple backend instances would require moving this state to Redis or
MySQL, or running only one backend instance.

### 6.3 Prediction Service

File:

```text
app/services/prediction_service.py
```

Responsibilities:

- Estimates arrival rate.
- Computes current wait-time baseline.
- Computes forecasted wait times.
- Tracks recent count history.

Algorithms:

```text
arrival rate estimation
M/M/c queueing approximation
linear trend projection
Holt double exponential smoothing
mean-reversion forecast
```

### 6.4 Ticket Service

File:

```text
app/services/ticket_service.py
```

Responsibilities:

- Runs background ticket worker.
- Receives ticket generation jobs.
- Calls ticket printer.
- Attaches short code and PDF path to active queue person.
- Saves ticket record to MySQL.

### 6.5 Ticket Printer

File:

```text
app/services/ticket_printer.py
```

Responsibilities:

- Generates short code.
- Generates JWT token.
- Builds QR URL.
- Generates PDF ticket.
- Deletes tickets during reset or completion.
- Validates ticket short code and JWT.

## 7. Persistence Layer

The persistence layer stores permanent and semi-permanent data.

### 7.1 MySQL Database

File:

```text
app/database/database_handler.py
```

Responsibilities:

- Creates database connection pool.
- Saves ticket records.
- Updates queue record status.
- Supports ticket validation.
- Checks database availability.

Stored data:

```text
users
queue_records
queue_number
short_code
jwt_token
pdf_path
status
expires_at
created_at
served_at
```

Not stored in MySQL:

```text
raw camera frames
live JPEG snapshots
temporary crowd state
active queue tracker memory
```

## 8. Cache / Runtime State Layer

### 8.1 In-Memory State

File:

```text
app/state.py
```

Responsibilities:

- Stores latest detector state.
- Stores latest snapshot.
- Stores recent history.
- Provides helper functions for dashboard/API routes.

In-memory data:

```text
latest_state
latest_snapshot
latest_snapshot_seq
history
```

### 8.2 Optional Redis Cache

File:

```text
app/services/cache_service.py
```

Purpose:

- Helps cloud deployment share temporary live data.
- Keeps latest snapshot and latest crowd state available across backend workers.
- Falls back to memory if Redis is not configured.

Redis keys:

```text
queueflow:state:latest
queueflow:snapshot:latest
queueflow:snapshot:seq
queueflow:history:recent
```

Redis stores:

```text
latest snapshot
snapshot sequence number
latest crowd state
recent crowd history
```

Redis does not store:

```text
permanent ticket records
users
long-term history
PDF files
```

## 9. Security Architecture

### 9.1 Staff Dashboard Auth

File:

```text
app/core/security.py
```

Used by:

```text
pages.py
auth.py
queue.py
crowd.py
```

Responsibilities:

- Authenticates staff login.
- Creates session tokens.
- Checks staff-only routes.

### 9.2 Detector Authentication

Detector calls are protected by a camera token.

Environment variable:

```env
CAM_TOKEN=detector-secret-token
```

Detector sends:

```text
X-CAM-TOKEN
```

Backend verifies it before accepting:

```text
POST /yolo/push-frame
POST /yolo/update
```

### 9.3 Ticket Security

Ticket access is protected by:

```text
short_code
JWT token stored in MySQL
expiration time
queue number match
```

Mobile QR status lookup requires:

```text
queue_number
short_code token
```

## 10. Local Deployment Architecture

Local testing setup:

```text
Laptop
  -> FastAPI backend on port 5000
  -> Detector process
  -> MySQL database
  -> Ticket PDFs in app/tickets

Phone
  -> Mobile app
  -> Same Wi-Fi network as laptop
  -> Calls laptop IP address
```

Local URL example:

```text
http://192.168.43.236:5000
```

Local environment:

```env
PORTAL_BASE_URL=http://192.168.43.236:5000
API_BASE_URL=http://localhost:5000
REDIS_URL=
```

Architecture diagram:

```text
Phone Mobile App
      |
      | Wi-Fi LAN
      v
Laptop FastAPI Backend
      |
      +--> MySQL
      +--> In-memory queue tracker
      +--> app/tickets PDFs
      ^
      |
Detector Process + Camera
```

## 11. Cloud Deployment Architecture

Cloud setup:

```text
Mobile App
Dashboard Browser
      |
      | HTTPS
      v
Cloud FastAPI Backend
      |
      +--> Cloud MySQL / Managed MySQL
      +--> Redis Cache
      +--> Ticket PDF storage

Detector Process
      |
      | HTTPS
      v
Cloud FastAPI Backend
```

Cloud environment:

```env
PORTAL_BASE_URL=https://your-domain.com
API_BASE_URL=https://your-domain.com
REDIS_URL=redis://default:password@your-redis-host:6379/0
JWT_SECRET_KEY=replace-with-fixed-secret
SECRET_KEY=replace-with-fixed-secret
CAM_TOKEN=replace-with-fixed-camera-token
```

Important cloud note:

For the thesis prototype, use one backend instance first. If multiple backend
instances are used, the active queue tracker must be moved fully to Redis or
MySQL so all instances share the same queue state.

## 12. Component Interaction Diagram

```text
                 +----------------------+
                 |      Mobile App      |
                 |  QR Scan + Status UI |
                 +----------+-----------+
                            |
                            | GET /api/queue/status
                            v
+------------+     +--------+---------+      +----------------+
|  Detector  | --> | FastAPI Backend  | ---> | MySQL Database |
| YOLO + CV  |     | API + Dashboard  |      | Ticket Records |
+-----+------+     +--------+---------+      +----------------+
      |                     |
      | POST /yolo/update   |
      | POST /yolo/push     |
      v                     v
+------------+     +------------------+
|  Camera    |     | Redis, optional  |
|  Frames    |     | Live State Cache |
+------------+     +------------------+
                            ^
                            |
                 +----------+-----------+
                 |   Dashboard Browser  |
                 | Analytics + Snapshot |
                 +----------------------+
```

## 13. Request Flow Examples

### 13.1 New Queue Person

```text
Camera detects person
  -> detector.py processes frame
  -> POST /yolo/push-frame
  -> queue_service filters detection
  -> queue_tracker confirms new person
  -> ticket_service queues ticket job
  -> ticket_printer creates PDF and QR
  -> database_handler saves ticket record
```

### 13.2 Mobile Queue Status

```text
User scans QR
  -> mobile extracts q and token
  -> GET /api/queue/status
  -> queue.py validates short code
  -> MySQL validates JWT record
  -> queue_tracker checks active queue
  -> API returns live queue status
```

### 13.3 Dashboard Analytics

```text
Browser opens analytics page
  -> queue_analytics.html loads
  -> fetch /api/queue/analytics every 2.5 seconds
  -> queue_service builds analytics
  -> frontend renders cards, bars, tables, histogram
```

### 13.4 Live Snapshot

```text
Detector draws overlay
  -> encodes frame as JPEG
  -> POST /yolo/update
  -> state.set_snapshot
  -> optional Redis cache
  -> dashboard reads /api/snapshot or /api/crowd/video
```

## 14. Architecture Summary

QueueFlow uses a separated architecture:

```text
Detector process:
  camera, YOLO, snapshots, metadata upload

FastAPI backend:
  routes, auth, queue logic, analytics, dashboard, mobile API

Business services:
  queue tracking, prediction, tickets, database persistence

MySQL:
  permanent ticket and user records

Redis:
  optional cloud cache for temporary live data

Mobile app:
  QR scanning and user queue status

Web dashboard:
  staff monitoring, analytics, live video/snapshot
```

This separation keeps computer vision processing independent from API request
handling, while still allowing the backend to use YOLO results to create queue
numbers, tickets, dashboard analytics, and mobile queue status updates.
