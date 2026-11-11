# QueueFlow System Architecture Documentation

## Current Progress Snapshot

Last updated: November 11, 2026

QueueFlow (now shown to students as QueuEx) has four parts:

- the QueuEx app (Flutter) on the student's phone;
- the detector (`app/detector.py`, YOLOv8 + InsightFace) on the office laptop
  that has the camera;
- the staff dashboard and display board (React, served by Vercel);
- the FastAPI backend on Render (0.5 CPU, 512 MB), with Aiven MySQL, Upstash
  Redis, Supabase Storage for ticket PDFs, and Google for Gbox sign-in.

A student asks for a number in the app and fills the payment slip; the number
is assigned when the camera sees them in line and their face matches. Only the
512-number face embedding leaves the laptop, never a face image. Staff call
numbers to their own window with Call next, or switch to automatic calling.
Sections 1 and 12 hold the current diagrams; some later sections still
describe the May 2026 build.

## 1. Architecture Overview

QueueFlow is designed as a modular computer-vision queue monitoring system.
It separates camera detection, backend API services, queue business logic,
ticket generation, data persistence, live cache, dashboard pages, and mobile
queue tracking.

High-level architecture (November 2026):

```text
Student's phone                    Cashier office (NCF campus)
  QueuEx app (Flutter)               Camera -> Detector (laptop: YOLOv8 + InsightFace)
        |                            Staff dashboard (React, Vercel)
        |                            Display board (/queue-display)
        |                                   |
        +-----------------+-----------------+
                          | HTTPS
                          v
             FastAPI backend (Render, 0.5 CPU, 512 MB)
                          |
                          +--> Sign-in & join (Gbox or face, consent, payment slip)
                          +--> Queue tracker + face match (1:N embedding index)
                          +--> Calling & windows (Call next, or automatic)
                          +--> Status & wait estimate (queue engine)
                          +--> Wait log & prediction accuracy
                          +--> Ticket worker (PDF ticket)
                          +--> Retention scheduler (archive, then delete)
                          |
                          +--> Aiven MySQL      (accounts, embeddings, queue records)
                          +--> Upstash Redis    (sessions, latest frame, crowd stats)
                          +--> Supabase Storage (ticket PDFs)
                          +--> Google           (verifies the Gbox sign-in)

The detector also serves the dashboard's live view directly at 127.0.0.1:8765.
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
app/services/wait_model.py
```

Responsibilities:

- Estimates arrival rate.
- Computes the current wait from the trained model.
- Computes forecasted wait times.
- Tracks recent count history.

Algorithms:

```text
arrival rate estimation
median regression on people ahead (trained, see WAIT_TIME_MODEL.md)
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
- Must be configured as volatile/in-memory only for live camera snapshots.

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

Persistence rule:

```text
If Redis is self-hosted, use redis.conf with:
save ""
appendonly no
```

If a managed Redis provider is used, RDB/AOF persistence and provider backups
should be disabled for the QueueFlow cache instance, or disclosed in deployment
and manuscript notes.

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
                         +--------------------------+
                         |   QueuEx app (Flutter)   |
                         | join, status, Window N   |
                         +----+---------------^-----+
   POST /api/students/me/join |               | GET /api/students/me/queue
   (payment slip)             v               | (every 2 s)
+------------------+     +----+---------------+-----+     +------------------+
| Detector         |---->|  FastAPI backend (Render)|<--->| Aiven MySQL      |
| YOLOv8 tracking  | (1) |                          |     +------------------+
| InsightFace      |<----|  sign-in & join          |<--->| Upstash Redis    |
| embeddings       | (2) |  queue tracker + face    |     +------------------+
+--+-----------+---+     |    match (1:N index)     |---->| Supabase Storage |
   ^           |         |  calling & windows       |     +------------------+
   | frames    |         |  wait estimate + log     |<--->| Google sign-in   |
+--+-----+     | (5)     |  ticket worker           |     +------------------+
| Camera |     |         |  retention scheduler     |
+--------+     v         +----+---------------+-----+
     +---------------------+  ^               | (6)
     | Staff dashboard     |<-+ (3) (4)       v
     | (React, Vercel)     |            +---------------+
     +---------------------+            | Display board |
                                        +---------------+

(1) POST /yolo/push-frame: tracked people + face embeddings, 2-3 a second.
    POST /yolo/update: the annotated frame for the backup camera stream.
    Only the 512-number embedding leaves the laptop, never a face image.
(2) Reply to each push: the line state and who to clear.
(3) POST /api/queue/call_next, /done, /no_show; the staff account's window
    claim is renewed every 30 s.
(4) GET /api/queue/list (the line and each payment slip), prediction
    accuracy, service records CSV; GET /api/crowd/video as the backup stream.
(5) Live view straight from the detector: http://127.0.0.1:8765/video.
(6) GET /api/queue/display: number, place in line, window. No names.
```

## 13. Request Flow Examples

### 13.1 New Queue Number

```text
Student taps Get a Queue Number and fills the payment slip
  -> POST /api/students/me/join (payment_items)
  -> queue_tracker.arm_join_intent holds the request and the slip
Student walks to the camera at the line
  -> detector.py tracks the person; InsightFace makes the face embedding
  -> POST /yolo/push-frame
  -> queue_tracker confirms the person (1 s in line, 3 face frames)
  -> embedding_index matches the face to the student who asked (1:N)
  -> the queue number is linked to the student, with the payment slip
  -> ticket_service builds the PDF ticket and uploads it to Supabase
  -> database_handler saves the queue record
```

### 13.2 Mobile Queue Status and Calling

```text
QueuEx app polls GET /api/students/me/queue every 2 s while in line
  -> number, people ahead, wait estimate, Now serving
Staff press Call next (POST /api/queue/call_next)
  -> the next number is called to that staff account's window
  -> the app shows "It's your turn" with the window number
  -> the dashboard shows the payment slip lines and TOTAL
Staff press Done or No-show
  -> the wait log saves the estimate, the real wait and the window
  -> the app shows the Done screen, or Skipped after a No-show
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
  -> Upstash Redis cache
  -> dashboard shows the detector's local view (127.0.0.1:8765/video),
     or /api/crowd/video with the staff token when that is not reachable
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

Redis (Upstash):
  sessions, latest camera frame, crowd stats

Mobile app (QueuEx):
  sign-in, Get a Queue Number with the payment slip, queue status, your turn

Web dashboard:
  Call next per window, payment slip, analytics, prediction accuracy, live view
```

This separation keeps computer vision processing independent from API request
handling, while still allowing the backend to use YOLO results to create queue
numbers, tickets, dashboard analytics, and mobile queue status updates.
