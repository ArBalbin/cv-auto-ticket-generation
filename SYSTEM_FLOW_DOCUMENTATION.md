# QueueFlow System Flow Documentation

## 1. System Overview

QueueFlow is a computer-vision-based queue monitoring and ticket tracking system.
It uses a camera and YOLO object detection to identify people in a queue, assigns
queue numbers automatically, generates ticket PDFs with QR codes, and allows
users to check their queue status from a mobile app.

The system has five main parts:

1. FastAPI backend
2. YOLO detector process
3. Queue tracker and ticket worker
4. MySQL database
5. Optional Redis cache for cloud deployment

High-level flow:

```text
Camera
  -> YOLO detector
  -> FastAPI backend
  -> Queue tracker
  -> Ticket generator
  -> MySQL ticket record
  -> QR code
  -> Mobile queue tracker
```

For live dashboard video:

```text
Camera
  -> YOLO detector
  -> Annotated JPEG snapshot
  -> FastAPI backend
  -> Memory and optional Redis cache
  -> Dashboard live snapshot/video
```

## 2. Main Components

### 2.1 FastAPI Backend

Main file:

```text
app/main.py
```

Responsibilities:

- Starts the FastAPI application.
- Loads environment variables.
- Warms up the database connection pool.
- Registers all API routers.
- Serves dashboard pages.
- Receives detector updates.
- Provides queue status APIs for the mobile app.
- Provides live snapshot/video APIs for the dashboard.

Registered routers:

```text
app/routers/pages.py          Dashboard pages, login pages, video stream
app/routers/auth.py           Login/logout API
app/routers/detector_api.py   Detector upload endpoints
app/routers/crowd.py          Crowd stats, snapshot, history
app/routers/queue.py          Queue status, queue list, staff queue actions
app/routers/health.py         Health check endpoint
```

The backend runs on:

```text
http://localhost:5000
```

In cloud, it should run behind a public HTTPS URL:

```text
https://your-domain.com
```

## 3. Detector Flow

Main file:

```text
app/detector.py
```

The detector is a separate process from the FastAPI backend. It opens the
camera, loads the YOLO model, detects people, creates queue metadata, and
sends results to the backend.

Detector startup flow:

```text
Load .env
  -> Load OpenCV
  -> Load HTTP client
  -> Load YOLO model
  -> Open camera
  -> Start YOLO worker thread
  -> Start snapshot uploader thread
  -> Start camera capture loop
```

Important environment variables:

```env
API_BASE_URL=http://localhost:5000
CAM_TOKEN=detector-secret-token
CAM_WIDTH=640
CAM_HEIGHT=480
YOLO_IMGSZ=320
YOLO_EVERY=8
SNAPSHOT_FPS=12
```

### 3.1 Camera Frame Processing

For every camera frame:

```text
Read camera frame
  -> Prepare frame
  -> Send selected frames to YOLO worker
  -> Reuse latest YOLO results for overlay
  -> Draw queue boxes and HUD
  -> Encode annotated frame as JPEG
```

YOLO output contains:

```text
person bounding boxes
track IDs
confidence scores
person count
```

The detector also calculates:

```text
average density
maximum density
estimated wait time
arrival rate
system utilization
queue length estimate
```

## 4. Detector to Backend Communication

The detector sends two different kinds of data to the backend.

### 4.1 Metadata Push

Endpoint:

```text
POST /yolo/push-frame
```

Handled by:

```text
app/routers/detector_api.py
```

Purpose:

- Sends crowd count.
- Sends detected people.
- Sends bounding boxes.
- Sends track IDs.
- Sends prediction values.
- Lets the backend update queue state.

Example payload:

```json
{
  "count": 3,
  "avg_density": 0.4,
  "max_density": 0.8,
  "queue_length": 3,
  "estimated_wait_time": 6.0,
  "yolo_frame_idx": 20,
  "tracked_persons": [
    {
      "track_id": 12,
      "bbox": [120, 50, 230, 390],
      "conf": 0.72,
      "appearance": []
    }
  ]
}
```

After the backend receives this payload:

```text
detector_api.py
  -> state.update_from_detector_payload()
  -> queue_service.process_tracked_persons()
  -> queue_tracker.process_frame()
  -> returns queue state to detector
```

### 4.2 Snapshot Push

Endpoint:

```text
POST /yolo/update
```

Handled by:

```text
app/routers/detector_api.py
```

Purpose:

- Sends the latest annotated JPEG image.
- Used for dashboard live video and snapshot display.
- Not saved to MySQL.

Flow:

```text
Camera frame
  -> Draw overlay
  -> Encode JPEG bytes
  -> POST /yolo/update
  -> state.set_snapshot()
  -> Memory and optional Redis cache
```

## 5. Queue Service Flow

Main file:

```text
app/services/queue_service.py
```

The queue service is the bridge between raw YOLO detection data and queue
business logic.

Responsibilities:

- Filters weak detections.
- Removes bounding boxes that are too small.
- Remaps unstable track IDs.
- Passes cleaned detections to the queue tracker.
- Triggers ticket creation when a new person is confirmed.
- Updates database status when someone is served or marked no-show.

Flow:

```text
Raw YOLO tracked people
  -> Filter by confidence and bbox size
  -> Remap track IDs if needed
  -> Send to queue_tracker.process_frame()
  -> Receive updated queue state
```

## 6. Queue Tracker Flow

Main file:

```text
app/services/queue_tracker.py
```

The queue tracker is the main queue logic component. It keeps the active queue
in process memory.

It decides:

```text
Is the detected person inside the queue zone?
Is this a new person?
Is this the same person as before?
Did someone leave the frame?
Did someone return?
Is someone missing?
Should someone be marked no-show?
Who is next in line?
```

Each active queue person contains:

```text
queue_number
track_id
bbox
status
position_in_line
entered_at
last_seen
wait_time
short_code
pdf_path
```

Common statuses:

```text
waiting
missing
done_pending
served
no_show
```

### 6.1 New Person Confirmation

When a detected person remains valid long enough:

```text
Person enters queue zone
  -> Tracker confirms person
  -> Assigns queue number Q001, Q002, Q003, ...
  -> Calls queue_service new-person callback
  -> Enqueues ticket generation
```

## 7. Ticket Generation Flow

Main files:

```text
app/services/ticket_service.py
app/services/ticket_printer.py
```

The ticket service runs a background worker so ticket generation does not block
the detector or API request handling.

Ticket flow:

```text
New queue person
  -> ticket_service.enqueue_ticket()
  -> background ticket worker
  -> ticket_printer.issue_ticket()
  -> generate JWT token
  -> generate short code
  -> generate QR code URL
  -> generate PDF ticket
  -> save ticket record to MySQL
  -> attach short code and PDF path to active queue person
```

Ticket data includes:

```text
queue number
short code
JWT token
PDF path
expiry time
```

## 8. QR Code Flow

The QR code is created in:

```text
app/services/ticket_printer.py
```

QR URL format:

```text
{PORTAL_BASE_URL}/api/queue/status?q={queue_number}&token={short_code}
```

Local example:

```text
http://192.168.43.236:5000/api/queue/status?q=4&token=ABCD-EFGH
```

Cloud example:

```text
https://your-domain.com/api/queue/status?q=4&token=ABCD-EFGH
```

Important:

- The QR code stores the URL at ticket generation time.
- Old tickets keep the old URL.
- After changing `PORTAL_BASE_URL`, new tickets must be generated.

## 9. Mobile Queue Tracker Flow

The mobile app scans the QR code printed on the ticket.

Mobile flow:

```text
Open mobile app
  -> Scan ticket QR
  -> Extract queue number
  -> Extract token
  -> Extract backend host from QR URL
  -> Call /api/queue/status
  -> Display live queue status
```

The mobile app calls:

```text
GET /api/queue/status?q=4&token=ABCD-EFGH
```

Handled by:

```text
app/routers/queue.py
```

The backend validates:

```text
Does the short code exist in MySQL?
Does the queue number match the short code?
Is the JWT token valid?
Is the JWT token not expired?
Is the queue number still active?
Does the token match the active queue person?
```

If valid, the backend returns:

```text
queue number
queue label
status
position in line
wait time
prediction
no-show warning
```

## 10. Dashboard Flow

Dashboard pages are served by:

```text
app/routers/pages.py
```

The dashboard can access:

```text
/dashboard/computer-vision
/dashboard/queueflow
/dashboard/queue-analytics
/dashboard/profile
```

Live image endpoints:

```text
GET /api/snapshot
GET /api/crowd/video
GET /video
```

Crowd and queue data endpoints:

```text
GET /api/stats
GET /api/crowd/data
GET /api/history
GET /api/queue/list
GET /api/queue/prediction
GET /api/queue/analytics
```

Staff actions:

```text
POST /api/queue/done
POST /api/queue/reset
POST /api/queue/zone
POST /api/queue/adjust_counters
POST /api/queue/noshow_config
```

## 11. Database Role

Database module:

```text
app/database/database_handler.py
```

MySQL stores permanent records such as:

```text
users
ticket records
queue numbers
short codes
JWT tokens
PDF ticket paths
ticket expiry time
served/no-show status updates
```

The database does not store the live camera snapshot image.

Reason:

- Live snapshots update many times per second.
- Storing every image in MySQL would make the system slower.
- The system only needs the latest image for live display.

## 12. Redis Cache Role for Cloud

Redis is optional locally, but recommended in cloud.

Redis stores temporary live data:

```text
queueflow:snapshot:latest
queueflow:snapshot:seq
queueflow:state:latest
queueflow:history:recent
```

Configured in:

```text
app/services/cache_service.py
app/state.py
```

Cloud environment variables:

```env
REDIS_URL=redis://default:password@your-redis-host:6379/0
CACHE_KEY_PREFIX=queueflow
CACHE_STATE_TTL_SECONDS=30
CACHE_SNAPSHOT_TTL_SECONDS=10
CACHE_HISTORY_TTL_SECONDS=3600
```

Redis is used for:

```text
latest live JPEG snapshot
latest crowd state
recent crowd history
latest prediction values
```

Redis is not used for:

```text
permanent ticket records
permanent user records
long-term queue history
```

Those belong in MySQL.

## 13. Local Deployment Flow

Local setup:

```text
FastAPI backend on laptop
Detector process on same laptop
MySQL on local machine
Mobile app connects through laptop Wi-Fi IP
Redis optional and disabled by default
```

Important local environment variables:

```env
PORTAL_BASE_URL=http://192.168.43.236:5000
API_BASE_URL=http://localhost:5000
REDIS_URL=
```

Local QR flow:

```text
Ticket QR points to laptop Wi-Fi IP
Mobile phone must be on same network
Backend must listen on 0.0.0.0:5000
Windows Firewall must allow port 5000
```

## 14. Cloud Deployment Flow

Cloud setup:

```text
FastAPI backend deployed to cloud
MySQL database hosted locally or cloud
Redis hosted as managed cache
Detector sends data to cloud API URL
Mobile app scans QR with public HTTPS URL
Dashboard reads live data through public backend URL
```

Important cloud environment variables:

```env
PORTAL_BASE_URL=https://your-domain.com
API_BASE_URL=https://your-domain.com
REDIS_URL=redis://default:password@your-redis-host:6379/0
JWT_SECRET_KEY=replace-with-one-fixed-long-secret
SECRET_KEY=replace-with-one-fixed-long-secret
```

Cloud QR flow:

```text
Ticket QR points to public HTTPS domain
Mobile app scans QR
Mobile app calls cloud backend
Backend validates token through MySQL
Backend returns live queue status
```

Important cloud note:

For the thesis prototype, run one backend instance first. Redis now shares
latest snapshot and state, but the active queue tracker is still mainly held in
process memory. If multiple backend instances are used, active queue tracker
state must be moved fully into Redis or MySQL.

## 15. Health Check Flow

Health endpoint:

```text
GET /health
```

Handled by:

```text
app/routers/health.py
```

Example response:

```json
{
  "status": "ok",
  "db": true,
  "cache": {
    "configured": true,
    "available": true
  },
  "snapshot": true,
  "timestamp": "2026-05-06T22:30:00"
}
```

Meaning:

```text
status      Backend is running
db          MySQL is reachable
cache       Redis configuration and connection status
snapshot    Latest camera snapshot exists
timestamp   Server time
```

## 16. End-to-End Sequence

Complete queue-ticket-mobile flow:

```text
1. Backend starts.
2. Detector starts.
3. Detector reads camera frame.
4. YOLO detects people.
5. Detector sends metadata to /yolo/push-frame.
6. Backend updates latest crowd state.
7. Queue service filters detections.
8. Queue tracker confirms person inside queue zone.
9. Queue tracker assigns queue number.
10. Ticket service receives ticket job.
11. Ticket printer creates PDF ticket with QR.
12. Ticket record is saved to MySQL.
13. User scans QR with mobile app.
14. Mobile app calls /api/queue/status.
15. Backend validates token through MySQL.
16. Backend checks active queue tracker.
17. Backend returns queue status.
18. Mobile app displays position and wait time.
```

Complete dashboard snapshot flow:

```text
1. Detector reads camera frame.
2. Detector draws boxes and queue overlay.
3. Detector encodes frame as JPEG.
4. Detector sends JPEG to /yolo/update.
5. Backend stores latest JPEG in memory.
6. If Redis is configured, backend also stores it in Redis.
7. Dashboard requests /api/snapshot or /api/crowd/video.
8. Backend returns latest JPEG snapshot.
9. Dashboard displays live annotated video.
```

## 17. Summary

QueueFlow works by separating permanent data from live data.

Permanent data:

```text
MySQL
  -> users
  -> ticket records
  -> short codes
  -> JWT tokens
  -> queue status history
```

Temporary live data:

```text
Memory and optional Redis
  -> latest snapshot
  -> latest crowd state
  -> latest prediction state
  -> recent history
```

Main system idea:

```text
YOLO detects people.
Queue tracker converts detections into queue numbers.
Ticket service prints secure QR tickets.
Mobile app scans QR to check status.
Dashboard displays live snapshots and analytics.
Redis helps share live data in cloud.
MySQL keeps permanent ticket and user records.
```
