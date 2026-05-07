# QueueFlow System Dataflow Documentation

## 1. Purpose

This document explains how data moves through the QueueFlow system.
It describes where data comes from, what component processes it, where it is
stored, and how it reaches the dashboard and mobile queue tracker.

QueueFlow has two major dataflows:

1. Queue and ticket dataflow
2. Live snapshot and dashboard dataflow

## 2. High-Level Dataflow

```text
Camera
  -> Detector
  -> FastAPI Backend
  -> Queue Service
  -> Queue Tracker
  -> Ticket Service
  -> Ticket PDF + QR Code
  -> MySQL Database
  -> Mobile Queue Tracker
```

For dashboard video:

```text
Camera
  -> Detector
  -> Annotated JPEG Snapshot
  -> FastAPI Backend
  -> Memory / Redis Cache
  -> Dashboard
```

## 3. Camera to Detector Dataflow

Source:

```text
Camera device
```

Main file:

```text
app/detector.py
```

The detector reads frames from the camera using OpenCV.

Input data:

```text
raw camera frame
```

Processing:

```text
raw camera frame
  -> resize / prepare frame
  -> YOLO person detection
  -> object tracking
  -> queue overlay drawing
  -> JPEG snapshot encoding
```

Output data:

```text
person count
bounding boxes
track IDs
confidence scores
density values
arrival rate
estimated wait time
forecast values
annotated JPEG snapshot
```

## 4. Detector to Backend Dataflow

The detector sends two types of data to the FastAPI backend.

### 4.1 Detection Metadata

Endpoint:

```text
POST /yolo/push-frame
```

Backend file:

```text
app/routers/detector_api.py
```

Data sent:

```json
{
  "count": 3,
  "avg_density": 0.4,
  "max_density": 0.8,
  "queue_length": 3,
  "estimated_wait_time": 6.0,
  "arrival_rate": 1.2,
  "system_utilization": 0.6,
  "predicted_wait_5min": 8.0,
  "predicted_wait_15min": 10.0,
  "predicted_wait_30min": 12.0,
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

Backend processing:

```text
/yolo/push-frame
  -> state.update_from_detector_payload()
  -> queue_service.process_tracked_persons()
  -> queue_tracker.process_frame()
```

### 4.2 Annotated Snapshot

Endpoint:

```text
POST /yolo/update
```

Backend file:

```text
app/routers/detector_api.py
```

Data sent:

```text
JPEG image bytes
```

Backend processing:

```text
/yolo/update
  -> state.set_snapshot()
  -> memory latest_snapshot
  -> Redis cache, if configured
```

The snapshot is temporary live data. It is not stored in MySQL.

## 5. Backend State Dataflow

Main file:

```text
app/state.py
```

The backend keeps the latest live data in memory.

Memory data:

```text
latest_state
latest_snapshot
latest_snapshot_seq
history
```

`latest_state` contains:

```text
count
avg_density
max_density
timestamp
queue_length
active_counters
estimated_wait_time
arrival_rate
system_utilization
predicted_wait_5min
predicted_wait_15min
predicted_wait_30min
queue_state
```

If Redis is configured, the same live data is mirrored to Redis.

Redis keys:

```text
queueflow:state:latest
queueflow:snapshot:latest
queueflow:snapshot:seq
queueflow:history:recent
```

## 6. Queue Service Dataflow

Main file:

```text
app/services/queue_service.py
```

Input:

```text
tracked_persons from detector
```

Processing:

```text
filter weak detections
remove tiny bounding boxes
remap unstable track IDs
send cleaned detections to queue tracker
```

Output:

```text
cleaned tracked people
updated queue state
ticket generation trigger for new people
```

The queue service acts as the bridge between YOLO detections and queue business
logic.

## 7. Queue Tracker Dataflow

Main file:

```text
app/services/queue_tracker.py
```

Input:

```text
cleaned tracked people
bounding boxes
track IDs
confidence scores
queue zone
appearance signatures
```

Processing:

```text
check if person is inside queue zone
identify new people
match returning people
track missing people
handle no-show countdowns
calculate queue positions
maintain active queue
```

Output:

```text
active_queue
completed_queue
noshow_alerts
queue_count
next_number
total_served
```

Each active queue person has:

```text
queue_number
queue_label
track_id
status
position_in_line
wait_time
wait_time_seconds
joined_at
bbox
access_token or short_code
```

## 8. New Person to Ticket Dataflow

When the queue tracker confirms a new person:

```text
queue_tracker
  -> queue_service._on_new_person()
  -> ticket_service.enqueue_ticket()
```

Main files:

```text
app/services/ticket_service.py
app/services/ticket_printer.py
```

Ticket service input:

```text
queue_number
position
estimated_wait_time
```

Ticket printer output:

```text
short_code
jwt_token
expires_at
pdf_path
QR code
PDF ticket
```

Ticket generation flow:

```text
new queue person
  -> enqueue ticket job
  -> background ticket worker
  -> generate short code
  -> generate JWT token
  -> build QR URL
  -> generate ticket PDF
  -> attach short code to active queue person
  -> save ticket record to MySQL
```

## 9. QR Code Dataflow

The QR code stores a URL.

Format:

```text
{PORTAL_BASE_URL}/api/queue/status?q={queue_number}&token={short_code}
```

Local example:

```text
http://192.168.43.236:5000/api/queue/status?q=1&token=EMZR-D5V2
```

Cloud example:

```text
https://your-domain.com/api/queue/status?q=1&token=EMZR-D5V2
```

QR data:

```text
backend URL
queue number
short code token
```

Important:

```text
Old tickets keep the old QR URL.
After changing PORTAL_BASE_URL, generate a new ticket.
```

## 10. Ticket to Database Dataflow

Main file:

```text
app/database/database_handler.py
```

Function:

```text
save_ticket_record()
```

Data saved to MySQL:

```text
queue_number
short_code
jwt_token
pdf_path
status
expires_at
created_at
served_at
```

The database is used for:

```text
ticket validation
token validation
ticket history
served/no-show status persistence
```

The database is not used for:

```text
live camera snapshots
raw video frames
temporary dashboard image data
```

## 11. Mobile App Dataflow

The mobile app scans the QR code from the ticket.

Input:

```text
QR code URL
```

Mobile processing:

```text
scan QR
  -> extract queue number
  -> extract token
  -> extract backend URL
  -> call backend queue status API
```

API endpoint:

```text
GET /api/queue/status?q=1&token=EMZR-D5V2
```

Backend file:

```text
app/routers/queue.py
```

Backend validation:

```text
read short code from request
  -> query MySQL queue_records
  -> validate JWT token
  -> check active queue tracker
  -> return queue status
```

Data returned to mobile:

```text
queue_number
queue_label
status
position_in_line
wait_time
wait_time_seconds
joined_at
noshow_warning
prediction
```

## 12. Dashboard Analytics Dataflow

Frontend file:

```text
app/templates/queue_analytics.html
```

API endpoint:

```text
GET /api/queue/analytics
```

Backend file:

```text
app/services/queue_service.py
```

Function:

```text
build_queue_analytics()
```

Input data:

```text
queue_tracker.active_queue
queue_tracker.completed_queue
state.crowd_prediction_fields()
prediction values
noshow alerts
```

Output data:

```text
overview
wait_times
throughput
live_crowd
new_arrival
forecast
charts
active_queue
recent_completed
noshow_alerts
recommendation
```

Frontend renders:

```text
summary cards
forecast bars
status breakdown
wait band histogram
active queue table
recent completed table
no-show alerts
```

## 13. Histogram / Bar Chart Dataflow

The analytics histogram is generated from backend queue analytics.

Frontend:

```text
queue_analytics.html
  -> fetch("/api/queue/analytics")
  -> renderBars()
```

Backend:

```text
queue_service.build_queue_analytics()
  -> charts.wait_bands
```

Wait band function:

```text
_wait_band_counts(active_queue)
```

Bands:

```text
0-2 min
2-5 min
5-10 min
10+ min
```

Flow:

```text
active_queue wait_time_seconds
  -> group by wait bands
  -> return counts in charts.wait_bands
  -> frontend renderBars()
  -> histogram display
```

## 14. Snapshot / Live Video Dataflow

Detector:

```text
camera frame
  -> draw boxes and labels
  -> encode JPEG
  -> POST /yolo/update
```

Backend:

```text
detector_api.py
  -> state.set_snapshot()
  -> latest_snapshot memory
  -> Redis snapshot cache, if configured
```

Dashboard endpoints:

```text
GET /api/snapshot
GET /api/crowd/video
GET /video
```

Data returned:

```text
latest annotated JPEG image
```

## 15. Prediction Dataflow

Main file:

```text
app/services/prediction_service.py
```

Input:

```text
recent person counts
timestamps
average service time
number of active counters
current queue length
```

Processing:

```text
arrival rate estimation
M/M/c queueing model
linear trend projection
Holt double exponential smoothing
longer-term mean-reversion forecast
```

Output:

```text
estimated_wait_time
arrival_rate
system_utilization
predicted_wait_5min
predicted_wait_15min
predicted_wait_30min
```

These values are sent from the detector to the backend through:

```text
POST /yolo/push-frame
```

Then exposed through:

```text
GET /api/queue/prediction
GET /api/queue/status
GET /api/queue/analytics
```

## 16. Storage Summary

### MySQL

Stores permanent and semi-permanent records:

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

### Memory

Stores live runtime state:

```text
active queue tracker
latest crowd state
latest snapshot
recent history
```

### Redis Cache

Optional for cloud deployment.

Stores temporary shared live data:

```text
latest snapshot
snapshot sequence number
latest crowd state
recent crowd history
```

### Mobile App

Stores current session data while the app is running:

```text
queue number
access token
backend URL from QR
latest queue snapshot
```

## 17. Complete End-to-End Dataflow

```text
Camera
  ↓
detector.py
  ↓
YOLO detection and tracking
  ↓
POST /yolo/push-frame
  ↓
detector_api.py
  ↓
state.py latest_state
  ↓
queue_service.py
  ↓
queue_tracker.py
  ↓
active_queue
  ↓
new person confirmed
  ↓
ticket_service.py
  ↓
ticket_printer.py
  ↓
PDF ticket with QR code
  ↓
database_handler.py
  ↓
MySQL queue_records
  ↓
mobile app scans QR
  ↓
GET /api/queue/status
  ↓
MySQL token validation
  ↓
active queue lookup
  ↓
mobile displays queue status
```

Dashboard dataflow:

```text
Camera
  ↓
detector.py
  ↓
annotated JPEG snapshot
  ↓
POST /yolo/update
  ↓
state.latest_snapshot
  ↓
Redis cache, optional
  ↓
/api/snapshot or /api/crowd/video
  ↓
dashboard live display
```

Analytics dataflow:

```text
queue_tracker active/completed data
  ↓
queue_service.build_queue_analytics()
  ↓
GET /api/queue/analytics
  ↓
queue_analytics.html
  ↓
cards, tables, histogram, and bars
```

## 18. Simple Summary

```text
Camera data becomes detection metadata and live snapshots.
Detection metadata becomes queue state.
Queue state creates ticket records and QR codes.
QR codes let mobile users request their live queue status.
Snapshots feed the live dashboard.
Analytics are calculated from queue tracker state and prediction data.
MySQL stores tickets and user data.
Redis optionally stores temporary live data for cloud deployment.
```
