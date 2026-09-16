# QueuEx — Core Algorithm Reference

QueuEx is an enhancement layer over Naga College Foundation's existing
queue arrangement, not a replacement for it. The camera watches the queue
area, recognizes students who have registered their face, and issues their
queue number automatically. Walk-ins are unaffected: they take a printed
kiosk ticket exactly as before, and staff enter that number.

The guiding rule throughout is **refuse rather than guess**. Every decision
below has an explicit "not confident enough" branch that escalates to a
human instead of inventing an answer, because in a queue a wrong identity
is far more damaging than a slow one.

---

## What changed from the pre-oral design

This document was rewritten after the panel's revisions. Three structural
changes matter when comparing against the earlier version:

1. **Identity is now face recognition, not colour histograms.** The old
   design tracked people by a split-body HSV appearance signature, with
   separate algorithms for comparison, re-identification, and a
   twin/lookalike guard. All four are retired and replaced by one
   face-embedding match (Algorithm 4). Colour histograms cannot tell two
   students in the same uniform apart — the exact failure case a queue at
   a college produces constantly.

2. **Queue numbers for registered students are issued by the system**
   (Algorithm 5), per SOP #2. The kiosk's own numbering for walk-ins is
   untouched.

3. **Ticket-number OCR was removed.** An earlier revision included a
   custom-trained digit classifier to read printed kiosk tickets. Once
   registered students stopped needing kiosk tickets, that code path became
   unreachable and was deleted rather than left running for no purpose.

**Algorithm count: 16 → 9.** Ticket generation (JWT/QR/PDF) and TTS
announcements are documented as *system features* rather than algorithms —
neither performs CV or predictive reasoning; both are deterministic output
formatting triggered by decisions made elsewhere.

---

## Effective parameters

Values are read from `app/core/config.py` and overridden by `.env`. The
column below is what the deployed system actually runs with; class-level
defaults inside `QueueTracker` are overwritten at startup by
`queue_service.wire_callbacks()` and should not be read as authoritative.

| Parameter | Value | Governs |
|---|---|---|
| `FRAME_SCALE` | 0.75 | Frame downscale before YOLO |
| `YOLO_IMGSZ` | 320 | YOLO inference resolution |
| `YOLO_EVERY` | 3 | Run YOLO every Nth camera frame |
| `YOLO_CONF` | 0.55 | Minimum person-detection confidence |
| `MIN_BBOX_AREA` | 800 px² | Reject specks |
| `MAX_BBOX_FRAC` | 0.85 | Reject boxes covering most of the frame |
| `QUEUE_MIN_PORTRAIT_ASPECT` | 0.60 | Reject non-person-shaped boxes |
| `QUEUE_MIN_CONFIRM_FRAMES` | 20 | Frames before presence is confirmed |
| `QUEUE_MIN_MOTION_PIXELS` | 8 | Static-object rejection threshold |
| `QUEUE_STATIC_CONF_BYPASS` | 0.70 | Confidence that skips the motion test |
| `QUEUE_MAX_MISSING_FRAMES` | 240 | Frames absent before removal |
| `QUEUE_NOSHOW_WINDOW_SECONDS` | 300 | No-show countdown |
| `FACE_MODEL_PACK` | `buffalo_s` | InsightFace model pack |
| `FACE_MATCH_THRESHOLD` | **0.30** | Minimum cosine similarity to accept |
| `FACE_MARGIN_THRESHOLD` | **0.15** | Required lead over the runner-up |
| `FACE_MIN_DETECT_CONF` | 0.60 | Minimum face-detector confidence |
| `FACE_ONLY_QUEUE_NUMBER_START` | 5000 | Start of the system-minted range |
| `PENDING_LINK_TIMEOUT_SECONDS` | 45 | Before staff are alerted |

The two face thresholds are **calibrated against real photographs**, not
chosen by intuition — see `FACE_RECOGNITION_CALIBRATION.md` for the
methodology and results, and `ACCURACY.md` for the full biometric
evaluation (FAR/FRR, EER, ROC-AUC, and the open-set rejection test that
set the margin at 0.15).

---

## 1. Person Detection and Tracking — YOLOv8n + ByteTrack

**Where:** `app/detector.py`

YOLOv8n (pretrained, `Model/yolov8n.pt`) detects people class-0 in each
sampled frame; ByteTrack assigns a persistent `track_id` across frames so
the same person keeps one identity while visible.

Every detection passes five gates before it is accepted as a person:

| Gate | Rejects |
|---|---|
| Confidence ≥ `YOLO_CONF` | Weak detections |
| Area ≥ `MIN_BBOX_AREA` | Distant specks, noise |
| Area ≤ `MAX_BBOX_FRAC` of frame | Boxes swallowing the whole scene |
| Aspect ratio h/w ≥ `QUEUE_MIN_PORTRAIT_ASPECT` | Non-person shapes |
| Non-maximum suppression | The same person detected twice |

Box coordinates are then smoothed by an exponential moving average
(α = 0.45) so the overlay does not jitter frame to frame:

```
smoothed = α · new + (1 − α) · previous
```

Running YOLO every 3rd frame rather than every frame is a deliberate
trade: at ~12.6 fps capture this yields ~4.2 detections/sec, which is
comfortably faster than a person walks into position, while leaving CPU
budget for face embedding (Algorithm 4), the far more expensive stage.

---

## 2. Zone Presence and Candidate Confirmation

**Where:** `queue_tracker.process_frame()`

A detection inside the configured queue rectangle is not yet a person in
the queue — it is a *candidate*. Promotion to confirmed presence requires
surviving three tests.

**Zone membership** uses the box centroid, not overlap, so someone merely
brushing the zone edge is not counted:

```
inside = zone.x1 ≤ (x1+x2)/2 ≤ zone.x2  AND  zone.y1 ≤ (y1+y2)/2 ≤ zone.y2
```

**Frame accumulation:** the candidate must be seen for
`QUEUE_MIN_CONFIRM_FRAMES` (20) frames. This is the anti-ghost filter — a
person walking past the camera, or a one-frame false positive, never
reaches 20.

**Static-object rejection:** a candidate whose bounding-box centre moves
less than `QUEUE_MIN_MOTION_PIXELS` (8 px) across the accumulation window
is rejected as a poster, reflection, or photograph rather than a live
person. Two escape hatches exist, because the naive version of this test
punishes exactly the behaviour the system wants:

- Detection confidence ≥ 0.70 bypasses the test outright.
- **A successfully extracted face embedding bypasses it.** A student
  standing deliberately still to be recognized is the *intended* use case,
  and would otherwise be repeatedly rejected as a static object. A live
  face detection is stronger evidence of a real person than pixel jitter.

Passing all three yields a `QueuePerson` with `identity_status =
'pending_link'` — confirmed present, no queue number yet. **This state
never mints a number.** That happens only in Algorithm 5, or by staff
entry for walk-ins.

---

## 3. Track Stability — Remapping and Deduplication

**Where:** `queue_tracker._find_overlapping_candidate()`, `_dedup_active_queue()`

ByteTrack IDs are not stable across occlusion: a person briefly hidden may
return with a new `track_id`, producing a phantom second entry. Two
mechanisms defend against this.

**Remapping** merges a new track into an existing one when they plainly
describe the same body — IoU ≥ 0.10 or centroid distance < 180 px, within
45 frames of absence.

**Deduplication** scans the active queue for entries overlapping at
IoU ≥ 0.10 or centre distance < 0.50 × box diagonal and drops the younger
one. Newly restored entries carry 60 frames of *dedup immunity*, so a
person who just returned is not immediately deleted as a duplicate of
themselves.

---

## 4. Face Recognition Identity Validation

**Where:** `app/services/face_service.py`

This replaces the retired HSV appearance signature, appearance comparison,
re-identification, and twin-guard algorithms — four algorithms collapsed
into one.

### The model

**InsightFace `buffalo_s`, on ONNX Runtime (CPU).** Two models participate:

- **SCRFD** (`det_500m.onnx`) — locates the face
- **ArcFace** (`w600k_mbf.onnx`) — converts it to a **512-dimensional
  embedding**

Both are **pretrained**. This is a deliberate choice, not a shortcut:
training an ArcFace-quality embedding requires millions of images across
tens of thousands of identities, a dataset neither obtainable nor
necessary for this project. The scientific contribution here is not the
embedding — it is the decision rule built on top of it, and its
calibration against real data.

### Library, not a cloud service

InsightFace is an **open-source Python library**, installed as an ordinary
package dependency. It is not a commercial face-recognition API such as
AWS Rekognition, Azure Face, or Face++, and this distinction has three
consequences that matter for this deployment:

- **It runs entirely on the local machine.** No internet connection, no API
  keys, no per-request billing, and no dependence on a vendor's uptime
  during a queue rush.
- **No face data ever leaves the premises.** Frames are processed in the
  detector process and discarded; only the derived 512-value embedding is
  stored. Nothing is transmitted to a third party. Under the Data Privacy
  Act, where biometrics are sensitive personal information, keeping the
  entire pipeline on-site removes a whole category of disclosure risk that
  a cloud API would introduce.
- **The model files are fixed and inspectable.** The pack downloads once to
  `~/.insightface/models/buffalo_s/` and does not change underneath the
  system, so results stay reproducible — a cloud provider silently
  upgrading its model could invalidate a calibration overnight.

### Two face libraries, two different jobs

The system uses face-related libraries in two places, and they are
frequently confused. Only one performs recognition:

| Component | Library | Role |
|---|---|---|
| Backend (Python) | **InsightFace** | Recognition — *who is this person?* |
| Mobile app (Flutter) | **Google ML Kit** | Capture guidance only — *is a face centred, angled correctly, well lit?* |

Google ML Kit never identifies anyone and never sees the enrolled roster.
It exists solely to drive the guided auto-capture on the registration
screen (`lib/screens/face_capture_screen.dart`): it reports the head yaw
angle and face size so the app knows when to take each of the three
enrollment shots automatically, and it measures frame brightness to prompt
"too dark" or "too bright." The resulting photographs are uploaded to the
backend, where InsightFace performs the only identity-bearing computation
in the system.

### Extraction

The upper **75%** of the YOLO person box is cropped and passed to SCRFD.
The fraction is deliberately generous. An earlier value of 0.45 assumed a
full-body standing person, but a student stepping close to the camera to
be recognized produces a head-and-shoulders box, and 0.45 then sliced
through the middle of their face. Measured on a real failing frame:

| Crop fraction | Detector confidence | Outcome |
|---|---|---|
| 0.45 | 0.572 | **Rejected** (below 0.60) |
| 0.55 | 0.663 | Accepted |
| 0.65 | 0.696 | Accepted |
| **0.75** | **0.722** | Accepted |
| 1.00 | 0.719 | Accepted |

Latency was flat (~240–290 ms) at every crop size, because InsightFace
resizes its input to 320×320 internally. A tighter crop buys no speed; it
only risks cutting the face in half.

### Enrollment

A student registers three guided poses (centre, and two opposite turns).
Each yields an embedding; the three are averaged and L2-normalised into
one canonical embedding:

```
canonical = mean(e₁, e₂, e₃) / ‖mean(e₁, e₂, e₃)‖
```

Averaging across angles makes the stored signature more robust than any
single photograph.

**Only the embedding is stored — never the photographs.** Under the Data
Privacy Act biometrics are sensitive personal information; the system
keeps a non-reversible numeric derivative and discards the images.

### Matching — the decision rule

A live embedding is compared against every enrolled student by cosine
similarity:

```
similarity(a, b) = (a · b) / (‖a‖ ‖b‖)
```

Acceptance requires **both** conditions:

```
accepted = (best_score ≥ 0.30) AND (best_score − second_best ≥ 0.10)
```

The second condition is the important one and is original to this system.
If two enrolled students both score around 0.5, the best match is not
meaningfully better than the runner-up, so the system **refuses to choose**
and leaves the person pending for staff resolution. This is the structural
replacement for the old twin-guard: rather than a special case for
lookalikes, ambiguity is rejected by construction.

A rejection is not an error. It is the safe outcome — the person stays
pending and a human resolves it.

### Cost

| Stage | Measured |
|---|---|
| Embedding extraction (live path) | **258 ms** mean, p95 308 ms |
| Match scan, 50 enrolled | 0.76 ms |
| Match scan, 100 enrolled | 0.96 ms |
| Match scan, 300 enrolled | 2.9 ms |

Extraction dominates matching by roughly 100×. The brute-force O(N) scan
is therefore not a scaling concern: growing from 50 to 300 enrolled
students adds about 2 ms. Recognition speed is bounded by how often a
frame yields a usable face, not by roster size.

Reproduce with `load_testing/bench_face_pipeline.py`.

---

## 5. System-Minted Queue Number

**Where:** `queue_tracker.mint_face_only_number()`

This is the algorithm that answers SOP #2 — identifying a student and
issuing their queue number with no manual input.

When Algorithm 4 accepts a match for a person in `pending_link` state, the
system issues a queue number directly:

```
number = next unused value ≥ FACE_ONLY_QUEUE_NUMBER_START (5000)
```

The separate numeric range is what makes this safe to run alongside the
kiosk. Kiosk tickets occupy ordinary low numbers; system-minted numbers
start at 5000. The two sequences can never collide, so the kiosk's
numbering is untouched — satisfying the panel's instruction to leave the
existing queue flow alone while still allowing automatic issuance for
registered students.

**One active entry per student.** Before minting, the system checks whether
that student already holds a pending or waiting entry. If so, nothing is
issued. A student can only obtain a new number after staff mark the
previous one served or no-show — preventing a recognized student from
accumulating duplicate numbers by re-entering the camera's view.

Walk-ins never reach this path: an unenrolled face produces no accepted
match, so they are added by staff via `force_new_person()` using the
number printed on their kiosk ticket.

---

## 6. No-Show Detection

**Where:** `queue_tracker._check_noshow()`

A person at the front of the queue who disappears from the camera starts a
countdown of `QUEUE_NOSHOW_WINDOW_SECONDS` (300 s). Staff see a live
warning and may bump them immediately or let the timer expire.

Two guards prevent false no-shows:

- Entries still in `pending_link` (no number yet) are excluded — there is
  nothing to no-show.
- Automatic bumping is **disabled by default**
  (`QUEUE_AUTO_NOSHOW_ENABLED = false`); the timer raises an alert and a
  human decides. Removing someone from a queue is not a decision the
  system makes unsupervised.

---

## 7. Wait-Time Prediction

**Where:** `app/services/prediction_service.py`

> **Status:** the trained predictive model for this objective is being
> developed as separate work. What ships in this system today is the
> analytical queueing model described here. Report accuracy against
> whichever is in place at evaluation time.

An **M/M/c queueing model** estimates waiting time from arrival rate λ,
service rate μ, and counter count c:

```
ρ = λ / (c · μ)                    system utilisation
W = queue_length / (c · μ)         expected wait
```

Arrival rate is estimated from the recent count history (up to 30 samples):

```
λ = (count_now − count_then) / minutes_elapsed
```

Forecasts at 5, 15, and 30 minutes extend this with a linear trend
projection over recent queue-length samples.

**No accuracy measurement exists yet.** MAE and MAPE require comparing
predictions against realised waits, which needs data from live operation.

---

## 8. Dynamic Service-Time Measurement

**Where:** `app/services/queue_service.py`

Rather than trusting a configured average, a background thread re-measures
the real average service time from completed records every 5 minutes and
blends it into the running estimate:

```
avg_service_time = 0.7 · measured + 0.3 · previous
```

The 70/30 blend is damping: a single unusually slow transaction should
nudge the estimate, not redefine it. This keeps the wait-time model honest
when counters are genuinely faster or slower than configured. The dashboard
labels the figure "Measured from DB" or "Default (.env)" so staff can see
which is in use.

---

## 9. Sticky Counter Assignment

**Where:** `queue_tracker._recalculate_positions()`

People within counter range (position ≤ active counter count) are assigned
to whichever counters are currently free. Assignment is **sticky**: the
loop skips anyone who already holds a counter, so once a person is given
counter 2 they keep counter 2 as the queue shifts behind them.

Without stickiness, a person's displayed counter could change between
reading it and walking over — a small implementation detail with a large
effect on whether the display can be trusted. Each newly assigned number
is announced once, tracked in a set so it is never called twice.

---

## System features (not algorithms)

**Ticket generation** — `app/services/ticket_printer.py`. A JWT (HS256,
4-hour expiry) is embedded in a QR code; a PDF is rendered with the queue
number, the recognized student's ID and name, position, and a short access
code with ambiguous characters (0/O, 1/I) removed. Deterministic output
formatting; no decision logic.

**TTS announcements** — Web Speech API in the browser display board. Reads
newly assigned numbers aloud. No CV, no prediction.

---

## Summary

| # | Algorithm | Technique | Refuses when |
|---|---|---|---|
| 1 | Person detection & tracking | YOLOv8n + ByteTrack, 5 gates, EMA | Fails any gate |
| 2 | Presence confirmation | Centroid-in-zone + 20-frame buffer + motion test | Too few frames, or static |
| 3 | Track stability | IoU/centroid remap + dedup | — |
| 4 | Face recognition | ArcFace 512-d + threshold & margin rule | Score < 0.30 or margin < 0.10 |
| 5 | System-minted number | Reserved 5000+ range, one-per-student | Student already holds an entry |
| 6 | No-show detection | Missing-frame countdown | Entry still pending |
| 7 | Wait-time prediction | M/M/c + trend forecast | — |
| 8 | Service-time measurement | Inter-departure gap blending | — |
| 9 | Counter assignment | Sorted position, sticky | — |

Five of the nine (1, 2, 4, 5, 6) have an explicit refuse-and-escalate
branch. That is the design stance: the system is built to hand ambiguous
cases to a human rather than resolve them by guessing.
