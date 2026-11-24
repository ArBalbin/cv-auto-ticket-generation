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

**Algorithm count: 16 → 4 major algorithms.** The manuscript (Section 3.2)
groups the sections below the same way:

| Manuscript | Sections here | |
|---|---|---|
| **Algorithm 1** Person detection and tracking | 1 and 3 | major |
| **Algorithm 2** Presence confirmation | 2 | major |
| **Algorithm 3** Face identity validation and number issuance | 4 and 5 | major |
| **Algorithm 4** Wait-time estimation | 6 and 7 | major |
| Procedure A/B — calling to a window (Call next, or automatic) | 8 | supporting rule |
| Procedure C — ticket generation | System features | supporting rule |

Calling, ticket generation (JWT/QR/PDF) and TTS announcements apply fixed
rules; none performs detection, recognition or prediction.

**Gradient Boosting is not deployed.** A Gradient Boosting predictor was
trained on the historical cashier log and evaluated offline (MAE 2.35 min),
but nothing in `app/` loads it: simpler models were as accurate or better on
the data, its runtime calibration was never built, and scikit-learn would use
about 100 MB of the 512 MB server. The manuscript states this in its scope and
delimitation.

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
| `QUEUE_MIN_CONFIRM_SECONDS` | 1.0 | With a face seen: time in the zone that confirms presence instead |
| `QUEUE_MIN_CONFIRM_FACE_FRAMES` | 3 | With a face seen: fewest frames for the time rule |
| `QUEUE_MIN_MOTION_PIXELS` | 7 | Static-object rejection threshold |
| `QUEUE_STATIC_CONF_BYPASS` | 0.70 | Confidence that skips the motion test |
| `QUEUE_MAX_MISSING_FRAMES` | 240 | Frames absent before removal |
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

The frame count assumed the tracker received about 14 updates a second, so it
meant roughly one second. With the backend in the cloud, the detector reaches
it 2–3 times a second (each update waits for the server's reply), and the same
count took 5–7 seconds. So a candidate whose face the detector has captured is
also confirmed after `QUEUE_MIN_CONFIRM_SECONDS` (1 s) over at least
`QUEUE_MIN_CONFIRM_FACE_FRAMES` (3) frames, which restores the intended one
second. A candidate without a face still needs the full count, so the
static-object check below keeps enough positions to judge it.

**A student who asked for a number is confirmed by their face** (since 25 Nov
2026). When a new person's face, in one update, is a registered student who
tapped Get a Queue Number and holds no number, they are confirmed and issued
the number on that update (`queue_service.recognised_newcomers`). At about
one update a second, the one-second-and-three-updates wait had put three
seconds or more before the face was even checked. Not for a face that two new
boxes match in the same update, nor for a box the backend has already held as
someone else, nor for a student the camera already knows on another box:
those reads may have caught a neighbour's face.

A new box that overlaps someone already in line is treated as a possible
second box on that person until a face tells them apart. A face already
captured for the newcomer now counts on the updates that carry none; before,
a person standing close beside a student with a number was thrown out on
every update without a fresh face read and could go a minute unconfirmed.

**After someone is served** the same correction applies:

- They count as gone once out of view for 1 second over at least 3 updates. This was 10 updates, which became 4–5 seconds.
- For 10 seconds after being served, nobody new is confirmed where they stood. This stops their own box from being taken for someone new as they walk away. It was 150 updates, which became 50–75 seconds and kept a student who stepped away and rejoined from getting a new number for about a minute.

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

Since 25 Nov 2026 it is limited by what the camera has seen
(`services/track_sightings.py`), because a third person stepping in front of
someone in line was taken for that hidden person:

- Two numbers seen in the same update are two people and are never merged.
- Without a face, a new box is merged only if it is in the same place and of
  the same size as the person was (IoU ≥ 0.5, neither box half as big again)
  and they vanished within 8 updates. The 8 was set at about 15 updates a
  second; at the cloud backend's one a second it had let anyone within
  180 px over 8 s take a missing person's number.
- With a face, the face must match and the boxes be of similar size. A face
  seen not to match rules that pair out for good.
- A merged box counts as settled for the detector's face budget only after
  two of its face reads were taken as that person's.

**A number moves to a new box** (a student back in view on a new track) only
when the reads on that box have settled on them: at least
`MIN_READS_TO_MOVE_A_NUMBER` (3) name them and more name them than not. Never
onto a box seen together with them, nor onto one that came into view the
moment they vanished and is clearly bigger or smaller, which is someone who
stepped in front of them. A single read had handed a hidden student's number
to the person in front.

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

A student submits one to five photographs at varied angle and lighting
(`register_my_face`). Each yields an embedding; those that produce a
confident face are averaged and L2-normalised into one canonical
embedding:

```
canonical = mean(e₁ … eₙ) / ‖mean(e₁ … eₙ)‖
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
accepted = (best_score ≥ 0.30) AND (best_score − second_best ≥ 0.15)
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
previous one done — preventing a recognized student from
accumulating duplicate numbers by re-entering the camera's view.

Walk-ins never reach this path: an unenrolled face produces no accepted
match, so they are added by staff via `force_new_person()` using the
number printed on their kiosk ticket.

---


## 6. Wait-Time Prediction

**Where:** `app/services/queue_engine.py` (default), `app/services/wait_model.py`
(fallback), `queue_service.estimate_wait_for_position`

**By default a real-time queue engine plays the live line forward**, again on
every request. Each window is free, free when its current transaction is
expected to end, or closed until a set time (lunch). A transaction that has run
*e* minutes is expected to need R(e) = μ · m(e/μ) more, the mean residual life of
a lognormal (σ = 0.425) fitted to the October cashier log; μ is the live average
of the last 20 services (section 7). People waiting are taken in queue order by
whichever window frees first, μ each; when a person is taken is their estimated
time of service, sent to the app in seconds. With one window,
wait = R(e) + people ahead × μ. Formulas, worked example and validation:
`WAIT_TIME_MODEL.md`. `WAIT_ESTIMATOR=regression` switches back to
the regression below.

As the benchmark and fallback, a **median regression trained on observed
cashier transactions** gives the wait for someone with a given number of people
ahead:

```
wait = 1.1214 + 0.1445 × people_ahead × (3 / active_counters)     minutes
```

Trained by `ML/train_wait_model.py` on 528 transactions over 14 days, and stored
as plain coefficients in `Model/wait_time_model.json`, so neither the server nor
the detector needs scikit-learn. On unseen days its MAE is **3.00 minutes**
(RMSE 5.90, R² −0.04, MAPE 72.2%), the same as a constant 2-minute prediction
but responsive to the queue. See `WAIT_TIME_REGRESSION.md` for the validation and
limitations, especially that long lines are under-estimated.

**Each student** gets the model's estimate for their own position, with
`people_ahead = position − 1`.

**Forecasts at 5, 15, and 30 minutes** project the camera's queue count forward,
then apply the model to the projected count:

| Horizon | Projection of the count |
|---|---|
| 5 min | Linear trend over the last 20 samples |
| 15 min | Holt's double exponential smoothing, damped |
| 30 min | Growth ratio with mean reversion, capped at 60 min |

The count history holds one sample every 10 seconds, covering 10 minutes. Only
the mapping from count to wait is trained; the projections are not validated.

The arrival rate λ and utilisation ρ = λ / (c · μ) are still computed from the
same history for the dashboard, but no longer drive the wait.

**Every ticket logs its own check.** When a number is issued, the ticket row
stores the people ahead, the counters open and the estimate shown; when staff
mark it done, it stores when the person was called to a counter. Comparing the
two gives the live accuracy, and the same log re-fits the model on exact counts
(`WAIT_TIME_MODEL.md`, section 9; `WAIT_TIME_REGRESSION.md`, section 6).

---

## 7. Live Average Service Time

**Where:** `queue_engine.current_average()`, `queue_service.record_service_time()`

Every service staff mark done is timed from Call next to Done and counted at
once. The average the wait estimate uses is

```
μ = ( k·μ₀ + ΣSᵢ + Σeⱼ ) / ( k + n + L )
```

with μ₀ = 1.87 min (`AVG_SERVICE_TIME`, the October mean) counted as k = 10
services, Sᵢ the last n ≤ 20 services finished in the past two hours (5 s to
30 min), and eⱼ the L students still at a window who have already taken longer
than the average, at their time so far. With nothing timed it is the usual
pace; it then follows the cashier's real pace, and a window running long raises
it at once. Students and staff see it as "Average service time". It replaced
the inter-departure gap measurement (a 70/30 blend refreshed every five minutes),
which could not react within a transaction.

---

## 8. Calling to a Window

**Where:** `queue_tracker.call_next()` (default), `queue_tracker._recalculate_positions()` (automatic mode)

**By default staff call numbers** (`QUEUE_CALL_MODE=staff`). Each staff
account holds one window (a claim renewed by the dashboard, lapsing after
120 s). Pressing **Call next** gives the first waiting number to that window,
unless the window is closed or still serving someone. A switch on the
dashboard changes to automatic calling, which uses the sticky rule below.

**Automatic mode — sticky assignment.** People within counter range (position ≤ active counter count) are assigned
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

**No-show** — `queue_tracker.mark_no_show()`, reached from the dashboard.
When the student whose turn it is has not come, a staff member marks them
a no-show and the system bumps the number and moves the queue on. There is
no timer and no detection behind it: whether somebody has turned up is a
judgement made by the person at the desk who can see the queue area, and
the earlier automatic version of this was removed for that reason. A
no-show is recorded separately from a served ticket and does not count
towards `total_served`, since nobody was served.

---

## Summary

| # | Algorithm | Technique | Refuses when |
|---|---|---|---|
| 1 | Person detection & tracking | YOLOv8n + ByteTrack, 5 gates, EMA | Fails any gate |
| 2 | Presence confirmation | Centroid-in-zone + frame buffer and motion test, or 1 s with 3 face frames | Too few frames, or static |
| 3 | Track stability | IoU/centroid remap + dedup | — |
| 4 | Face recognition | ArcFace 512-d + threshold & margin rule, in-memory index | Score < 0.30 or margin < 0.15 |
| 5 | System-minted number | Reserved 5000+ range, one-per-student, only after a request | Student already holds an entry, or did not ask |
| 6 | Wait-time prediction | Queue simulation per window (default); median regression on people ahead (fallback) | — |
| 7 | Service-time measurement | Live average of timed services, shrunk toward the October mean | — |
| 8 | Calling to a window | Call next per staff window (default); sticky assignment (automatic mode) | Window busy or closed |

Four of the eight (1, 2, 4, 5) have an explicit refuse-and-escalate
branch. That is the design stance: the system is built to hand ambiguous
cases to a human rather than resolve them by guessing.
