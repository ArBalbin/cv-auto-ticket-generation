# QueuEx Current Progress

Last updated: September 12, 2026

This file summarizes implementation status for advisers, panelists, and
developers: what works and is verified, what is measured, what is demo-only,
and what is still outstanding.

## Project status

QueuEx is a working end-to-end prototype. A student enrols once through the
mobile app, taps **Join the Queue** on a later visit, walks into the queue
area, and the camera recognises them and issues a queue number automatically —
no button, no staff input. Staff mark the number done from a web dashboard.

This full path has been walked end to end on real hardware: join from the
phone → detection → face match → ticket issued → marked done on the staff
dashboard, with no intervention.

### What changed since May 2026

The original design had the system read a number off a printed kiosk ticket
using a custom digit CNN, then validate the holder's face. The official SOP was
**reissued after the pre-oral defense**, and its item #2 requires that
"YOLO-based AI assigns queue numbers without manual input." The kiosk-reading
path was therefore removed and the system now mints numbers itself for
recognised students.

Consequences, all complete:

- The digit CNN and its OCR pipeline were **deleted**, not disabled.
- The HSV appearance-signature re-identification chain — appearance signatures,
  twin guard, done blacklist, spatial fallback — was **replaced entirely** by
  ArcFace face matching.
- A student mobile app and a staff web dashboard became separate repositories;
  this backend now serves JSON only and has no HTML pages.

## Completed components

### Face recognition — the core of the system

`app/services/face_service.py`, `app/routers/students.py`

- Student self-registration via institutional Google (Gbox) sign-in, restricted
  by domain and verified against Google's `sub` claim rather than email
- Guided face capture in the mobile app with pose and lighting prompts
- 512-dimension ArcFace embeddings (InsightFace `buffalo_s`, ONNX Runtime CPU);
  **embeddings are stored, photographs are not**
- Two-part match decision rule: accept only when the best match clears an
  absolute similarity threshold **and** beats the runner-up by a margin;
  anything ambiguous is refused and escalated to staff
- **Join the Queue consent gate** — recognition alone issues nothing; the
  student must have opted in for this visit
- Face-based re-entry matching, replacing the retired appearance chain

### Person detection and tracking

`app/detector.py`, `app/services/queue_tracker.py`

- OpenCV capture with camera-backend scanning and retry handling
- YOLOv8n + ByteTrack, stock COCO weights (see "Model training" below)
- Queue-zone membership, candidate confirmation, static-object rejection,
  bounding-box smoothing, duplicate suppression, track-ID remapping
- Face-embedding bypass for the static-motion test, so a person standing
  perfectly still is not rejected as an object
- Supervised worker threads that report fatal errors instead of dying silently
- No-show countdown, staff done/reset, counter adjustment

### Ticket generation

`app/services/ticket_service.py`, `app/services/ticket_printer.py`

- Background ticket worker with failure isolation — a ticket error can no
  longer kill the worker thread
- Plain white PDF ticket carrying the queue number, the recognised student's
  name, a QR code and a short access code
- JWT generation and validation; short-code status authentication
- QR target address auto-detected from the machine's LAN address, so a printed
  ticket is scannable from a phone without editing configuration

### Backend API

`app/main.py`, `app/routers/`

JSON only — 50 routes across auth, students, queue, crowd, detector and health.
Interactive documentation at `/docs`. Error handlers cover every router.

### Prediction

`app/services/prediction_service.py`

M/M/c baseline, short-term trend projection, Holt's double exponential
smoothing, growth-ratio mean reversion; current, 5-, 15- and 30-minute
horizons. **A separately-trained wait-time model is not yet integrated** — this
module is the interim implementation.

### Database

Eight tables: `users`, `student_profiles`, `queue_records`, `queue_events`,
`counter_config_history`, `face_match_events`, `recognition_metrics`,
`crowd_snapshots`. The last three exist specifically so accuracy can be
reported from real operation rather than estimated. See
`DATABASE_RELATIONSHIPS_DOCUMENTATION.md`.

### Cloud readiness

`Dockerfile`, `Procfile`, `render.yaml`, `.env.production.example`. Optional
Redis cache and S3-compatible ticket storage. `APP_ENV=production` validates
critical values and refuses to start on a loopback portal URL. The detector
stays local because it needs direct camera access.

## Measured results

Full detail and reproduction steps in `ACCURACY.md` Part A.

| Measurement | Result |
|---|---|
| Face recognition ROC AUC | 0.999907 |
| Equal Error Rate | 0.42 % |
| Rank-1 identification (45 probes, 17 identities) | 100 % |
| Students linked automatically | 44/45 (97.8 %) |
| **Wrong identities issued** | **0** |
| **Unenrolled strangers refused** | **45/45 (100 %)** |
| API, 50 concurrent users, 2 min | p50 12 ms, p99 85 ms, **0 failures** |
| Face extraction latency | ~258 ms |
| Match scan, 300 enrolled students | 2.93 ms |

The strongest single result: at the deployed thresholds, the score rule **on
its own** would have produced 5 false identifications of strangers. The
two-part rule with the margin check produces **0**. That rule is the project's
original contribution, and this is measured evidence it does real work.

## Model training — what to say when asked

Neither model was trained by this project, and the documentation says so
plainly rather than implying otherwise.

- **YOLOv8n** uses stock COCO-pretrained weights. A fine-tune was attempted on
  a custom dataset; person detection collapsed afterwards and the change was
  reverted. The stock model is more accurate here.
- **ArcFace** (`buffalo_s`) is pretrained and used unmodified. Fine-tuning a
  face-recognition backbone on 17 people would degrade it, not improve it.

What *was* built and measured is the **decision layer**: the threshold and
margin rule, calibrated against real photographs and validated with an
open-set test. See `FACE_RECOGNITION_CALIBRATION.md` and `ACCURACY.md`.

## Known limitations

- **Person-detection accuracy on the deployment camera is not yet measured.**
  The harness is built and verified; ground-truth labelling is outstanding.
- Building that harness surfaced an unresolved risk: on a saved test frame,
  YOLO detected a person at 0.93 confidence and the `MAX_BBOX_FRAC = 0.70`
  filter discarded the box for covering 72 % of the frame. A student standing
  close to the camera may therefore be dropped entirely. **Verify against the
  real camera placement before the beta.**
- Accuracy figures come from cooperative enrolment-style photos. Live probes
  are YOLO crops at queue-zone distance, which is harder — treat the published
  figures as an upper bound and report the live numbers alongside them.
- Thermal printer hardware is not wired up; PDF is the demonstration output.
- The wait-time model is the interim M/M/c implementation.
- Queue-zone coordinates must be calibrated per camera placement.
- Liveness detection is not implemented — a photograph held to the camera could
  be recognised. Mitigated by the Join the Queue gate (an attacker also needs
  the victim's phone and account), not eliminated.

## Outstanding work

| # | Item | Needed by |
|---|---|---|
| 1 | Commit the working tree — four months of work is uncommitted | Immediately |
| 2 | Verify `MAX_BBOX_FRAC` against the real camera placement | Before beta |
| 3 | Satisfaction survey instrument for SOP #4 | Before beta, Sept 15 |
| 4 | YOLO ground-truth labelling, then `evaluate_yolo_accuracy.py score` | Before defense |
| 5 | Rewrite the three `SYSTEM_*` documents — still describe the May 2026 system | Before defense |
| 6 | Add pivot-era entries to `PITFALLS.md` | Before defense |
| 7 | Run the beta and report live accuracy via `report_recognition_metrics.py` | Sept 15 |

## Documentation map

| File | Contents | Status |
|---|---|---|
| `README.md` | Overview, setup, run instructions | Current |
| `ALGORITHMS.md` | The nine algorithms and deployed thresholds | Current |
| `ACCURACY.md` | Measured (Part A) and estimated (Part B) | Current |
| `FACE_RECOGNITION_CALIBRATION.md` | How thresholds were chosen | Current |
| `DATABASE_RELATIONSHIPS_DOCUMENTATION.md` | All eight tables | Current |
| `SYSTEM_ARCHITECTURE_DOCUMENTATION.md` | Component architecture | **Stale — May 2026** |
| `SYSTEM_FLOW_DOCUMENTATION.md` | Runtime flow | **Stale — May 2026** |
| `SYSTEM_DATAFLOW_DOCUMENTATION.md` | Data movement | **Stale — May 2026** |
| `PITFALLS.md` | Real bugs and design errors | Missing pivot-era entries |
| `CLOUD_DEPLOYMENT_CHECKLIST.md` | Deployment checklist | Needs review |
| `CLOUD_CACHE_SETUP.md` | Redis cache setup | Needs review |
