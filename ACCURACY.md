# QueueFlow Algorithm Accuracy Reference

Last updated: September 8, 2026

> ### Read this before quoting any number from this document
>
> This file contains **two kinds of numbers, and they must not be mixed.**
>
> **Part A — Measured.** Results from experiments that were actually run, on
> real data, with a named script that reproduces them. These are the numbers
> for the evaluation chapter and the defense.
>
> **Part B — Estimated.** Sections 1–16 below, written in May 2026. They
> describe *expected* behaviour derived from published figures and prototype
> observation, not from experiments on this system. The column header says
> "Expected Accuracy" for that reason. **Do not present them as results.**
>
> Part B also predates the face-recognition pivot, so it is stale in places:
> it has no face-recognition section at all, and its §8 "Done Blacklist" has
> since been removed from the system entirely.

---

# PART A — MEASURED RESULTS

## A1. Face recognition — biometric accuracy

**Script:** `ML/evaluate_face_accuracy.py` → `ML/face_accuracy_report.json`
**Figures:** `ML/figures/06_far_frr.png`, `07_roc_curve.png`, `08_full_score_distribution.png`
**Dataset:** 17 enrolled identities, 45 held-out probe photos, 765 comparisons
(45 genuine + 720 impostor)

### Threshold-independent quality

| Metric | Value | Meaning |
|---|---|---|
| ROC AUC | **0.999907** | Probability a genuine pair outscores an impostor pair |
| Equal Error Rate | **0.42 %** (at 0.320) | Where false-accept and false-reject rates cross |
| d′ (separation) | **4.99** | Gap between the two distributions, in standard deviations |
| Genuine score | 0.695 ± 0.140 | Same person |
| Impostor score | 0.108 ± 0.090 | Different people |

AUC is quoted to six decimals deliberately. At four it prints `1.0000`, which
reads as perfect separation and would be an overclaim — there is one inverted
pair in the set.

### Identification accuracy at the deployed operating point

Thresholds: `FACE_MATCH_THRESHOLD = 0.30`, `FACE_MARGIN_THRESHOLD = 0.15`.

| Outcome | Result | Why it matters |
|---|---|---|
| Rank-1 accuracy | **100 %** (45/45) | The correct student always ranked first |
| Correctly linked | **97.8 %** (44/45) | Ticket issued automatically |
| Wrong identity | **0 %** (0/45) | The unsafe failure — none occurred |
| Refused to staff | 2.2 % (1/45) | The safe failure — system declined to guess |

The one refusal scored 0.321 with a margin of 0.027: a genuine student the
system could not confidently distinguish, so it escalated instead of guessing.
That is the intended behaviour, not an error.

### Open-set rejection — the safety test

Closed-set accuracy cannot answer the question that actually matters here:
*does an unenrolled person walking past the camera get handed someone else's
queue number?* Every probe in a closed-set test comes from someone already
enrolled, so the case is never presented.

This test removes each person from the gallery entirely, then probes with
their photos — they are now a stranger.

| Result | Value |
|---|---|
| Stranger probes | 45 |
| Correctly refused | **45/45 (100 %)** |
| False identifications | **0 (0.00 %)** |
| Highest stranger score | 0.353 |

### Why the margin rule exists — the strongest result in this evaluation

The highest stranger score, **0.353, is above the 0.30 match threshold.** The
score rule alone would have accepted it. Five stranger probes cleared the
score threshold; the margin rule caught all five:

| Stranger | Would have been accepted as | Score | Margin |
|---|---|---|---|
| archie | Malate | 0.353 | 0.046 |
| rparcero | Nayawan | 0.331 | 0.110 |
| rparcero | Nayawan | 0.326 | 0.056 |
| sanota | Malate | 0.305 | 0.012 |
| sanota | Saycon | 0.302 | 0.015 |

> **Score rule alone → 5 false identifications.
> Score + margin rule → 0.**

This is the measured justification for the decision rule that is this
project's original contribution. The pretrained ArcFace model is not the
contribution; the two-part rule built on top of it is, and this table is the
evidence that it does real work.

### How the margin threshold was set

The margin was **raised from 0.10 to 0.15** as a direct result of this
evaluation. At 0.10, the `rparcero → Nayawan` case above cleared the rule by
0.010 and produced a real false identification. Measured separation:

| | Strangers | Genuine students |
|---|---|---|
| Margin | ≤ **0.110** | ≥ **0.222** * |

<sub>* excluding the one probe already refused at any setting</sub>

0.15 sits between the two with headroom on both sides, and costs nothing:
still 44/45 linked, now with 0 false identifications instead of 1.

The score threshold was deliberately **left at 0.30** rather than also being
raised. Raising it would work on this dataset, but these are cooperative
close-up photos; live probes come from a YOLO person-crop at queue-zone
distance and score lower. Raising the absolute bar would risk refusing real
students in deployment. The margin is the safer lever because it measures a
*relative* gap, which survives an overall drop in score quality.

### Limitation — state this in the thesis

These probes are enrollment-style photos: close-up, cooperative, well-lit.
Production probes are YOLO person-crops at queue-zone distance, which is a
harder input. **Treat these figures as an upper bound on live performance.**
The live counterpart is measured by `ML/report_recognition_metrics.py` from
the `recognition_metrics` and `face_match_events` tables, and should be
reported alongside these once beta data exists.

---

## A2. API performance under load

**Script:** `load_testing/run_load_test.py` → `load_testing/results/final_readonly_*.csv`
**Configuration:** 50 concurrent users, 2 minutes, 3 user classes
(student status poll, public display board, staff analytics), against a real
seeded ticket rather than a 404 path.

| Endpoint | p50 | p95 | p99 | Failures |
|---|---|---|---|---|
| `/api/queue/display` | 7 ms | 13 ms | 46 ms | 0 |
| `/api/queue/prediction` | 13 ms | 24 ms | 43 ms | 0 |
| `/api/queue/status` | 13 ms | 26 ms | 85 ms | 0 |
| `/api/queue/analytics` | 21 ms | 36 ms | 50 ms | 0 |
| `/api/auth/login` | 410 ms | 710 ms | 710 ms | 0 |
| **Aggregated** | **12 ms** | **31 ms** | **85 ms** | **0 (0.00 %)** |

2,536 requests, 21.3 req/s sustained, **zero failures.**
(Counts match `load_testing/results/final_readonly_stats.csv` exactly — the
run's own closing terminal line reads 2,539, three requests that completed
after the CSV was written. Cite the CSV: it is the artifact in the repo.)

Login is slow *by design* — scrypt password hashing is deliberately expensive
to resist brute force. Because the endpoint is a plain `def` rather than
`async def`, FastAPI dispatches it to a threadpool, so the cost stays on that
thread: the rest of the API held p50 12 ms while logins ran concurrently.

> **Measurement note.** Earlier runs showed a ~2,100 ms p98/p99 tail. That was
> **not** server latency. On Windows, `localhost` resolves to `::1` (IPv6)
> first while uvicorn binds IPv4 only, so each new connection waits ~2 s to
> fall back. Measured first-request latency: `localhost` 2,036–2,072 ms vs
> `127.0.0.1` 7–9 ms. The load-test host default is now `127.0.0.1`. Do not
> quote the earlier CSVs (`readonly_*`, `student_only_*`, `low_concurrency_*`)
> — only `final_readonly_*` is valid.

---

## A3. Recognition latency and scaling

**Script:** `load_testing/bench_face_pipeline.py` → `ML/figures/04_latency_scaling.png`

| Stage | Cost |
|---|---|
| Face extraction (detect + embed), live path | ~258 ms mean |
| Match scan, 50 enrolled students | 0.76 ms (p50) |
| Match scan, 100 enrolled students | 0.96 ms (p50) |
| Match scan, 300 enrolled students | 2.93 ms (p50) |

The match scan is O(N) over the roster, which sounds like a scaling limit but
is not: even at 300 students it costs ~1 % of the extraction step. Recognition
speed is bounded by per-frame vision work, not by how many students enrol.

---

## A4. YOLO person detection — NOT YET MEASURED

**Status:** harness built and verified, ground-truth labelling outstanding.
**Script:** `ML/evaluate_yolo_accuracy.py` (`capture` then `score`)

No accuracy claim about person detection on *this* camera can be made yet.
The published COCO figures in Part B §1 describe the stock model on a public
benchmark, not this deployment.

What the harness measures is **counting accuracy** (MAE, bias, exact-count
rate, recall/precision, and error broken down by crowd size) rather than mAP,
because counting is what the system depends on — queue length, density and the
wait estimate all derive from the person count, and box-regression quality
beyond "tight enough to crop a head" changes nothing. Ground truth is also one
integer per frame, which a human can label in seconds, where mAP would require
boxing every person by hand.

If the panel asks specifically for mAP: this is stock YOLOv8n with unmodified
COCO-pretrained weights, so its published person-class AP applies as-is, and
this project makes no claim to have improved it.

> **The size filter has less headroom than the frame suggests.** On a saved
> test snapshot, YOLO detected a person at **0.93 confidence** whose box
> covered **72 %** of the frame — against a `MAX_BBOX_FRAC` of **0.85**. It
> was kept, with about 13 percentage points to spare, but a student standing
> closer than that one would cross the limit and be discarded entirely: no
> detection, no face crop, no ticket, and indistinguishable from a plain miss
> in the count.
>
> The capture stage therefore draws filtered boxes in **red** and reports them
> separately, so a real capture run shows immediately whether the filter is
> removing anyone who belongs in the queue.
>
> *(An earlier revision of this section stated the box had been discarded at
> `MAX_BBOX_FRAC = 0.70`. That was the fallback written in `app/detector.py`,
> not the deployed value — `.env` sets 0.85 and overrides it. The box was
> kept. Every threshold in this document is the `.env` value where one is
> set, because that is what actually runs.)*

---

# PART B — ESTIMATED BEHAVIOUR (May 2026, not measured)

The sections below predate the face-recognition pivot and were written from
published figures and prototype observation. They are retained as design
rationale. **The "Expected Accuracy" column is an expectation, not a result.**

All threshold values are taken directly from `app/services/queue_service.py`
(`wire_callbacks`) and `app/services/queue_tracker.py`. Values that come from
`.env` are the runtime defaults in `app/core/config.py`. Some have since
drifted from the code — where Part A and Part B disagree, Part A is correct.

---

## Summary Table

The **Status** column maps each row onto the current system. Rows marked
**RETIRED** describe code that no longer exists: the HSV appearance /
twin-guard / done-blacklist / spatial-fallback chain was replaced wholesale
by face-based re-entry matching during the SOP #2 pivot
(`app/services/queue_tracker.py`, "FACE-BASED RE-ENTRY MATCHING"). They are
kept here as a record of what was tried and why it was replaced — **do not
present them as part of the delivered system.**

Section numbers below the table do not line up with these row numbers past
row 8, because rows 8 and 9 are covered by a single combined section §8.

| # | Algorithm | Status | Accuracy Metric | Threshold / Parameter | Expected Accuracy |
|---|-----------|--------|----------------|----------------------|-------------------|
| 1 | YOLOv8n Person Detection | live (§1) | mAP50 (COCO person class) | conf ≥ 0.50, imgsz 480 | ~52–58 % mAP50 |
| 2 | Detection Gate Pipeline | live (§2) | False-positive suppression rate | 5 gates (area, fraction, aspect, motion, NMS) | Removes ≥ 90 % non-person FP |
| 3 | Candidate Confirmation | live (§3) | Ghost-track rejection rate | 14 frames + motion range > 8 px | Rejects static objects in ~1–2 s |
| 4 | Track-ID Remapping | live (§4) | Remap hit rate | IoU ≥ 0.10 or dist ≤ 180 px | High when person stays near last bbox |
| 5 | Appearance Signature | **RETIRED** (§5) | Intra-class correlation (same person) | 512-D HSV, EMA α = 0.4 | Same-person scores typically 0.5–0.95 |
| 6 | Re-identification (1 missing) | **RETIRED** (§6) | Correct match rate | score ≥ 0.30 or spatial < 0.20 | High for distinct clothing; lower for similar colours |
| 7 | Re-identification (multi) | **RETIRED** (§7) | Swap-free match rate | score ≥ 0.30, gap ≥ 0.10 | Conservative — ambiguous cases → new number |
| 8 | Twin Guard | **RETIRED** (§8) | Lookalike block rate | similarity ≥ 0.85 | Fires only on near-identical appearance |
| 9 | Done Blacklist | **RETIRED** (§8) | Re-entry suppression accuracy | similarity ≥ 0.55 | Catches served persons reliably |
| 10 | Duplicate Suppression | live (§9) | Ghost-track merge rate | IoU > 0.40 or centre_frac < 0.15 | Catches YOLO ghost tracks; safe for queue lines |
| 11 | Queue Zone Membership | live (§10) | Zone containment accuracy | polygon / rectangle test | Deterministic given a correct zone |
| 12 | No-show Detection | live (§11) | Timer accuracy | position = 1, missing ≥ 300 s | Deterministic — no false positives |
| 13 | M/M/c Wait-time Baseline | live (§12) | Mean absolute error vs actual | ρ = λ/(cμ), W = Q/λ + 1/μ | ±1–2 min when ρ < 0.7 |
| 14 | Holt's Smoothing (15-min) | live (§13) | Forecast error at horizon | α = 0.4, β = 0.2, φ = 0.85 | Better than naive for trending queues |
| 15 | Linear Projection (5-min) | live (§14) | Short-term trend accuracy | OLS slope, 20-sample window | Accurate for steady arrival rates |
| 16 | Growth Ratio (30-min) | live (§15) | Long-horizon error bound | 70 % growth + 30 % mean reversion | Capped at 60 min; conservative by design |
| 17 | Service Time Estimation | live (§16) | Measured vs assumed gap | 70/30 blend, 0.1–15 min filter | Converges to real gap after ~5 transactions |
| 18 | **Face Recognition Validation** | live — **MEASURED** | See Part A §A1 | 0.30 score + 0.15 margin | **Measured, not expected** |
| 19 | System-Minted Queue Number | live | Number-collision rate | range starts at 5000 | Deterministic — collision-free by construction |
| 20 | Sticky Counter Assignment | live | Counter-reassignment stability | keeps existing assignment while active | Deterministic |

Rows 18–20 have no Part B section: 18 is measured and documented in Part A
§A1, and 19–20 are deterministic bookkeeping with nothing to estimate. They
appear here so this table matches `ALGORITHMS.md`, which is the authoritative
description of what the system does.

---

## 1. YOLOv8n Person Detection

**File:** `app/detector.py`

### Accuracy Metric

mAP50 (mean Average Precision at IoU = 0.50) on the COCO person class.

### Published Performance

| Model | mAP50 (COCO all classes) | mAP50-95 | Parameters | Speed (CPU) |
|-------|--------------------------|----------|------------|-------------|
| YOLOv8n | 52.9 % | 37.3 % | 3.2 M | ~80 ms/frame |

Source: Ultralytics YOLOv8 documentation.

The person class alone typically achieves higher precision than the all-class
average because people are one of the highest-frequency classes in COCO
training data.

### Operating Thresholds

```
Confidence threshold (default): 0.45
High-confidence bypass:         0.55  (API_HIGH_CONF)
Static-object confidence bypass: 0.45 (QUEUE_STATIC_CONF_BYPASS)
```

### Expected Performance at Threshold

- At conf ≥ 0.45: moderate precision, higher recall — some false positives
  remain, which the gate pipeline (Section 2) removes.
- At conf ≥ 0.55: higher precision, lower recall — may miss partially occluded
  people. Resolved by accepting lower-confidence detections that pass motion and
  area gates.

### Failure Modes

| Mode | Condition | Effect |
|------|-----------|--------|
| False positive | Mannequin, large poster, occluded chair | Enters gate pipeline; rejected by motion gate |
| Missed person | Heavy occlusion, very small bbox | Not detected; enters as new number on return |
| Track loss | Person briefly exits frame | ByteTrack assigns new track_id; handled by remap or re-ID |
| Ghosting | YOLO emits two overlapping bboxes | Caught by duplicate suppression (Section 10) |

---

## 2. Detection Gate Pipeline

**File:** `app/detector.py`

The five gates are applied after YOLO output before any detection reaches the
queue tracker. Their combined purpose is to suppress non-person false positives.

### Gate Parameters and Pass Conditions

| Gate | Parameter | Default | Rejects |
|------|-----------|---------|---------|
| 1 — Minimum area | `API_MIN_BBOX_AREA` | 3 000 px² | Tiny objects, distant background |
| 2 — Maximum frame coverage | `MAX_BBOX_FRAC` | 0.85 of frame | Camera obstructions, wall decorations |
| 3 — Portrait aspect ratio | `QUEUE_MIN_PORTRAIT_ASPECT` | 0.60 (h/w) | Wide objects: bags, chairs, boxes |
| 4 — Motion energy | `QUEUE_MIN_MOTION_PIXELS` | 8 px changed | Static objects; bypassed at conf ≥ 0.45 |
| 5 — NMS (internal) | YOLO threshold | Built-in | YOLO's own overlapping detections |

### Accuracy Characteristic

The gates are conservative false-positive filters, not person classifiers.
They reduce the false-positive burden on the queue tracker. Each gate is
independent; a detection must pass all four manual gates plus YOLO's internal
NMS.

Expected false-positive suppression: ≥ 90 % of non-person objects that YOLO
misclassifies as persons are removed by Gate 1 (area) or Gate 4 (motion).

### Failure Modes

| Mode | Condition | Effect |
|------|-----------|--------|
| Gate 3 rejects large person | Camera angle gives wide bbox | Miss detection; person assigned new number on zone re-entry |
| Gate 4 rejects slow mover | Person stands completely still | May be rejected; resolved by confidence bypass at ≥ 0.45 |
| Gate 2 rejects near-camera person | Person is very close to camera | Detection skipped |

---

## 3. Candidate Confirmation — Anti-Ghost Filter

**File:** `app/services/queue_tracker.py` — `process_frame()`

### Accuracy Metric

Ghost-track rejection rate: the fraction of false detections (shadows,
momentary mis-detections, static objects) that are correctly suppressed before
a queue number is issued.

### Operating Thresholds

```
MIN_CONFIRM_FRAMES = 14     (from config; ~1.0 s at 14 fps)
MIN_MOTION_PIXELS  = 8      (centroid range must exceed this)
STATIC_STDEV_THRESHOLD = 1.5 px   (standard deviation of centroid positions)
STATIC_CONF_BYPASS = 0.45   (skip motion check for high-confidence detections)
```

### Expected Performance

A shadow or motionless object must be continuously detected for ≥ 14 frames.
Even if it persists, the motion check (centroid range ≤ 8 px or stdev ≤ 1.5 px)
rejects it as a static object. Real people naturally shift by more than 8 px
while walking into or adjusting in a queue.

The confirm buffer also absorbs brief track-ID flickers: a new track that
appears for 2 frames and disappears never accumulates the 14 required frames.

### Failure Modes

| Mode | Condition | Effect |
|------|-----------|--------|
| Queue-stationary person rejected | Person enters queue and freezes completely | Bypassed if conf ≥ 0.45 |
| Slow entry | Person enters frame very slowly | Accumulates confirm count slowly; ticket delayed by a few frames |

---

## 4. Track-ID Remapping

**File:** `app/services/queue_service.py` — `remap_track_ids()`

### Accuracy Metric

Remap hit rate: the fraction of ByteTrack ID reassignments that are correctly
mapped back to the known queue entry before reaching the queue tracker.

### Operating Thresholds

```
QUEUE_REMAP_IOU_THRESH      = 0.10  (IoU path — preferred)
QUEUE_REMAP_DIST_THRESH     = 180 px  (centroid distance path)
QUEUE_REMAP_ABSENT_FRAMES   = 45 frames (~3 s)  (max frames absent to remap)
```

### Expected Performance

When a person briefly leaves and returns with a new YOLO track_id, their
bounding box overlaps or is close to the last known position. IoU ≥ 0.10 fires
for any partial spatial overlap; the 180 px distance path covers re-entries
where the person stands within ~half a body-width of where they left.

Remapping is a best-effort pre-filter. When it succeeds, the queue tracker
sees the correct track_id and avoids re-identification overhead entirely.
When it fails, re-identification handles the new ID.

### Failure Modes

| Mode | Condition | Effect |
|------|-----------|--------|
| Remap miss | Person returns to a completely different position | Falls through to re-identification |
| Wrong remap | Two people swap positions during brief absence | Could mismap; appearance check in re-ID corrects it |
| Absent too long | Person absent > 45 frames | Remap disabled; re-identification must handle |

---

## 5. Appearance Signature Quality

> ### ⚠ RETIRED — not in the delivered system
> The 512-D HSV appearance signature was removed during the SOP #2 pivot and
> replaced by ArcFace face embeddings (Part A §A1). Sections 5–8 all describe
> this same retired chain. Kept as a record of what was tried; **do not present
> as a delivered feature.**

**File:** `QueueTracker._extract_appearance()`

### Accuracy Metric

Intra-class Pearson correlation: correlation between two signatures extracted
from the same person at different times. Higher is better (1.0 = identical).

Inter-class separation: correlation between signatures from two different
people. Lower means the signatures are more distinguishable.

### Parameters

```
Histogram dimensions:  16 × 16 bins per body half (H × S channels)
Signature length:      512 values (256 upper + 256 lower body)
EMA update weight:     0.4 (new frame) / 0.6 (stored signature)
Update interval:       every 30 s of wait time
Minimum crop size:     height ≥ 20 px, width ≥ 10 px
```

### Expected Correlation Ranges

| Pair Type | Typical Pearson Score | Interpretation |
|-----------|-----------------------|----------------|
| Same person, same lighting | 0.75 – 0.95 | Strong match |
| Same person, different lighting | 0.50 – 0.80 | Moderate match |
| Different people, different clothing | 0.05 – 0.35 | Clear separation |
| Different people, similar colour clothing | 0.30 – 0.60 | Potential confusion |
| Identical / uniform clothing (e.g. school uniforms) | 0.60 – 0.90 | High confusion risk |

### Failure Modes

| Mode | Condition | Effect |
|------|-----------|--------|
| Signature unavailable | Crop too small (person far from camera) | Returns None; falls back to spatial score |
| Poor lighting | Dark room, harsh backlight | Lower intra-class score; may fall below 0.30 threshold |
| Uniform clothing | All students in same uniform | Low inter-class separation; multi-person re-ID falls back to new number |
| Partial occlusion | Only top half of body visible | Lower body histogram becomes zeros; upper body still discriminates |

---

## 6. Re-identification — Single Missing Person

> **⚠ RETIRED.** Replaced by face-based re-entry matching
> (`queue_tracker._get_missing_face_candidates`). See §5 banner.

**File:** `QueueTracker._find_returning_person()`

### Accuracy Metric

Correct match rate: the fraction of re-entries that are matched to the right
queue entry (true positive). Also: false match rate — the fraction matched to
the wrong entry.

### Operating Thresholds

```
APPEARANCE_TIEBREAK_THRESHOLD = 0.30   (minimum similarity to re-identify)
Spatial fallback threshold    = 0.20   (centre_dist / avg_diag)
RECENCY_SINGLE_MATCH_SECONDS  = 60 s   (only match if missing < 60 s)
```

### Decision Logic

```
1. Appearance score ≥ 0.30  →  restore (confirmed appearance match)
2. Appearance score < 0.30 AND spatial_score < 0.20  →  restore (same pixel spot)
3. Otherwise  →  new number
```

The spatial fallback (step 2) only fires when the person is within roughly
15–20 % of a bounding-box diagonal from their last known position — equivalent
to a few pixels of jitter at typical camera distance.

### Expected Performance

| Scenario | Expected Outcome |
|----------|-----------------|
| Person leaves briefly (< 60 s), same clothing, same spot | Correctly re-identified via appearance |
| Person leaves briefly, appearance score 0.25, same spot | Correctly re-identified via spatial fallback |
| Person with similar shirt to someone else | Blocked if score < 0.30; new number |
| Person absent > 60 s and returned | Only matched if appearance score ≥ 0.30 |
| Person returns with original YOLO track_id | Fast-pathed through same-track-id branch; dedup immunity still set |

### Failure Modes

| Mode | Condition | Effect |
|------|-----------|--------|
| False match | Another person with similar clothing enters same spot | Mitigated by 0.30 threshold; edge case for uniform clothing |
| Missed match | Genuine returnee has appearance score 0.28–0.30 | Assigned new number (conservative — safer than wrong match) |

---

## 7. Re-identification — Multiple Missing Persons

> **⚠ RETIRED.** Replaced by face-based re-entry matching. See §5 banner.

**File:** `QueueTracker._find_returning_person()` — multi-missing branch

### Accuracy Metric

Swap-free match rate: the fraction of multi-person re-entries where queue
numbers are correctly assigned (not swapped between people).

### Operating Thresholds

```
APPEARANCE_TIEBREAK_THRESHOLD = 0.30   (minimum score)
Required score gap            = 0.10   (winner must lead second-best by 0.10)
Spatial fallback              = REMOVED (was source of swap bugs)
Blind fallback                = REMOVED (was source of swap bugs)
```

### Decision Logic

```
best_score ≥ 0.30  AND  (best_score - second_best) ≥ 0.10
  →  restore to best-matching entry

Otherwise (ambiguous or close scores)
  →  return None  →  new queue number assigned
```

### Expected Performance

When multiple people are missing simultaneously and return, the system only
restores a match when the appearance evidence is unambiguous. This means
some genuine returnees will receive new numbers, but it prevents the worse
outcome of swapping queue positions.

| Scenario | Expected Outcome |
|----------|-----------------|
| Both people have very different clothing | Correct re-ID for both (high gap) |
| Similar clothing — one clear winner | Winner restored; second gets new number |
| Similar clothing — both ambiguous | Both get new numbers (conservative) |
| People return to swapped positions | Both get new numbers (spatial removed) |

---

## 8. Twin Guard and Done Blacklist

> **⚠ RETIRED — and the Done Blacklist was removed for a reason worth citing.**
> It blocked a student from re-entering the queue after being served, which
> broke the legitimate case of a student who needs a second transaction the
> same day. It was deleted outright, not replaced. The twin guard is likewise
> gone — the face margin rule (Part A §A1) now does that job, with measured
> evidence that it works.

**File:** `QueueTracker._has_active_lookalike()`, `_matches_done_person()`

### Operating Thresholds

```
TWIN_LOOKALIKE_SCORE  = 0.85  (active-person match → block re-identification)
DONE_BLACKLIST_THRESH = 0.55  (done-person match → suppress new entry)
```

### Accuracy Characteristic

The twin guard fires only at 0.85, which is near the top of the practical
Pearson correlation range for HSV histograms. At this threshold, two people
must have nearly identical colour distribution across both body halves — clothing
and trousers in the same shade. In practice, identical school uniforms can reach
0.80–0.90; random street clothing very rarely exceeds 0.70 unless the colours
are the same.

The done blacklist at 0.55 is intentionally more lenient to catch the same
person re-entering after being served, even if lighting changed slightly between
entry and re-entry.

### Failure Modes

| Mode | Condition | Effect |
|------|-----------|--------|
| Twin guard false positive | Two people in nearly identical uniforms | Treated as twin; re-entry blocked; staff uses force-new override |
| Done blacklist false positive | New person with coincidentally similar clothing to served person | Suppressed from queue; staff uses force-new override |
| Done blacklist miss | Served person re-enters with changed appearance (e.g. removed jacket) | Assigned new number |

---

## 9. Duplicate Suppression (Dedup)

**File:** `QueueTracker._dedup_active_queue()`, `_is_duplicate_of()`

### Accuracy Metric

Ghost-track merge accuracy: correctly identifying two active queue entries
as the same physical person (true positives) while not merging two genuinely
different people standing close in line (false positives).

### Operating Thresholds

```
DEDUP_IOU_THRESH    = 0.40  (IoU path — hardcoded in wire_callbacks)
DEDUP_CENTRE_FRAC   = 0.15  (centre_dist / avg_diag — hardcoded in wire_callbacks)
Dedup immunity      = 60 frames  (set on re-entry)
Appearance guard    = similarity < 0.55  →  skip dedup permanently
```

### IoU Overlap Ranges by Scenario

| Scenario | Typical IoU Range | Dedup Action |
|----------|--------------------|--------------|
| YOLO ghost track (same person, two bboxes) | 0.60 – 0.90 | Merged (correct) |
| Two people walking side by side | 0.05 – 0.20 | Not merged (correct) |
| Two people standing in queue, close | 0.00 – 0.15 | Not merged (correct) |
| Two people partially overlapping | 0.20 – 0.40 | Not merged (correct) |
| Person returning with same track_id | 0 (not overlapping yet) | Immunity prevents merge |

### Expected Performance

With IoU threshold at 0.40 and centre-fraction at 0.15 of the average
bounding-box diagonal:

- Ghost tracks (same person, two bboxes): typically merged within 1 frame.
- Adjacent people in queue line: IoU is typically 0–0.15, well below 0.40.
- Centre-fraction 0.15 × diagonal ≈ 20–25 px for a typical 150-px-tall bbox,
  which is smaller than the gap between any two adjacent people.

The 60-frame immunity window (≈ 4 seconds at 14 fps) protects re-entering
persons from being merged during the brief period when their new bbox may be
spatially close to another person's bbox.

The appearance guard permanently protects any two distinct people (similarity
< 0.55) from ever being merged, even after immunity expires.

### Failure Modes

| Mode | Condition | Effect |
|------|-----------|--------|
| Merge miss | Ghost track IoU just below 0.40 | Two entries persist briefly; ByteTrack usually resolves naturally |
| False merge | Two people with very similar clothing overlap | Appearance guard (< 0.55) prevents this when signatures are built |
| Merge during immunity | Immunity prevents genuine ghost-track merge | Extra entry expires via MAX_MISSING_FRAMES within ~16 s |

---

## 10. Queue Zone Membership

**File:** `QueueZone.is_person_inside()`

### Accuracy Metric

Zone classification accuracy: the fraction of frame-person pairs where the
inside/outside decision matches the ground truth (person visually inside or
outside the drawn zone).

### Method

The decision is based on the bounding-box centroid — not any edge. This means
a person's arm or foot can cross the zone boundary without the person being
counted.

```
inside = (zone_x1 ≤ centroid_x ≤ zone_x2)  AND  (zone_y1 ≤ centroid_y ≤ zone_y2)
```

### Accuracy Characteristic

Centroid-based zone membership is deterministic given a correctly calibrated
zone. Accuracy depends entirely on zone calibration quality:

- A well-calibrated zone (calibrated for the actual camera angle) gives near
  100 % correct inside/outside classification.
- A poorly calibrated zone (set on a different camera position) introduces
  systematic error.

### Failure Modes

| Mode | Condition | Effect |
|------|-----------|--------|
| Zone miscalibration | Camera moved after zone was set | People counted inside when outside (or vice versa) |
| Edge-case entry | Person enters zone from a side angle with centroid crossing late | 1–3 frame delay before zone entry is detected |
| Bounding-box error | YOLO bbox clips the person (too tight/loose) | Centroid shifts, person may appear outside when physically inside |

---

## 11. No-show Detection

**File:** `QueueTracker._check_noshow()`

### Accuracy Metric

Precision and recall of automatic no-show bumps:

- True positive: person at position 1 is genuinely absent for ≥ 300 s.
- False positive: person is briefly outside the camera frame but has not
  actually left.
- False negative: person is genuinely absent but is not at position 1.

### Operating Thresholds

```
NOSHOW_WINDOW_SECONDS = 300 s  (5 minutes, configurable)
Condition:            position_in_line == 1  AND  status == 'missing'
Auto-noshow enabled:  controlled by QUEUE_AUTO_NOSHOW_ENABLED (default: False)
```

### Expected Performance

The timer is deterministic and position-specific. Only the person at position 1
can be auto-bumped. The 300-second window is long enough to cover brief bathroom
breaks and short exits. The timer resets automatically if the person returns.

The `QUEUE_AUTO_NOSHOW_ENABLED = False` default means auto-bumping is off in
the current configuration — the no-show countdown and alert are shown to staff
but no automatic bump occurs. Staff must manually confirm the no-show.

### Failure Modes

| Mode | Condition | Effect |
|------|-----------|--------|
| False positive (if auto enabled) | Person steps out briefly, re-ID fails on return | Person bumped; staff can manually add back |
| False negative | Person is at position > 1, genuinely absent | Timer does not start; only position 1 is monitored |
| Re-ID success hides timer | Person returns with enough appearance score | Timer cleared correctly (correct behavior) |

---

## 12. M/M/c Wait-time Baseline

**File:** `app/services/prediction_service.py`

### Accuracy Metric

Mean absolute error (MAE) in minutes between the estimated wait time and
the actual wait time experienced by a person.

### Model Assumptions

The M/M/c (Erlang-C) model assumes:

1. Poisson arrivals at rate λ (persons per minute).
2. Exponentially distributed service times with mean 1/μ.
3. c identical parallel counters.
4. Infinite waiting capacity.

### Operating Formula

```
Server utilisation:  ρ = λ / (c × μ)

Wait time estimate:
  W = (Q / max(λ, 0.1)) + avg_service_time   when ρ < 1
  W = Q × avg_service_time / c               when ρ ≥ 1 (saturated)

Default service time:  3.0 min  (configurable; auto-updated every 5 min)
```

### Expected Accuracy by Utilisation

| Queue Load (ρ) | Expected MAE | Notes |
|----------------|--------------|-------|
| ρ < 0.5 (light) | ± 0.5 – 1.0 min | Model assumptions hold well |
| 0.5 ≤ ρ < 0.7 (moderate) | ± 1 – 2 min | Poisson assumption less accurate |
| 0.7 ≤ ρ < 1.0 (heavy) | ± 2 – 5 min | Erlang-C wait diverges quickly |
| ρ ≥ 1.0 (saturated) | Fallback formula used | Infinite-wait regime |

### Failure Modes

| Mode | Condition | Effect |
|------|-----------|--------|
| Underestimate | Actual arrivals are bursty (non-Poisson) | Wait time understated at peak |
| Service time wrong | Service time drifts from default 3 min | Wait estimate off by factor of (actual / assumed); corrected every 5 min from DB |
| Zero arrivals | No recent arrivals — λ = 0 | Falls back to Q × avg_service_time / c |

---

## 13. Holt's Double Exponential Smoothing (15-min Forecast)

**File:** `app/services/prediction_service.py`

### Accuracy Metric

Mean absolute percentage error (MAPE) at a 15-minute horizon.

### Operating Parameters

```
α (level smoothing)  = 0.4
β (trend smoothing)  = 0.2
φ (damping factor)   = 0.85
h (horizon)          = 20 samples
```

### Expected Performance

| Queue Pattern | Expected Accuracy |
|---------------|------------------|
| Steady arrival rate | Low error; level tracks actual count closely |
| Rising arrivals | Trend term accelerates forecast; damping (φ = 0.85) limits runaway |
| Falling arrivals | Trend term decelerates forecast; may slightly overestimate |
| Sudden spike | α = 0.4 adapts within ~5–7 samples (≈ 2.5 min at one sample/30 s) |
| Short history (< 10 samples) | Trend unreliable; level average dominates |

Damping with φ = 0.85 means the trend's influence after h = 20 steps is
reduced to φ²⁰ ≈ 0.039 of its initial value, preventing extreme long-range
extrapolation from current trend.

### Failure Modes

| Mode | Condition | Effect |
|------|-----------|--------|
| Step change | Queue suddenly doubles | Level adjusts in ~3–5 samples; under-forecasts during transition |
| Trend reversal | Queue grows then abruptly shrinks | Damping limits over-forecast; still positive for ~3–4 samples |
| Short history | System just started | β = 0.2 with few samples produces unstable trend |

---

## 14. Short-term Forecast (5-min) — Linear Trend Projection

**File:** `app/services/prediction_service.py`

### Accuracy Metric

Root mean square error (RMSE) at 5–10 minute horizon.

### Method

OLS regression slope over the last 20 samples, then projected 600 seconds
(10 min frame × fps) forward.

```
β = Σ(xᵢ · yᵢ) / Σ(xᵢ²)     window = 20 samples
Q̂ = max(0, Q_current + β × fps × 600)
```

### Expected Performance

| Scenario | Expected Accuracy |
|----------|------------------|
| Steady growth | Good; OLS slope matches actual rate |
| Stable queue | OLS slope ≈ 0; projection = current count |
| Rapid change | OLS smooths over last 20 samples; lags sudden changes |

### Failure Modes

| Mode | Condition | Effect |
|------|-----------|--------|
| Noisy arrivals | High variance counts | OLS slope underestimates/overestimates |
| Trend reversal | Queue peaks and drops within 20 samples | Slope sign changes; projection overshoots |

---

## 15. Long-term Forecast (30-min) — Growth Ratio + Mean Reversion

**File:** `app/services/prediction_service.py`

### Accuracy Metric

Directional accuracy (rising / falling / stable) at 30-minute horizon.

### Method

```
Q_early  = mean(first third of history buffer)
Q_recent = mean(last third of history buffer)
g        = Q_recent / max(Q_early, 1.0)

Q_long   = Q_recent × (0.7 × g + 0.3)
W_long   = min(W_raw, 60 min)   (capped)
```

### Expected Performance

The 30 % mean-reversion term prevents the model from extrapolating indefinitely.
The 60-minute cap limits the damage from over-forecasting. At 30-minute horizons,
exact prediction is unreliable for any queueing model; the goal is directional
correctness (will the queue grow or shrink?), not precise minute counts.

### Failure Modes

| Mode | Condition | Effect |
|------|-----------|--------|
| Short history | < 10 samples in buffer | g is dominated by noise |
| Flat queue | Q_early ≈ Q_recent → g ≈ 1.0 | Reasonable; projects current level |
| Queue spike | Sudden large arrival | g jumps; 30 % reversion partially corrects |

---

## 16. Dynamic Service Time Measurement

**File:** `app/database/database_handler.py` — `measure_avg_service_time()`

### Accuracy Metric

Agreement between measured inter-departure gap (in minutes) and the actual
average time a cashier spends per person.

### Method

```
gap = (timestamp[i] - timestamp[i-1]) / 60.0   (minutes)
Filter: 0.1 min ≤ gap ≤ 15.0 min
measured = mean(gaps) × num_counters
blended  = 0.7 × measured + 0.3 × old_value
```

Update interval: every 300 seconds (5 minutes).

### Expected Performance

| Transaction Count | Convergence |
|-------------------|-------------|
| < 5 transactions | Sparse; estimate unreliable |
| 5–10 transactions | Rough estimate; blending with previous value stabilises |
| > 15 transactions | Converges toward actual service time |

The 70/30 blend smooths one unusually fast or slow service event. The 0.1–15 min
filter excludes cashier idle time between customers.

### Failure Modes

| Mode | Condition | Effect |
|------|-----------|--------|
| Few records | New deployment or reset database | Falls back to ENV default (3.0 min) |
| Irregular service | One very fast cashier and one slow | Mean may not represent either; use separate counter config |
| Idle periods | Long gap between customers (> 15 min) | Filtered out correctly |

---

## Accuracy Caveats for Thesis Evaluation

1. **YOLOv8n mAP figures** are measured on the COCO validation set. Accuracy in
   a campus service office environment may differ due to different backgrounds,
   lighting conditions, and crowd density from COCO.

2. **Re-identification accuracy** is highly dependent on clothing diversity in
   the target deployment. Environments with school uniforms will have lower
   inter-class separation, meaning more conservative (new-number) decisions.

3. **Wait-time prediction accuracy** cannot be fully evaluated without real
   service-time data. The M/M/c model is a standard operations research baseline;
   its accuracy in practice depends on whether arrival and service distributions
   match Poisson/exponential assumptions.

4. **Dedup and re-ID thresholds** (0.30, 0.40, 0.55, 0.85) were tuned against
   observed behaviour in prototype testing, not against a labelled ground-truth
   dataset. A formal evaluation would require annotated video with ground-truth
   identities.

5. All prediction models use a rolling buffer of 60 samples. At system startup,
   fewer samples are available and accuracy is lower than the steady-state values
   described above.
