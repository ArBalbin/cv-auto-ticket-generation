# QueueFlow Development Pitfalls

Last updated: May 19, 2026

This document records the real bugs, design errors, and architectural surprises
encountered while building the QueueFlow prototype. Each entry describes what was
observed, why it happened, what was changed, and the lesson for future work.

The pitfalls are grouped by subsystem. They are drawn from actual testing sessions
and terminal log analysis, not hypothetical scenarios.

---

## 1. Re-identification Pitfalls

---

### 1.1 Two Re-entry Code Paths That Must Be Kept in Sync

**Symptom**

When Person 1 left the queue and returned, the system correctly re-identified them.
But Person 2's queue number disappeared immediately after — without being marked done
by staff.

Terminal log showed:
```
♻️ Dedup: Q002 is duplicate of Q001 — retiring ghost
```

This happened even though Person 1 and Person 2 were two different people standing
close together in the queue.

**Root Cause**

The re-entry immunity (`dedup_immune_frames = 60`) was only set inside
`_restore_missing_person`, which is called when YOLO assigns a **new** track_id
to the returning person.

When YOLO reuses the **original** track_id, the code takes a different branch:

```python
if track_id in self.active_queue:
    p.last_seen = datetime.now()
    p.missing_frames = 0
    ...
    continue   # _restore_missing_person is NEVER called
```

In this fast path, `dedup_immune_frames` was never set. On the very next frame,
`_dedup_active_queue` ran and saw two people close together, with no immunity on
either of them, and deleted the one with the higher queue number.

**Fix Applied**

Added immunity in the same-track-id fast path:

```python
if p.missing_frames > 0:
    p.dedup_immune_frames = 60   # re-entering — protect from dedup
p.missing_frames = 0
```

**Lesson**

When a person can return via two different code branches (new track_id vs same
track_id), every piece of state that must be set on re-entry must be set in
**both** branches. It is easy to add logic to one branch and forget the other.
Write a comment at the branch fork to make the relationship explicit.

---

### 1.2 Appearance Tiebreak Threshold Too Low

**Symptom**

Person 2 wore a T-shirt with a similar colour to Person 1. When Person 1 left the
queue and Person 2 remained, Person 2 was not affected. But when Person 1 returned,
the system re-identified Person 2 as Person 1 — giving Person 2 queue number Q001
instead of keeping Q002.

**Root Cause**

`APPEARANCE_TIEBREAK_THRESHOLD = 0.12` (the initial value) was far too low. A
Pearson correlation of 0.12 is barely above random noise — almost any two people
with non-monochrome clothing can score 0.12 or above because the histogram bins
always have some overlap.

**Fix Applied**

Raised to `APPEARANCE_TIEBREAK_THRESHOLD = 0.30` in `wire_callbacks()`.

At 0.30, two people must have a meaningful statistical similarity in their HSV
colour distribution before the system treats them as the same person. Random
clothing overlap typically stays below 0.20.

**Lesson**

Pearson correlation on HSV histograms ranges from −1.0 to 1.0 but in practice
almost never goes below 0.0 for person crops, because all people have some similar
skin tone, background bleed, or fabric shading. A threshold near 0.0 or 0.12 is
effectively no threshold at all. Start calibration at 0.30+ and tighten from there
based on test observations.

---

### 1.3 Single-Person Spatial Fallback Too Lenient

**Symptom**

When Person 1 left the queue zone and Person 2 remained, Person 2 was occasionally
re-identified as Person 1 — inheriting the wrong queue number — even though they
were clearly different people.

**Root Cause**

The spatial fallback threshold was `sp_score < 0.55` (spatial score is
centre-distance divided by average bounding-box diagonal; lower = closer).
A spatial score of 0.55 corresponds to the new detection being more than half
a body-diagonal away from the last known position. This covers almost the entire
queue area if the persons are standing close.

The fallback fired because Person 2 was standing in roughly the same region of
the frame as Person 1 had been — not the exact same pixel spot.

**Fix Applied**

Tightened the single-person spatial fallback to `sp_score < 0.20`.

At 0.20, the new detection must be within about 20 % of an average bounding-box
diagonal from the last known position — roughly 20–30 px for a typical 150-px-tall
bbox. This corresponds to slight jitter, not a different person standing nearby.

**Lesson**

Spatial position is not a reliable identity signal in a queue. Multiple people
stand near each other by design. The spatial fallback should only fire as a
very last resort for tiny positional jitter, not as a general proximity check.
Reserve spatial cues for the case where the person literally returned to within
a few pixels of where they left.

---

### 1.4 Multi-Person Spatial and Blind Fallbacks Caused Queue Number Swaps

**Symptom**

Person 1 (Q001) and Person 2 (Q002) both left the queue. When they returned,
Q001 was assigned to Person 2 and Q002 was assigned to Person 1 — their numbers
were swapped.

Separately: when both returned, Q001 was correctly identified but Q002 disappeared
without being marked done.

**Root Cause**

Two fallbacks in the multi-person re-entry branch:

1. **Spatial fallback** — if no appearance winner was found, pick the missing person
   whose last known position is closest to the new detection.
2. **Blind fallback** — if the spatial score was also poor, assign the most recently
   missing person.

When both people had similar appearance scores (e.g. similar clothing), neither won
the appearance round clearly. The spatial fallback then fired — but if Person 1 and
Person 2 returned to each other's positions in the queue, the spatial fallback
systematically assigned the wrong queue number to each.

**Fix Applied**

Removed both fallbacks entirely from the multi-person branch:

```python
# Multiple missing persons + ambiguous appearance — do not guess.
print(f"⚠️  Multi-person ambiguous — new number")
return None, None, 0.0
```

A clear appearance winner (score ≥ 0.30 and gap ≥ 0.10 over second-best) is now
required. When appearance is ambiguous, the person receives a new number.

**Lesson**

Spatial position is unreliable when multiple people are simultaneously missing
because they may return to each other's spots. A wrong assignment (swap) is far
worse than a new number, because a swap silently corrupts queue order. It is
better to issue a fresh number conservatively than to guess and get it wrong.

---

### 1.5 Appearance Signature Not Available on First Frames

**Symptom**

A returning person was sometimes not re-identified on the first frame they appeared
in the zone, even when their appearance should have matched. They were then
buffered through the candidate confirmation pipeline (14 frames) before receiving
a new number — appearing as a duplicate for a brief moment.

**Root Cause**

`_extract_appearance(frame, bbox)` returns `None` when:

- `frame` is `None` (frame was not available at that point in the pipeline), or
- the crop is smaller than 20 × 10 pixels (person too far from camera).

The first-sight re-entry check skips entirely when `new_sig is None`, so the
person enters the candidate accumulation pipeline instead.

**Fix Applied**

Three re-entry check points in `process_frame`:

1. **First sight** (before candidate accumulation).
2. **Mid-accumulation** (inside the candidate frame loop).
3. **Final gate** (at `MIN_CONFIRM_FRAMES`).

Any of these three points can restore the person, so even if the first-sight
check is skipped (no signature yet), the mid-accumulation check fires as soon
as a signature becomes available.

**Lesson**

For any algorithm that depends on computed features (appearance, motion), design
the pipeline to retry the check when the feature first becomes available, rather
than failing silently on the first check. Multi-gate pipelines are more robust
than single-point checks.

---

## 2. Duplicate Suppression Pitfalls

---

### 2.1 Queue-Standing Pattern Matched by the Dedup Condition

**Symptom**

With four people in the queue, Q003 suddenly disappeared without being marked done.
Staff did not perform any action. The terminal showed:

```
♻️ Dedup: Q003 is duplicate of Q002 — retiring ghost
```

No YOLO ghost track was present — both Q002 and Q003 were real, distinct people.

**Root Cause**

The original `_is_duplicate_of` function had three conditions, the third being:

```python
# x_column_overlap: how much horizontal extent the bboxes share
# y_adjacent: whether one bbox is directly below the other
if x_column_overlap(bbox_a, bbox_b) > 0.60 and y_adjacent(bbox_a, bbox_b):
    return True
```

This was intended to catch a YOLO ghost track where one bbox is above another
overlapping bbox of the same person. But in a queue, people stand **in a column**
— each person directly behind the next, sharing horizontal extent. The exact
pattern this condition was designed to catch (column + vertical adjacency) is
also the exact standing pattern of a queue.

On top of that, the IoU threshold was `0.10` (any 10 % overlap) and the
centre-fraction was `0.50 × average_diagonal` — both far too loose for a
real-world queue line.

**Fix Applied**

- Removed the column + adjacent condition entirely.
- Raised IoU threshold from 0.10 → 0.40 in `wire_callbacks()`.
- Tightened centre-fraction from 0.50 → 0.15 in `wire_callbacks()`.

YOLO ghost tracks overlap by 60–90 %. Two adjacent queue members overlap by
0–15 %. The new IoU threshold of 0.40 cleanly separates these cases.

**Lesson**

When designing a dedup condition for a crowd-monitoring system, verify it against
the specific spatial pattern of the deployment environment — not just against generic
computer vision ghost-track scenarios. A queue is a column of people. Any condition
based on "two bboxes in a vertical column" will fire on real queue members.

---

### 2.2 Config Defaults and Class Defaults Were Inconsistent

**Symptom**

After the dedup thresholds were tightened via `wire_callbacks()`, testing with a
fresh server start still occasionally showed dedup firing incorrectly. The hardcoded
class constants in `QueueTracker` still had the old loose values.

**Root Cause**

`QueueTracker` has two sources of threshold values:

1. Class-level constants at the top of the class (e.g. `DEDUP_IOU_THRESH = 0.15`).
2. Values set by `wire_callbacks()` in `queue_service.py` on startup.

If `wire_callbacks()` is called after some frames are already processed, the
class defaults apply during that window. The class defaults were:

```python
DEDUP_IOU_THRESH   = 0.15
DEDUP_CENTRE_FRAC  = 0.55
```

And the `.env` defaults were even looser:

```
QUEUE_DEDUP_IOU_THRESH    = 0.10
QUEUE_DEDUP_CENTRE_FRAC   = 0.50
```

**Fix Applied**

`wire_callbacks()` hardcodes the correct values directly:

```python
queue_tracker.DEDUP_IOU_THRESH  = 0.40
queue_tracker.DEDUP_CENTRE_FRAC = 0.15
```

These override both the class default and the `.env` value regardless of what is
in the environment file.

**Lesson**

When a configuration value has a safety-critical correct value (one that prevents
data corruption), do not leave it to a config file that an operator might not
change. Hardcode the correct value in the startup wiring function and document why.
Leave the class-level default as a last resort, not the intended operating value.

---

### 2.3 Dedup Immunity Timer Not Long Enough

**Symptom**

After the same-track-id return path was fixed (Pitfall 1.1), testing revealed that
Person 1's re-entry still occasionally caused dedup to fire — but now on a delayed
frame, after the person had already been standing in the queue for a few seconds.

**Root Cause**

The initial immunity was set to `20 frames` (≈ 1.4 seconds at 14 fps). During
the EMA smoothing period, Person 1's bbox was still converging toward its final
position. In certain camera angles, the smoothed bbox briefly overlapped Person 2's
bbox during the convergence transient, firing dedup after immunity expired.

**Fix Applied**

Raised immunity from 20 → 60 frames (≈ 4.3 seconds). This covers the full EMA
convergence window plus a safety margin for minor positional drift.

**Lesson**

Any immunity timer protecting a system from its own convergence behavior must be
set longer than the convergence time of the underlying signal, plus a margin for
normal variation. EMA with α = 0.45 has a settling time of approximately 10–15
frames; the immunity timer must exceed this.

---

### 2.4 Appearance Guard Needed for Permanent Protection

**Symptom**

Even after immunity expired (60 frames), two people standing consistently close
together in the queue (e.g. friends who naturally group) remained at risk of one
being deduped as a ghost track of the other on future frames.

**Root Cause**

Immunity is temporary — it expires. Two people who always stand shoulder-to-shoulder
will have a small-but-nonzero IoU on every frame. If IoU creeps above 0.40 due to
camera angle or minor positional drift, dedup fires as soon as immunity runs out.

**Fix Applied**

Added a permanent appearance guard in `_dedup_active_queue`:

```python
if (p_i.appearance_signature is not None
        and p_j.appearance_signature is not None):
    app_sim = self._best_score_against_person(p_i, p_j.appearance_signature)
    if app_sim < self.DONE_BLACKLIST_THRESH:   # 0.55
        continue   # visually distinct — never dedup
```

If two people look visually distinct (similarity < 0.55), dedup is permanently
skipped for that pair, even if their bboxes occasionally overlap.

**Lesson**

Temporal immunity solves the brief-window problem. Appearance-based immunity solves
the persistent-proximity problem. Both are needed in a real queue environment. A
system that only has one or the other will eventually fail.

---

## 3. Prediction Pitfalls

---

### 3.1 M/M/c Assumptions Do Not Hold Under Burst Arrivals

**Symptom**

Predicted wait times were accurate when the queue grew slowly but significantly
underestimated when a class dismissed and 8 students arrived simultaneously.

**Root Cause**

The M/M/c (Erlang-C) model assumes Poisson arrivals — random independent arrivals
at a constant rate. Campus service traffic typically has burst arrivals correlated
with class schedules. When ρ approaches 1.0, the Erlang-C formula shows very
large predicted waits that diverge quickly. When ρ ≥ 1.0, the model switches to
a fallback formula.

**Fix Applied**

No fix to the model itself — this is a known limitation of M/M/c. The three
additional forecast models (linear projection, Holt's smoothing, growth ratio)
cover different time horizons. The prediction API returns all four so the
dashboard can display the range rather than a single point estimate.

**Lesson**

M/M/c is a textbook baseline, not a real-time predictor for campus queues.
It is most accurate during stable, gradual arrivals. For a thesis, it is appropriate
to document the model's assumptions and limitations explicitly rather than implying
it predicts perfectly.

---

### 3.2 Default Service Time Does Not Reflect Actual Service

**Symptom**

Wait-time predictions during the first minutes after system startup were consistently
wrong — either too optimistic or too pessimistic.

**Root Cause**

The default service time (`AVG_SERVICE_TIME = 3.0 min`) is an estimate. The actual
service time varies by transaction type, staff speed, and time of day. On startup,
the prediction service uses this default until the database refresh loop has
collected enough real completed transactions.

The database refresh loop runs every 5 minutes. On a fresh system with no completed
records, the loop returns `None` and the default is kept indefinitely.

**Fix Applied**

The 70/30 blend (`0.7 × measured + 0.3 × previous`) damps sudden swings once
real data starts arriving. The `.env` default should be set per deployment based
on the known service time of the target office — it is documented in `.env.example`
with a note to calibrate this value.

**Lesson**

Any prediction system that depends on a configurable baseline must document the
baseline clearly and warn operators to calibrate it before going live. An uncalibrated
service time of 3 minutes in an office where service takes 7 minutes produces
predictions that are off by a factor of two.

---

## 4. Detector Pitfalls

---

### 4.1 Static Objects Pass YOLOv8 Confidence But Fail in the Queue

**Symptom**

A chair with a bag on it near the queue zone was occasionally detected as a person,
assigned a candidate count, and would have received a queue number if it accumulated
enough frames.

**Root Cause**

YOLOv8n's confidence score is not a guarantee against non-person detections.
At low thresholds (below 0.55), bags on chairs, large posters, and mannequins
can score above threshold. The candidate accumulation buffer collects frames
silently without issuing a number until `MIN_CONFIRM_FRAMES` is reached.

**Fix Applied**

Gate 4 (motion energy) rejects detections where the pixel difference between
frames is below `QUEUE_MIN_MOTION_PIXELS = 8`. The motion check using centroid
standard deviation (`σ < 1.5 px`) catches objects that are detected consistently
but do not move.

The portrait aspect-ratio gate (Gate 3, `h/w ≥ 0.60`) also rejected the specific
bag-on-chair case because the combined detection was wide.

**Lesson**

In a fixed-camera deployment, most false positives are static. A simple motion
gate eliminates a large fraction of them without any model retraining. Always add
a motion or stability gate after the detector in a queue-tracking system.

---

### 4.2 ByteTrack Reassigns Track IDs on Brief Exits

**Symptom**

When a person briefly stepped out of the frame (for example, to move aside for
someone passing) and returned within 1–2 seconds, the queue tracker treated them
as a new person. This created a duplicate queue entry and triggered ticket
generation a second time.

**Root Cause**

ByteTrack maintains tracks as long as the person is in frame. When a person
exits and re-enters within the tracker's buffer window, it may reuse the same
track_id (typical case) or assign a new one (when the gap is long enough for
the track to be considered lost). The threshold varies depending on ByteTrack
configuration and frame rate.

**Fix Applied**

The Track-ID Remapping step (`remap_track_ids` in `queue_service.py`) runs before
the queue tracker on every frame. It maps new unknown track_ids to known active-queue
entries by IoU and centroid distance, preventing brief-exit re-entries from reaching
the queue tracker as new IDs.

When remapping is insufficient, the re-identification pipeline (three-stage:
first sight, mid-accumulation, final gate) provides a second chance.

**Lesson**

Do not rely on the tracker's track_id staying stable across brief frame exits.
ByteTrack and similar trackers make no such guarantee. Any downstream system
that uses track_id as a person identity must either implement ID remapping or
treat every new track_id as potentially a returning person.

---

### 4.3 Ghost Tracks From Partial Occlusion

**Symptom**

When Person A walked behind Person B, YOLO briefly assigned two track_ids to what
was physically one person — the visible front of Person A and the combined
detection of the overlapping pair. This created two active queue entries for one
physical person.

**Root Cause**

ByteTrack maintains a detection for each perceived bounding box. When occlusion
splits or blurs a person's detection, the tracker may emit two partially overlapping
boxes. Both reach the queue tracker and both can become confirmed queue entries.

**Fix Applied**

`_dedup_active_queue` merges entries where IoU > 0.40 or where the centre-distance
is less than 15 % of the average bounding-box diagonal. Ghost tracks from occlusion
overlap by 60–90 %, well above the 0.40 threshold.

**Lesson**

Any multi-object tracking system deployed in a crowded scene will produce ghost
tracks. The dedup algorithm is essential, but its thresholds must be calibrated
to the scene. Too loose and it deletes real people; too tight and it misses ghost
tracks. Use IoU overlap ranges from real test footage to set the threshold.

---

## 5. Architecture and Deployment Pitfalls

---

### 5.1 Queue Tracker State Is Process Memory — Not Shared Across Instances

**Symptom**

Not an observed bug in testing, but a design issue discovered during cloud
deployment planning: running two FastAPI instances behind a load balancer would
give each instance its own `queue_tracker` object in memory. The two instances
would disagree on who is in the queue.

**Root Cause**

`queue_tracker` is a module-level singleton in `queue_service.py`. Python's
in-process memory is not shared across multiple processes or containers.

**Fix Applied**

The current deployment uses a single backend instance. Cloud deployment
documentation explicitly warns:

> For the thesis prototype, run one backend instance first. If multiple backend
> instances are used, active queue tracker state must be moved fully into Redis
> or MySQL so all instances share the same queue state.

**Lesson**

Any mutable state that is critical to the application's correctness must be
externalized (Redis, database) before horizontal scaling is possible. Document
this constraint before deployment.

---

### 5.2 Ticket PDFs Lost After Cloud Restart

**Symptom**

After redeploying the backend on a cloud platform, previously generated PDF tickets
were gone. QR links in students' hands pointed to files that no longer existed.

**Root Cause**

Cloud platforms (Render, Railway, Fly.io, etc.) use ephemeral file systems. Any
local files written during a deployment are deleted when the container restarts
or redeploys. The `TICKETS_OUTPUT_DIR` pointed to a local path inside the container.

**Fix Applied**

Optional S3-compatible object storage support was added. When
`OBJECT_STORAGE_ENABLED=1` is set, generated PDFs are uploaded to object storage
and the public URL is stored in the database instead of the local path.

For the prototype without object storage, the deployment checklist notes that
PDF loss on restart is expected and that thermal printer output (the final target)
does not have this problem.

**Lesson**

Never rely on local filesystem persistence in a cloud deployment. Any generated
file (tickets, reports, exports) must either be stored in a persistent volume, an
object storage service, or a database. Document this assumption before deployment.

---

### 5.3 PORTAL_BASE_URL Baked Into QR Code at Ticket Generation Time

**Symptom**

After changing `PORTAL_BASE_URL` from the local IP address to the cloud domain,
old QR codes stopped working. Students who had scanned their tickets before the
change could no longer load their queue status.

**Root Cause**

The QR code encodes the full status URL at the moment the ticket is generated:

```
{PORTAL_BASE_URL}/api/queue/status?q={queue_number}&token={short_code}
```

If `PORTAL_BASE_URL` changes, all previously printed QR codes still point to the
old URL.

**Fix Applied**

No fix — this is a fundamental property of printed/shared QR codes. The
deployment checklist explicitly states:

> After changing PORTAL_BASE_URL, generate new tickets. Old tickets may still
> point to the old local IP.

The thermal printer use case (final deployment) avoids this problem because new
physical tickets are printed fresh for every queue session.

**Lesson**

QR codes are static at print time. If the backend URL might change during a
prototype phase, design the QR payload to contain only a session code and resolve
the URL at scan time via a redirect service, or accept that old tickets will break.

---

### 5.4 Redis Persistence Settings Enable Snapshot Disk Writes

**Symptom**

A self-hosted Redis instance was retaining live camera snapshots between restarts.
On restart, an 8-hour-old annotated frame was briefly served as the "live" snapshot.

**Root Cause**

Redis defaults to RDB snapshot persistence (`save 900 1 300 10 60 10000`). This
writes the in-memory dataset to disk periodically. Short-TTL cache keys (snapshot
TTL = 10 s) are included in the RDB dump if a snapshot interval fires before
the key expires.

**Fix Applied**

A `redis.conf` file is included in the repository with:

```conf
save ""
appendonly no
```

Deployment documentation notes that managed Redis providers may still apply their
own persistence/backup independently of these settings.

**Lesson**

Redis is not automatically a volatile cache. Disk persistence must be explicitly
disabled. If live camera data is stored in Redis and privacy or staleness is a
concern, always verify that `save ""` and `appendonly no` are set and effective.

---

### 5.5 Zone Coordinates Must Be Recalibrated After Camera Movement

**Symptom**

After the camera was accidentally bumped and repositioned, the queue zone still
showed the rectangle from the previous angle. People standing in the correct
physical queue area were counted as outside the zone and received no queue numbers.

**Root Cause**

The zone is defined in pixel coordinates relative to the camera's current view.
Moving the camera changes what those coordinates represent in physical space. The
zone must be redrawn to match the new camera angle.

**Fix Applied**

The dashboard provides a zone-drawing interface accessible via
`POST /api/queue/zone`. Operators can redraw the zone without restarting the
system. The zone coordinates are not stored between restarts — they default to
a full-frame zone until a staff member draws a new one.

**Lesson**

Any calibration that depends on camera position (zone, scale, aspect ratio) is
fragile in a prototype where the camera may be moved. Document calibration as a
required step after every camera adjustment.

---

## 6. Queue Management Pitfalls

---

### 6.1 Counter Numbers Were Reassigned When Earlier Persons Finished

**Symptom**

Person at Counter 2 looked at the display board and saw their number reassigned
to Counter 1 after the person at Counter 1 was marked done. This confused both
the student and the cashier, who had already called them to Counter 2.

**Root Cause**

The original counter assignment re-sorted all active persons by queue number and
assigned counters 1, 2, 3, ... sequentially. When Q001 was marked done, Q002
became the new #1 in the sorted list and was assigned Counter 1, displacing Q003
from Counter 2 to Counter 1, and Q004 from Counter 3 to Counter 2, etc.

**Fix Applied**

Sticky counter assignment: once a person is assigned a counter number, it does
not change. Only **free** counter numbers (not currently held by anyone in the
active queue) are assigned to newly arriving persons:

```python
occupied      = {p.counter_number for p in active_line if p.counter_number}
free_counters = sorted(c for c in range(1, num_counters + 1) if c not in occupied)

for p in active_line:
    if p.counter_number is None and free_counters:
        p.counter_number = free_counters.pop(0)
```

**Lesson**

In a queue system, counter assignments are a contract with the person being served.
Once told "go to Counter 2," that assignment must not change while the person is
waiting. Any recalculation of counter numbers should only fill vacancies, never
move existing assignments.

---

### 6.2 No-show Auto-bump Disabled by Default

**Symptom**

During early testing with `QUEUE_AUTO_NOSHOW_ENABLED=True`, a person who briefly
stepped out of the camera's field of view was automatically bumped as a no-show
when re-identification failed.

**Root Cause**

The no-show countdown is correct in logic — it fires only for Position 1 persons
missing for ≥ 300 seconds. But in a prototype where the camera angle is not yet
optimised, a person can leave the frame without physically leaving the queue area.
Auto-bumping them causes data loss in MySQL (status set to no_show) that is
difficult to reverse.

**Fix Applied**

`QUEUE_AUTO_NOSHOW_ENABLED` defaults to `False`. The no-show countdown and alert
still display to staff (showing remaining seconds and critical/warning status), but
no automatic bump occurs. Staff must explicitly confirm the no-show by pressing the
no-show button.

**Lesson**

In a prototype, destructive automated actions (deleting records, changing statuses
permanently) should be off by default. Show the information to the operator and
let them decide. Automation can be re-enabled once the detection accuracy is
validated in the specific deployment environment.

---

## Summary of All Pitfalls

| # | Category | Pitfall | Fix |
|---|----------|---------|-----|
| 1.1 | Re-ID | Two return code paths; immunity only set in one | Set immunity in both branches |
| 1.2 | Re-ID | Appearance threshold 0.12 — effectively no threshold | Raised to 0.30 |
| 1.3 | Re-ID | Spatial fallback 0.55 — covers whole queue area | Tightened to 0.20 |
| 1.4 | Re-ID | Multi-person spatial/blind fallbacks cause number swaps | Removed fallbacks; ambiguous → new number |
| 1.5 | Re-ID | No signature on first frame → missed first-sight re-ID | Three re-entry check points in pipeline |
| 2.1 | Dedup | Column+adjacent condition matches queue standing pattern | Removed condition; IoU 0.10→0.40 |
| 2.2 | Dedup | Class defaults and .env defaults inconsistent | Hardcode correct values in wire_callbacks |
| 2.3 | Dedup | Immunity 20 frames too short for EMA convergence | Raised to 60 frames |
| 2.4 | Dedup | Immunity temporary; permanent proximity still risky | Added appearance guard (permanent) |
| 3.1 | Prediction | M/M/c assumes Poisson; burst arrivals break it | Documented limitation; multi-model output |
| 3.2 | Prediction | Default service time uncalibrated on startup | Document calibration in .env.example |
| 4.1 | Detector | Static objects pass YOLO confidence | Motion gate + aspect-ratio gate |
| 4.2 | Detector | ByteTrack reassigns IDs on brief exits | Track-ID remapping pre-filter |
| 4.3 | Detector | Ghost tracks from occlusion create duplicates | Dedup with IoU > 0.40 |
| 5.1 | Architecture | Queue tracker in process memory | Single instance; Redis future-work |
| 5.2 | Architecture | Cloud file system is ephemeral | Object storage option; documented |
| 5.3 | Architecture | PORTAL_BASE_URL baked into QR at print time | Document; thermal printer avoids it |
| 5.4 | Architecture | Redis default settings persist live snapshots | redis.conf: save "" appendonly no |
| 5.5 | Architecture | Zone coordinates tied to camera position | Dashboard zone-draw; recalibrate after move |
| 6.1 | Queue mgmt | Counter numbers reassigned when earlier person finishes | Sticky counter assignment |
| 6.2 | Queue mgmt | Auto no-show bumped genuine queue members | Default auto-bump off; staff confirms |
