# QueueFlow — Core Algorithm Reference

This document describes the algorithms powering the five core subsystems of QueueFlow:
person detection, appearance-based re-identification, queue management, wait-time forecasting,
and the public display board. Every code sample is taken directly from the production source.

---

## 1. Person Detection (YOLOv8n + ByteTrack)

**File:** `app/detector.py`

Each video frame is passed to a YOLOv8n model with ByteTrack multi-object tracking enabled.
The model returns a bounding box, confidence score, and a persistent `track_id` per person.

### Pre-filtering

Before any detection reaches the queue tracker, five filters are applied:

```python
# Minimum bounding-box area (px²)
area = (x2 - x1) * (y2 - y1)
if area < API_MIN_BBOX_AREA:          # default 800
    continue

# Maximum box fraction of the frame
frac = area / (frame_w * frame_h)
if frac > MAX_BBOX_FRAC:              # default 0.85
    continue

# Portrait aspect ratio  (height / width)
aspect = (y2 - y1) / max(x2 - x1, 1)
if aspect < QUEUE_MIN_PORTRAIT_ASPECT:   # default 0.60
    continue

# Motion energy — pixel-level change since last frame
diff          = cv2.absdiff(gray_now, gray_prev)
motion_pixels = int(np.count_nonzero(diff > 25))
if motion_pixels < QUEUE_MIN_MOTION_PIXELS:    # default 8
    # allow through only if confidence is very high
    if conf < QUEUE_STATIC_CONF_BYPASS:        # default 0.70
        continue
```

### Bounding box smoothing (EMA)

Raw bounding boxes jitter between frames. An exponential moving average with α = 0.45
is applied per tracked person so the displayed box is stable:

```python
# QueueTracker class constant
BBOX_SMOOTH_ALPHA = 0.45

# Applied in process_frame() for every active person
p.bbox = tuple(
    int(BBOX_SMOOTH_ALPHA * nb + (1 - BBOX_SMOOTH_ALPHA) * ob)
    for nb, ob in zip(new_bbox, p.bbox)
)
```

---

## 2. Queue Zone Membership

**File:** `app/services/queue_tracker.py` — `QueueZone.is_person_inside()`

A rectangular zone is defined by four pixel coordinates. A person is inside when their
bounding box **centroid** falls within the rectangle (using the centroid avoids counting
someone whose arm alone overlaps the boundary):

```python
def is_person_inside(self, bbox: tuple) -> bool:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) >> 1, (y1 + y2) >> 1
    return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2
```

---

## 3. Candidate Confirmation (Anti-Ghost Filter)

**File:** `QueueTracker.process_frame()`

A detection is not immediately assigned a queue number. It is buffered in `_candidates`
for `MIN_CONFIRM_FRAMES` (default 14) consecutive frames. Only after that does it graduate
to a real queue entry, preventing shadows or momentary mis-detections from generating numbers:

```python
cand['count'] += 1
# ...
if cand['count'] < self.MIN_CONFIRM_FRAMES:
    continue   # still accumulating — not a confirmed person yet

# Motion check over the accumulated frame history
avg_conf = sum(cand['confs']) / max(1, len(cand['confs']))
if not self._has_sufficient_motion(cand['centers'], avg_conf):
    self._candidates.pop(track_id)
    continue

# Confirmed new person — assign next queue number (thread-safe)
with self._lock:
    self._highest_assigned += 1
    num = self._highest_assigned
    self._used_numbers.add(num)
new_p = QueuePerson(queue_number=num, track_id=track_id, bbox=bbox)
```

---

## 4. Appearance Signature (HSV Split-Body Histogram)

**File:** `QueueTracker._extract_appearance()`

When a person enters the queue, a 512-value colour signature is extracted by splitting the
bounding-box crop into upper and lower halves and computing a 16×16 HSV histogram for each:

```python
@staticmethod
def _extract_appearance(frame, bbox) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 10:
        return None

    def _part_hist(part: np.ndarray) -> np.ndarray:
        if part.shape[0] < 8 or part.shape[1] < 8:
            return np.zeros(256, dtype=np.float32)
        hsv = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
        h   = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(h, h)
        return h.flatten()   # 256 values

    mid_y = crop.shape[0] // 2
    # Concatenate upper + lower → 512-value signature
    return np.concatenate([_part_hist(crop[:mid_y, :]), _part_hist(crop[mid_y:, :])])
```

**Signature update (EMA):** The signature is refined across frames to reduce noise:

```python
def _update_appearance(self, person: QueuePerson, frame, bbox):
    new_sig = self._extract_appearance(frame, bbox)
    if new_sig is None:
        return
    if person.appearance_signature is None:
        person.appearance_signature = new_sig
    else:
        # 60% weight on the stable running average, 40% on the fresh observation
        person.appearance_signature = 0.6 * person.appearance_signature + 0.4 * new_sig
    person.appearance_history.append(new_sig)
    if len(person.appearance_history) > 5:
        person.appearance_history.pop(0)
```

---

## 5. Appearance Comparison

**File:** `QueueTracker._compare_sigs()`

Two 512-value signatures are compared using Pearson histogram correlation on each body half,
then combined with a weighted average. Upper body receives more weight (0.60) because
shirt/jacket colour is more consistently visible than leg colour:

```python
@staticmethod
def _compare_sigs(sig1: np.ndarray, sig2: np.ndarray) -> float:
    if sig1 is None or sig2 is None:
        return 0.0
    if sig1.shape[0] == 512 and sig2.shape[0] == 512:
        upper = float(cv2.compareHist(
            sig1[:256].reshape(16, 16).astype(np.float32),
            sig2[:256].reshape(16, 16).astype(np.float32),
            cv2.HISTCMP_CORREL,
        ))
        lower = float(cv2.compareHist(
            sig1[256:].reshape(16, 16).astype(np.float32),
            sig2[256:].reshape(16, 16).astype(np.float32),
            cv2.HISTCMP_CORREL,
        ))
        # Upper body (shirt) is more discriminative — weight it more
        return 0.6 * upper + 0.4 * lower
    # Fallback for legacy 256-value signatures
    return float(cv2.compareHist(
        sig1.reshape(16, 16).astype(np.float32),
        sig2.reshape(16, 16).astype(np.float32),
        cv2.HISTCMP_CORREL,
    ))
```

Score range: 0.0 (completely different) → 1.0 (identical).
The appearance tiebreak threshold is **0.20** — above this, two signatures are the same person.

---

## 6. Re-identification (Returning Person Matching)

**File:** `QueueTracker._find_returning_person()`

When the tracker sees an unknown track ID after a brief occlusion, it attempts to match the
detection back to a known missing queue entry before assigning a new number.

### Single missing person

```python
if len(missing) == 1:
    tid, p = missing[0]
    secs = p.seconds_missing

    if secs <= self.RECENCY_SINGLE_MATCH_SECONDS:   # default 60 s
        if new_sig is not None and p.appearance_signature is not None:
            score = self._best_score_against_person(p, new_sig)
            if score >= tbreak:      # tbreak = 0.20
                return tid, p, score          # ✅ restore
            return None, None, 0.0            # ⚠️ different person
        # No appearance data — fall back to spatial proximity
        sp_score = self._spatial_score(p.bbox, bbox)
        if sp_score < 0.60:
            return tid, p, 0.8                # ✅ close enough
        return None, None, 0.0               # ⚠️ too far away

    # Gone longer than recency window — appearance is required
    score = self._best_score_against_person(p, new_sig)
    if score >= tbreak:
        return tid, p, score
    return None, None, 0.0
```

### Spatial score (normalised centroid distance)

```python
@staticmethod
def _spatial_score(last_bbox: tuple, new_bbox: tuple) -> float:
    cx1 = (last_bbox[0] + last_bbox[2]) / 2
    cy1 = (last_bbox[1] + last_bbox[3]) / 2
    cx2 = (new_bbox[0]  + new_bbox[2])  / 2
    cy2 = (new_bbox[1]  + new_bbox[3])  / 2
    dist  = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
    d1    = ((last_bbox[2]-last_bbox[0])**2 + (last_bbox[3]-last_bbox[1])**2) ** 0.5
    d2    = ((new_bbox[2] -new_bbox[0]) **2 + (new_bbox[3] -new_bbox[1]) **2) ** 0.5
    denom = (d1 + d2) / 2
    return dist / denom if denom > 0 else 999.0
```

### Multi-person matching

```python
# Rank all missing persons by appearance score
app_scored = sorted(
    ((self._best_score_against_person(p, new_sig), tid, p) for tid, p in missing),
    reverse=True
)
best_app, best_app_tid, best_app_p = app_scored[0]
second_app = app_scored[1][0] if len(app_scored) >= 2 else 0.0

# Accept only if top score is good AND clearly better than second
if best_app >= tbreak and (best_app - second_app) >= 0.10:
    return best_app_tid, best_app_p, best_app

# Ambiguous appearance — fall back to spatial proximity
sp_tid, sp_p, sp_score = self._best_spatial_match(bbox, missing)
if sp_p is not None and sp_score < 0.50:
    return sp_tid, sp_p, 0.8
```

---

## 7. Twin / Lookalike Guard

**File:** `QueueTracker._has_active_lookalike()`

Before attempting re-identification, the system checks whether the incoming candidate already
matches someone **currently in frame**. If so, they are a different person and get a new number:

```python
TWIN_LOOKALIKE_SCORE = 0.85   # class constant

def _has_active_lookalike(self, new_sig: np.ndarray, current_track_ids: set) -> bool:
    if new_sig is None:
        return False
    thresh = self.TWIN_LOOKALIKE_SCORE
    for tid, p in self.active_queue.items():
        if tid not in current_track_ids:
            continue              # only check people currently visible
        if p.status == 'done_pending':
            continue
        score = self._best_score_against_person(p, new_sig)
        if score >= thresh:
            print(f"⚠️  Twin ambiguity: candidate matches active Q{p.queue_number:03d} "
                  f"(score={score:.2f}) — new number will be assigned")
            return True
    return False

# Called at the start of _find_returning_person()
if self._has_active_lookalike(new_sig, current_track_ids):
    return None, None, 0.0    # block re-match, assign new number
```

**Done-person blacklist:** Appearance signatures of served/no-show entries are stored and
checked against every new entrant to prevent a served customer re-joining with the same number:

```python
def _matches_done_person(self, sig: np.ndarray) -> bool:
    if sig is None or not self._done_appearances:
        return False
    thresh = self.DONE_BLACKLIST_THRESH   # 0.55
    return any(self._compare_sigs(d, sig) >= thresh for d in self._done_appearances)
```

**Manual override:** For identical twins in identical clothing, staff bypass CV entirely:

```python
def force_new_person(self) -> dict:
    with self._lock:
        self._highest_assigned += 1
        num = self._highest_assigned
        self._used_numbers.add(num)
    fake_tid = -(num)
    new_p = QueuePerson(queue_number=num, track_id=fake_tid, bbox=(0, 0, 1, 1))
    new_p.is_manual = True     # never auto-expires, immune to no-show timer
    self.active_queue[fake_tid] = new_p
    return new_p.to_dict()
```

---

## 8. No-Show Detection

**File:** `QueueTracker._check_noshow()`

Each queue entry at position 1 that goes `missing` starts a countdown timer. When
`NOSHOW_WINDOW_SECONDS` (default 300 s) elapses the entry is bumped automatically:

```python
def _check_noshow(self):
    now = datetime.now()
    win = self.NOSHOW_WINDOW_SECONDS

    for tid, p in self.active_queue.items():
        qn = p.queue_number
        if p.position_in_line == 1 and p.status == 'missing':
            if qn not in self._noshow_timers:
                self._noshow_timers[qn] = now          # start countdown
            elif (now - self._noshow_timers[qn]).total_seconds() >= win:
                to_bump.append((tid, p))               # time's up
        else:
            self._noshow_timers.pop(qn, None)          # reset if back

    for tid, p in to_bump:
        completed = p.to_dict()
        completed['bump_reason'] = 'no_show'
        with self._lock:
            self.completed_queue.append(completed)
            self.total_served += 1
        self._register_done_appearance(p)
        p.status = 'done_pending'
        if self.on_noshow:
            self.on_noshow(p.queue_number)
```

Staff warnings are generated in `get_noshow_alerts()`:

```python
def get_noshow_alerts(self) -> list:
    now = datetime.now()
    alerts = []
    for p in self.active_queue.values():
        qn = p.queue_number
        if qn in self._noshow_timers:
            elapsed   = (now - self._noshow_timers[qn]).total_seconds()
            remaining = max(0, self.NOSHOW_WINDOW_SECONDS - elapsed)
            alerts.append({
                'queue_number':      f"Q{qn:03d}",
                'seconds_remaining': int(remaining),
                'status': 'critical' if remaining <= 15 else 'warning',
            })
    return alerts
```

---

## 9. Wait-Time Prediction

**File:** `app/services/prediction_service.py`

Four models run every frame, all consuming a rolling history of up to 60 crowd-count samples.

### 9a. M/M/c Baseline (current snapshot)

```python
def mmch_wait(self, arrival_rate: float, service_rate: float,
              num_counters: int, queue: int) -> float:
    avg_svc = self.cfg['avg_service_time']
    nc      = max(num_counters, 1)
    if arrival_rate <= 0:
        return queue * avg_svc / nc           # no arrivals — pure service time
    rho = arrival_rate / max(nc * service_rate, 1e-9)
    if rho >= 1.0:
        return queue * avg_svc / nc           # saturated system fallback
    return max(0.0, (queue / max(arrival_rate, 0.1)) + avg_svc)
```

`ρ = λ / (c × μ)` is the server utilisation. When ρ ≥ 1 the system is overloaded and
the formula falls back to a simple per-counter division to avoid infinite results.

### 9b. Arrival Rate Estimation

```python
def calculate_arrival_rate(self) -> float:
    n      = min(30, len(historical_counts))
    if n < 2:
        return 0.0
    counts = list(historical_counts)
    dt     = (historical_timestamps[-1] - historical_timestamps[-n]) / 60.0
    return max(0.0, (counts[-1] - counts[-n]) / dt) if dt > 0 else 0.0
```

### 9c. Trend Slope (shared utility)

```python
def _trend_slope(self) -> float:
    counts = list(historical_counts)
    if len(counts) < 5:
        return 0.0
    window = counts[-20:]          # most recent 20 samples
    m = len(window)
    x = np.arange(m, dtype=float)
    x -= x.mean()
    y = np.array(window, dtype=float)
    denom = np.dot(x, x)
    return float(np.dot(x, y) / denom) if denom > 1e-9 else 0.0
```

### 9d. Short-Term Forecast (~5 min) — Linear Trend Projection

Projects the queue 10 minutes forward using the current slope, then re-applies M/M/c:

```python
def predict_short_term(self, base_wait: float, slope: float, nc: int) -> float:
    counts          = list(historical_counts)
    fps             = self._estimated_fps()
    projected_queue = max(0.0, counts[-1] + slope * fps * 600)   # 600 s = 10 min
    avg_svc         = self.cfg['avg_service_time']
    arrival         = self.calculate_arrival_rate()
    if arrival <= 0:
        return max(1.0, projected_queue * avg_svc / max(nc, 1))
    return max(1.0, (projected_queue / max(arrival, 0.1)) + avg_svc)
```

### 9e. Medium-Term Forecast (~15 min) — Holt's Double Exponential Smoothing

Level + trend smoothing with exponential damping to prevent runaway extrapolation:

```python
def predict_medium_term(self, base_wait: float, slope: float, nc: int) -> float:
    counts = list(historical_counts)
    if len(counts) < 4:
        return base_wait

    alpha, beta = 0.4, 0.2
    level = float(counts[0])
    trend = float(counts[1] - counts[0])

    for c in counts[1:]:
        prev  = level
        level = alpha * c + (1 - alpha) * (level + trend)
        trend = beta  * (level - prev)  + (1 - beta)  * trend

    # Damped projection: sum of φ^1 + φ^2 + ... + φ^20  where φ = 0.85
    horizon  = 20
    damping  = 0.85
    damp_sum = sum(damping ** i for i in range(1, horizon + 1))
    forecast_queue = max(0.0, level + trend * damp_sum)

    arrival = self.calculate_arrival_rate()
    if arrival <= 0:
        return max(1.0, forecast_queue * self.cfg['avg_service_time'] / max(nc, 1))
    return max(1.0, (forecast_queue / max(arrival, 0.1)) + self.cfg['avg_service_time'])
```

### 9f. Long-Term Forecast (~30 min) — Growth Ratio + Mean Reversion

```python
def predict_long_term(self, base_wait: float, slope: float, nc: int) -> float:
    counts = list(historical_counts)
    if len(counts) < 8:
        return base_wait

    third        = max(len(counts) // 3, 1)
    early        = np.mean(counts[:third])
    recent       = np.mean(counts[-third:])
    growth_ratio = recent / max(early, 1.0)

    # 70% follow the trend, 30% revert to recent mean
    long_queue = max(0.0, recent * (0.7 * growth_ratio + 0.3))

    arrival = self.calculate_arrival_rate()
    if arrival <= 0:
        raw = long_queue * self.cfg['avg_service_time'] / max(nc, 1)
    else:
        raw = (long_queue / max(arrival, 0.1)) + self.cfg['avg_service_time']

    return max(1.0, min(raw, 60.0))    # cap at 60 min — long forecasts are uncertain
```

### 9g. Main update — called once per video frame

```python
def update(self, count: int, current_data: dict, data_lock) -> None:
    historical_counts.append(count)
    historical_timestamps.append(time.time())

    arrival_rate = self.calculate_arrival_rate()

    with data_lock:
        sr    = current_data['service_rate']
        nc    = current_data['active_counters']
        ew    = self.mmch_wait(arrival_rate, sr, nc, count)
        util  = min(arrival_rate / (nc * sr) if sr > 0 else 0.0, 1.0)
        slope = self._trend_slope()

        current_data.update({
            'arrival_rate':         round(arrival_rate, 2),
            'system_utilization':   round(util, 2),
            'estimated_wait_time':  round(ew, 1),
            'predicted_wait_5min':  round(self.predict_short_term(ew, slope, nc), 1),
            'predicted_wait_15min': round(self.predict_medium_term(ew, slope, nc), 1),
            'predicted_wait_30min': round(self.predict_long_term(ew, slope, nc), 1),
        })
```

---

## 10. Dynamic Service Time Measurement

**File:** `app/database/database_handler.py` — `measure_avg_service_time()`

Instead of relying on a fixed configured value, real service time is measured from the DB
every 5 minutes using consecutive inter-departure gaps between served timestamps:

```python
def measure_avg_service_time(num_counters: int, window_minutes: int = 120,
                              min_samples: int = 5) -> float | None:
    cursor.execute("""
        SELECT served_at FROM queue_records
        WHERE status = 'served'
          AND served_at IS NOT NULL
          AND served_at >= DATE_SUB(NOW(), INTERVAL %s MINUTE)
        ORDER BY served_at ASC
    """, (window_minutes,))

    timestamps = [row["served_at"].timestamp() for row in cursor.fetchall()]

    gaps = []
    for i in range(1, len(timestamps)):
        delta_min = (timestamps[i] - timestamps[i - 1]) / 60.0
        if 0.1 <= delta_min <= 15.0:   # filter noise and idle gaps
            gaps.append(delta_min)

    if len(gaps) < min_samples:
        return None

    # With c parallel counters: inter_departure ≈ service_time / c
    measured = (sum(gaps) / len(gaps)) * max(1, num_counters)
    return round(max(0.5, min(measured, 30.0)), 2)
```

The result is blended with the previous value in `queue_service.py` to prevent sudden jumps:

```python
# In _service_time_refresh_loop() — runs every 300 s in a daemon thread
measured = measure_avg_service_time(num_counters)
if measured is not None:
    old     = QUEUE_CONFIG.get("avg_service_time", 3.0)
    blended = round(0.7 * measured + 0.3 * old, 2)
    QUEUE_CONFIG["avg_service_time"] = blended
```

---

## 11. Counter Assignment and Display Board

**File:** `QueueTracker._recalculate_positions()`

After every frame, the active queue is sorted by queue number and each person is assigned
a position. The first `num_counters` positions also receive a counter number:

```python
def _recalculate_positions(self):
    active_line = sorted(
        (p for p in self.active_queue.values() if p.status in ('waiting', 'missing')),
        key=lambda x: x.queue_number
    )
    newly_called = []
    for i, p in enumerate(active_line):
        p.position_in_line = i + 1
        if (i + 1) <= self._num_counters:
            p.counter_number = i + 1
            if p.queue_number not in self._announced_numbers:
                self._announced_numbers.add(p.queue_number)
                newly_called.append({
                    'queue_number':   p.queue_number,
                    'queue_label':    f"Q{p.queue_number:03d}",
                    'counter_number': i + 1,
                })
        else:
            p.counter_number = None
    self._newly_called = newly_called
```

**TTS announcement (frontend — `QueueDisplayBoard.tsx`):**

When the display board polls and finds a new counter assignment, it queues a speech utterance.
A user-gesture activation overlay is shown on first load to satisfy browser security policy:

```typescript
// Activated once by user click — unlocks Web Speech API
function activateVoice() {
  const unlock = new SpeechSynthesisUtterance(' ');
  unlock.volume = 0;
  unlock.onend = () => {
    setVoiceUnlocked(true);
    setTimeout(speakNext, 200);
  };
  window.speechSynthesis.speak(unlock);
}

// Called whenever data.counter_assignments changes
data.counter_assignments.forEach((p) => {
  const key = `${p.queue_number}-${p.counter_number}`;
  if (announcedRef.current.has(key)) return;
  announcedRef.current.add(key);
  pendingRef.current.push({ queueNumber: p.queue_number, counterNumber: p.counter_number });
});

// Speaks: "Customer number 3, please proceed to Counter 1."
function speakNext() {
  const next = pendingRef.current.shift();
  const utterance = new SpeechSynthesisUtterance(
    `Customer number ${next.queueNumber}, please proceed to Counter ${next.counterNumber}.`
  );
  utterance.rate = 0.85;
  utterance.onend = () => setTimeout(speakNext, 800);
  window.speechSynthesis.speak(utterance);
}
```

---

## Summary Table

| Subsystem | Algorithm | Key Parameter |
|-----------|-----------|---------------|
| Person detection | YOLOv8n + ByteTrack | conf ≥ 0.40 |
| Bbox smoothing | EMA | α = 0.45 |
| Zone membership | Centroid-in-rectangle | — |
| Candidate confirmation | Frame counter buffer | 14 frames |
| Appearance signature | Split-body HSV histogram | 512 values (16×16 × 2 halves) |
| Appearance comparison | Pearson correlation | 0.60 upper + 0.40 lower |
| Re-identification | Appearance + spatial proximity | threshold 0.20 |
| Twin guard | Active-lookalike score | threshold 0.85 |
| Done blacklist | Appearance blacklist | threshold 0.55 |
| Static object filter | Motion energy + confidence bypass | 8 px motion, 0.70 bypass |
| No-show detection | Missing-frame countdown timer | 300 s default |
| Current wait time | M/M/c (Erlang-C) | ρ = λ/(cμ) |
| 5-min forecast | Linear trend projection | 20-sample window |
| 15-min forecast | Holt double-exponential smoothing | α=0.4, β=0.2, φ=0.85 |
| 30-min forecast | Growth-ratio + mean-reversion | 70/30 blend, 60-min cap |
| Service time | Inter-departure gaps from DB | 70/30 blend, 5-min refresh |
| Counter assignment | Sorted position index | first N = num_counters |
| TTS announcement | Web Speech API | client-side, per-session dedup |
