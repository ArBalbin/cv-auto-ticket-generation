# Face-Recognition Threshold Calibration

This document records the calibration of `FACE_MATCH_THRESHOLD` and
`FACE_MARGIN_THRESHOLD` (Algorithm 4 — Face Recognition Identity Validation)
against real captured face photos, replacing the untested placeholder
defaults that shipped with the initial implementation.

**Date:** 2026-08
**Tool:** `ML/calibrate_face_recognition.py`
**Result:** `FACE_MATCH_THRESHOLD` updated from 0.45 (placeholder) → **0.30**
(calibrated). `FACE_MARGIN_THRESHOLD` stayed at 0.10 — already well-supported
by this data.

> **This document records the August calibration as it happened. The margin
> conclusion was later overturned** — see §6 and `ACCURACY.md`. The deployed
> values today are `FACE_MATCH_THRESHOLD = 0.30` and
> `FACE_MARGIN_THRESHOLD = 0.15`.

---

## 1. Why this calibration was necessary

QueueFlow's face recognition does not train a neural network — it uses a
pretrained ArcFace model (InsightFace `buffalo_s`) to produce a 512-value
embedding per face, and cosine similarity to compare two embeddings. What
*is* original to this system is the **decision rule** built on top of that
model: a live face is only linked to an enrolled student if the best match
clears an absolute similarity threshold **and** beats the second-best
candidate by a safety margin — otherwise the system refuses to guess and
escalates to staff.

Retraining or fine-tuning the embedding model itself would require a
dataset of a scale (millions of images, tens of thousands of identities)
completely impractical for this project — using a peer-reviewed pretrained
model is the correct, standard engineering choice, not a shortcut. What
*does* need real data is the **threshold**: the values that decide where
"confidently the same person" ends and "not confident enough" begins are
specific to a deployment's population, cameras, and lighting, and cannot be
guessed correctly from first principles. Before this calibration, both
thresholds were placeholder values, untested against any real face.

## 2. Methodology

`ML/calibrate_face_recognition.py` mirrors the exact comparison the live
system performs (`face_service.match_student()`), rather than approximating
it:

1. For each enrolled person, their **enrollment photos** are averaged into
   one canonical embedding — identical to how `POST /api/students/me/face`
   builds a real student's profile (`face_service.build_enrollment_embedding`).
2. For each of that person's held-out **verification photos**, a fresh
   embedding is computed and compared against *every* enrolled person's
   canonical embedding (not just their own) — identical to how a live
   camera capture is matched against the full roster of enrolled students.
3. Two numbers are recorded per verification photo: the **genuine score**
   (similarity against the verification photo's own true person) and the
   **worst-case impostor score** (the highest similarity against any *other*
   enrolled person).
4. A grid sweep evaluates every `(match_threshold, margin_threshold)`
   combination from 0.20–0.70 (match) × 0.00–0.20 (margin) against all
   trials, counting: correctly linked, wrongly linked (a different
   identity — the dangerous outcome), and safely rejected (stays pending,
   escalates to staff — the safe fallback outcome).

This is the metric that matters for choosing a threshold, because it
reproduces the actual accept/reject decision the deployed system makes,
trial by trial.

A second, stricter recommender (`recommend_from_separation`) also compares
the single worst genuine score against the single worst impostor score
*across the whole dataset*. This is intentionally conservative, but as
section 4 explains, it produced a misleading warning on real data and should
be read as a sanity check, not the primary result.

## 3. Dataset

| | |
|---|---|
| People | 16 |
| Total photos | ~120 |
| Verification trials | 43 |
| Source | A classmate's face-dataset collection (multiple photos per person, varied angle/lighting/background) plus 3 photos of the researcher himself |
| Enrollment/verification split | Explicit `enroll_*`/`verify_*` filename prefixes where present; automatic 70/30 split by file order otherwise |

**This is real evidence, not the final word.** These are 16 people from one
photo collection, not actual NCF students captured at the actual deployment
camera and lighting. The threshold below should be re-validated (rerun the
same script) once real enrolled-student photos and real queue-zone footage
are available, and updated again if the population or setup differs
meaningfully.

## 4. Finding: a contaminated data point, and why it mattered

The first calibration run flagged an anomaly: one verification photo (of a
person we'll call Person A) scored only 0.126 against their *own* enrollment
embedding — low enough to overlap with impostor scores from unrelated
people. Inspecting the actual image file identified the cause: it was not a
genuine photo at all, but a screenshot of a TikTok-style AI face filter,
explicitly watermarked *"AI-generated content, for entertainment only."* It
had been placed in the dataset folder as if it were a real capture.

After excluding that one file and re-running the calibration:

- Genuine score range: **0.321 – 0.926**
- Worst-case impostor score range: **0.125 – 0.353**
- Correct identification: **43/43 (100%)**
- Wrong-identity accepts: **0**, at every threshold from 0.20 to 0.70

The raw ranges still technically overlap (0.321 vs. 0.353), which is why
`recommend_from_separation()` still printed an overlap warning. But that
comparison is between the worst genuine score and worst impostor score from
**two different, unrelated trials** — it does not mean any single trial was
ever actually ambiguous. Per-trial, the hardest real case (a person we'll
call Person B, genuine score 0.321) still beat every impostor by a margin
of 0.027 — a thin margin, but a real one, and it was never wrong. This is
why the grid-sweep result (0 wrong-accepts at every threshold tested) is the
number to trust here, not the separation check's conservative warning.

**Lesson for the methodology write-up:** a single mislabeled or synthetic
photo can meaningfully distort a small calibration dataset, and the
"overlap" safety check — while a reasonable worst-case guard — can produce
false alarms on otherwise-clean real data. Both are worth stating plainly:
it shows the calibration process was interrogated rather than taken at face
value.

## 5. Full grid (clean data, contaminated photo excluded)

| match_thr | margin_thr | correct | wrong-accept | rejected |
|---|---|---|---|---|
| 0.20 | 0.00 | 43 | 0 | 0 |
| 0.20 | 0.10 | 42 | 0 | 1 |
| 0.30 | 0.00 | 43 | 0 | 0 |
| 0.30 | 0.10 | 42 | 0 | 1 |
| 0.40 | 0.10 | 42 | 0 | 1 |
| 0.45 | 0.10 | 40 | 0 | 3 |
| 0.50 | 0.10 | 39 | 0 | 4 |
| 0.55 | 0.10 | 37 | 0 | 6 |
| 0.60 | 0.10 | 33 | 0 | 10 |
| 0.65 | 0.10 | 30 | 0 | 13 |
| 0.70 | 0.10 | 27 | 0 | 16 |

**Wrong-accept count is 0 at every single threshold tested**, from the
loosest (0.20) to the strictest (0.70). The only effect of raising the
threshold is trading correctly-auto-linked cases for safely-escalated ones
— exactly the intended fail-safe behavior, never a wrong identity.

Full raw trial data: `calibration_data/results_clean.json` (gitignored —
contains derived scores from real people's photos, not committed).

## 6. Recommendation and what was applied

**`FACE_MATCH_THRESHOLD = 0.30`** — comfortably below the lowest genuine
score observed (0.321), leaving margin for photos slightly worse than this
calibration set, while still well above typical unrelated-face similarity.
Applied to `app/core/config.py`, replacing the placeholder 0.45.

**`FACE_MARGIN_THRESHOLD = 0.10`** — unchanged from its placeholder value,
but now genuinely supported: real data shows it costs only 1 of 43 genuine
matches (safely rejected, not wrongly accepted) relative to the loosest
possible margin.

> **SUPERSEDED (2026-09).** The margin was later raised to **0.15**. The
> conclusion above was drawn from a closed-set test only — every probe came
> from someone already enrolled, so the question "does a NON-enrolled person
> get accepted?" was never asked. `ML/evaluate_face_accuracy.py` added that
> open-set test and found one stranger accepted as an enrolled student with
> a margin of 0.110, clearing 0.10 by a hundredth. Raising the margin to
> 0.15 removes it at no cost to genuine matches. See `ACCURACY.md`.
>
> Two data problems also skewed the original run and are fixed:
> `.jfif` files were excluded by the image-extension filter, silently
> dropping one enrolled person (and with them the very identity involved in
> the false accept); and only the single worst impostor score per trial was
> retained, which cannot produce a real False Accept Rate.

## 7. Reproducing this calibration

```
python ML/calibrate_face_recognition.py --data calibration_data/faces --out calibration_data/results.json
```

Expected data layout — one folder per person, at least 2 people:

```
calibration_data/faces/<person_id>/enroll_1.jpg
calibration_data/faces/<person_id>/enroll_2.jpg
calibration_data/faces/<person_id>/verify_1.jpg
calibration_data/faces/<person_id>/verify_2.jpg
```

See the script's own docstring for the automatic 70/30 fallback split when
`enroll_`/`verify_` filename prefixes aren't used.

## 8. Next steps

- [ ] Recalibrate against real enrolled NCF students once available.
- [ ] Recalibrate at the actual deployment camera and lighting conditions.
- [ ] Screen any future calibration dataset for non-genuine photos (filters,
      screenshots, edited images) before trusting results — as this run
      demonstrated, a single bad file can distort a small dataset's
      apparent separation.
- [ ] Delete `calibration_data/faces/` once no longer needed locally — it
      holds copies of other people's face photos (gitignored, never
      committed, but no reason to keep duplicating them longer than
      necessary).
