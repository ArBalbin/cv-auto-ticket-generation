# The Face Embedding ("Face Signature")

When a student registers their face, the system saves a list of 512 decimal
numbers in `student_profiles.embedding`. This document explains what those
numbers are, what they do and do not represent, how they are made and used,
and how to answer questions about them.

For the full recognition algorithm see [ALGORITHMS.md §4](ALGORITHMS.md); for
how the match thresholds were chosen see
[FACE_RECOGNITION_CALIBRATION.md](FACE_RECOGNITION_CALIBRATION.md) and
[ACCURACY.md](ACCURACY.md).

---

## 1. What is stored

| Column (`student_profiles`) | Type | Holds |
|---|---|---|
| `embedding` | `json` | 512 decimal numbers: the face embedding |
| `embedding_model_version` | `varchar(50)` | The model that produced it (`buffalo_s`) |
| `num_enrollment_samples` | `int` | How many photos were averaged into it |
| `face_enrolled_at` | `datetime` | When it was saved |

A stored embedding looks like this (illustrative values, not a real student):

```
[0.0213, -0.0587, 0.0041, 0.0932, -0.0178, 0.0466, ... 512 numbers in total]
```

The photos used to make it are **not** stored anywhere.

## 2. What the numbers mean

**No single number has a meaning of its own.** None of them is a measurement
such as eye distance, nose width or face length. The 512 numbers only mean
something together, as one point in a 512-dimensional space.

The numbers come from a neural network that was trained on photos of about
600,000 people for one goal:

- photos of **the same person** should produce points that are **close
  together**, even across different lighting, angles and expressions;
- photos of **different people** should produce points that are **far
  apart**.

The network worked out for itself which facial patterns serve that goal. The
512 numbers are those learned patterns, and they are not designed to be read
by a person.

**Analogy.** A GPS coordinate such as `13.62, 123.19` works the same way. The
`13.62` alone says nothing about a place, but the two numbers together pin
down one exact spot on a map. A face embedding is a coordinate on a "map of
faces" with 512 axes instead of 2. Each student is one spot on that map, and
students who look alike sit closer together.

### Why the values look the way they do

- **They are small, positive and negative.** The vector is
  *L2-normalised*: its total length is exactly 1 (the squares of all 512
  numbers add up to 1). Spread over 512 numbers, almost every value falls
  under about ±0.2. Only the **direction** of the vector carries identity,
  not its size.
- **There are exactly 512.** That is the output size of the model, fixed by
  its designers. 512 is the standard size for this family of face models:
  enough capacity to tell hundreds of thousands of people apart, while
  staying small to store and fast to compare.
- **They are never exactly the same twice.** Two photos of the same person
  give slightly different numbers because light, angle and expression change.
  This is why the system measures how *close* two embeddings are, never
  whether they are *equal*.

## 3. The model that produces them

The embedding comes from **InsightFace `buffalo_s`**, an open-source
library that runs on the backend server, not a cloud face API:

| Step | Model file | Job |
|---|---|---|
| Find the face | `det_500m.onnx` (SCRFD) | Locates the face and five keypoints: eyes, nose tip, mouth corners |
| Describe the face | `w600k_mbf.onnx` (MobileFaceNet, trained with the ArcFace method on WebFace600K) | Turns the aligned face into the 512 numbers |

Both models are **pretrained**. QueuEx did not train the face model; doing so
would need millions of labelled photos. What the project contributes is the
**decision rule** built on top of the embedding (section 5) and its
**calibration** against real faces.

## 4. How a student's embedding is made

1. **Capture (mobile app).** The registration screen takes photos at three
   poses: looking straight, turned to one side, and turned to the other.
   Google ML Kit in the app only checks pose and lighting. It does not
   recognise anyone.
2. **Embed (backend).** For each photo, SCRFD finds the face and uses the
   five keypoints to straighten and crop it to a standard size. MobileFaceNet
   then turns it into 512 numbers. Photos where no face is found with at
   least 0.60 confidence are skipped.
3. **Average.** The per-photo embeddings are averaged and normalised again
   into one canonical embedding:

   ```
   canonical = mean(e1 … en) / ‖mean(e1 … en)‖
   ```

   Averaging across angles makes the stored signature more reliable than any
   single photo.
4. **Duplicate check.** Before saving, the new embedding is matched against
   every other enrolled student. If it clearly matches someone else, the
   registration is refused ("This face is already registered under another
   account"), so one person cannot hold two accounts.
5. **Save.** Only the canonical embedding, the model version and the sample
   count are written to the database. The photos are discarded.

Code: `register_my_face` in `app/routers/students.py`,
`build_enrollment_embedding` in `app/services/face_service.py`.

## 5. How it is used at the queue

1. The camera's detector finds a person, crops the upper part of their body
   box, and produces a live embedding the same way as step 2 above.
2. The live embedding is compared with every enrolled embedding using
   **cosine similarity**: a score from −1 to 1 of how closely two vectors
   point in the same direction. Because every embedding has length 1, this is
   simply the sum of the 512 pairwise products.
3. The match is **accepted only if both** conditions hold:

   ```
   best score ≥ 0.30   AND   best score − second-best score ≥ 0.15
   ```

   The second condition stops the system from choosing between two
   lookalikes. If no one clearly wins, the person is left unrecognised for
   staff to resolve, and the system does not guess.

Code: `EmbeddingIndex.match` in `app/services/embedding_index.py` (all
enrolled embeddings in one matrix, compared in a single operation) and
`match_student` in `app/services/face_service.py`.

## 6. Privacy

- **It is not a photo.** The embedding cannot be opened or viewed as a
  picture, and the system keeps no face images.
- **It is still treated as biometric personal data.** Published research
  shows that another AI model can produce an *approximate* face image from an
  embedding (a "model inversion" attack). So the system does **not** claim the
  embedding is impossible to reverse. It treats the embedding as sensitive
  personal information under the Data Privacy Act (RA 10173), and protects
  it accordingly:
  - **Consent first.** The privacy notice must be accepted before an account
    is created or a face is registered.
  - **Erased on archive.** After one year without use, the account is
    archived and its embedding is set to `NULL`
    (`app/services/retention_service.py`).
  - **Deleted on request.** "Delete my account" in the app removes the
    student and their embedding.
- **Tied to the model.** An embedding is only meaningful to the model that
  produced it. If the model changes, existing embeddings stop matching and
  students must register again. This is why `embedding_model_version` is
  stored.

## 7. Size and scale

| Measure | Value |
|---|---|
| One embedding, as JSON text in the database | about 11 KB |
| One embedding, in memory for matching (512 × float32) | 2 KB |
| 10,000 students, database | about 110 MB |
| 10,000 students, matching index in memory | 20.5 MB |
| Matching one face against 10,000 students | 3.2 ms (p50), 4.6 ms (p95) |

Source: `load_testing/results/face_match_scaling.csv`, produced by
`load_testing/bench_face_match.py`.

---

## 8. Questions the panel may ask

**What does each number represent?**
Nothing on its own. The numbers are patterns the network learned during
training. Only all 512 together describe a face, the way latitude and
longitude together describe a location.

**Why 512 numbers?**
It is the output size of the pretrained model. It is the standard size for
this family of face models: enough to separate hundreds of thousands of
people, yet small to store and fast to compare.

**Is the face signature unique to each person?**
Each person's embedding is different, and registration refuses a face that
already matches another account. The numbers are not fixed, though: the same
person produces slightly different numbers in every photo. Identity is
decided by how close two embeddings are, with a minimum score and a lead over
the runner-up.

**How do you know two embeddings are the same person?**
Cosine similarity. The match is accepted only if the best score is at least
0.30 and beats the second-best by at least 0.15. Both values were calibrated
on real face photos (see `FACE_RECOGNITION_CALIBRATION.md`).

**Did you train the face model?**
No. It is a pretrained open-source model (InsightFace `buffalo_s`). Training
one needs millions of labelled photos. The project's work is the decision
rule and its calibration.

**Can someone rebuild a student's face from these numbers?**
Not directly, since it is not an image. Research has shown that an
approximate face can be generated from an embedding, so the system treats it
as biometric personal data: consent before collection, erasure on archive,
and deletion on request.

**What happens if the database is leaked?**
No photos would be exposed, because none are stored. The embeddings are only
usable with the same model, but they are still personal data, which is why
they are minimised (one per student), erased after a year of inactivity, and
deletable by the student.

**Why not store the photos instead?**
Storing the photos would be more sensitive and much larger, and recognition
does not need them. Matching works entirely on embeddings.
