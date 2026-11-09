# QueuEx Database Entities and Relationships

This document explains the database structure used by QueuEx.

Last updated: September 12, 2026
Verified directly against the live schema (`SHOW COLUMNS` plus
`information_schema.KEY_COLUMN_USAGE` for every foreign key), not against the
migration files — migrations record what was *intended*, the live schema
records what is actually there.

The active schema for new databases is `database_sql/schema_cloud_ready.sql`.
For a full Aiven reset during development, use
`database_sql/aiven_clean_full_schema.sql`. Three later migrations extend it:

| Migration | Adds |
|---|---|
| `2026_08_face_recognition_migration.sql` | `student_profiles`, `face_match_events` |
| `2026_09_student_self_registration_migration.sql` | Google sign-in columns on `student_profiles` |
| `2026_09_face_only_ticket_migration.sql` | `student_id`, `linked_via`, `is_walkin` on `queue_records` |
| `2026_09_recognition_metrics_migration.sql` | `recognition_metrics` |

## Overview

Eight tables in three groups:

- **Identity** — `users` (staff), `student_profiles` (enrolled students)
- **Queue operations** — `queue_records`, `queue_events`,
  `counter_config_history`
- **Measurement** — `face_match_events`, `recognition_metrics`,
  `crowd_snapshots`

The measurement group exists because the thesis has to report accuracy on real
use, not only on a photo dataset. Those three tables are the source for
`ML/report_recognition_metrics.py`.

---

## Identity

### users

Staff accounts that can log in to the dashboard.

- `id` — primary key
- `username` — unique staff login name
- `password` — scrypt-hashed, never plaintext
- `email`, `full_name`, `role` — profile and role label
- `is_active` — account status
- `created_at`, `updated_at`, `last_login`

### student_profiles

Enrolled students. One row per student, created when they sign in through the
mobile app and completed when they capture their face.

- `id` — primary key
- `school_id` — unique; the student's institutional ID number
- `gbox_email` — unique; their Gbox (institutional Google) address
- `google_sub` — unique; Google's stable account identifier, the value actually
  trusted for sign-in (an email address can be reassigned, `sub` cannot)
- `full_name`
- `embedding` — JSON array of 512 floats, the averaged ArcFace enrollment
  embedding
- `embedding_model_version` — which model produced it, so a future model change
  can identify rows that need re-enrolment rather than silently comparing
  embeddings from two different models
- `num_enrollment_samples` — how many capture photos were averaged
- `enrolled_by_user_id` — staff member, if enrolment was staff-assisted
- `is_active`, `enrolled_at`, `last_login`, `face_enrolled_at`, `updated_at`

> **Privacy — state this in the thesis.** `embedding` holds numbers, not an
> image. No student photograph is stored anywhere in this database. The
> embedding is a one-way representation: it supports comparison against another
> embedding, but the original face cannot be reconstructed from it. Enrolment
> photos exist only in the phone's memory during capture and are never
> transmitted or written to disk.

Relationship:

- `student_profiles.enrolled_by_user_id` references `users.id`

---

## Queue operations

### queue_records

Every issued queue ticket.

- `id` — primary key
- `service_date` — the date the queue number belongs to
- `queue_number` — the number shown to the student
- `short_code` — the printed access code (`XXXX-XXXX`), used to authenticate
  status lookups
- `jwt_token` — signed token backing the QR link
- `pdf_path` — local PDF path or object-storage URL
- `student_id` — the recognised student, when the ticket came from a face match
- `is_walkin` — true when staff entered the number manually
- `linked_via` — how the ticket became attached to a person
- `linked_at` — when that link happened
- `status` — `waiting`, `served`, `no_show`, or `expired`
- `expires_at` — ticket expiry (4 hours after issue)
- `created_by_user_id`, `served_by_user_id` — staff attribution
- `created_at`, `served_at`

**`linked_via` values.** The column is an enum of `face_ocr`, `manual`,
`face_only`. Only two are reachable now:

| Value | Meaning | Still produced? |
|---|---|---|
| `face_only` | Face recognition minted the number | Yes — the normal path |
| `manual` | Staff typed a walk-in number | Yes |
| `face_ocr` | Face match **plus** a digit CNN reading the number off a printed kiosk ticket | **No** — dead value |

`face_ocr` is a remnant of the earlier design, in which the system read an
existing kiosk ticket instead of issuing its own number. That OCR path and its
digit CNN were deleted when the reissued SOP required the system to assign
numbers itself. The enum value is left in place so historical rows stay
readable; no new row can receive it.

Relationships:

- `queue_records.created_by_user_id` references `users.id`
- `queue_records.served_by_user_id` references `users.id`
- `queue_records.student_id` references `student_profiles.id`

### queue_events

Audit trail of queue actions.

- `id` — primary key
- `queue_record_id` — the related ticket
- `service_date`, `queue_number` — denormalised for reporting, so a query does
  not need to join back for the common case
- `event_type` — `created`, `served`, `no_show`, `expired`, or `reset`
- `actor_user_id` — the staff member who acted, if any
- `event_note` — short description
- `created_at`

Relationships:

- `queue_events.queue_record_id` references `queue_records.id`
- `queue_events.actor_user_id` references `users.id`

### counter_config_history

History of how many service counters were open.

- `id` — primary key
- `old_counters`, `new_counters`
- `avg_service_time` — the service-time value in effect at the change
- `changed_by_user_id`
- `created_at`

This feeds the wait-time model: the M/M/c baseline needs to know how many
servers were active during a period, and that changes during a shift.

Relationship:

- `counter_config_history.changed_by_user_id` references `users.id`

---

## Measurement

### face_match_events

One row per face-match attempt, accepted or not.

- `id` — primary key
- `student_id` — the best-matching student
- `track_id` — the camera track the attempt belongs to
- `matched_score` — cosine similarity of the best match
- `margin` — how far the best match beat the runner-up
- `accepted` — whether both thresholds were cleared
- `created_at`

Recording **rejected** attempts is the point. A rejection means the system saw
a face and declined to guess, which is the designed safe outcome; without these
rows there is no way to show how often that safeguard fires, or to justify the
threshold and margin values from live data rather than from photos alone.

Relationship:

- `face_match_events.student_id` references `student_profiles.id`

### recognition_metrics

One row per successful link — the end-to-end story of a single recognition.

- `id` — primary key
- `track_id`, `student_id`, `queue_number`, `linked_via`
- `first_seen_at` — camera first saw this person
- `confirmed_at` — presence confirmed (anti-ghost filter passed)
- `linked_at` — identity resolved and number issued
- `seconds_to_confirm`, `seconds_to_link` — the two durations a student
  actually experiences
- `embed_attempts` — frames containing this person
- `embed_successes` — frames that yielded a usable face
- `match_score`, `match_margin`
- `created_at`

`embed_attempts` and `embed_successes` together give the **live face-detection
rate**, which is the honest counterpart to the photo-dataset accuracy in
`ACCURACY.md` Part A: a missed detection and a rejected match are different
failures with different fixes, and only this pair separates them.

No foreign key on `student_id` here: these rows are evaluation history and must
survive a student record being removed.

### crowd_snapshots

Time-series analytics of the queue area.

- `id` — primary key
- `recorded_at`
- `person_count`, `avg_density`, `max_density`
- `queue_length`, `active_counters`, `est_wait_minutes`

No foreign key to a ticket: one snapshot describes the whole area at a moment,
not one person.

---

## ERD Summary

```text
users
  1 ──< queue_records.created_by_user_id
  1 ──< queue_records.served_by_user_id
  1 ──< queue_events.actor_user_id
  1 ──< counter_config_history.changed_by_user_id
  1 ──< student_profiles.enrolled_by_user_id

student_profiles
  1 ──< queue_records.student_id
  1 ──< face_match_events.student_id

queue_records
  1 ──< queue_events.queue_record_id

recognition_metrics    evaluation history, intentionally unlinked
crowd_snapshots        area time-series, intentionally unlinked
```

## Data flow with relationships

1. A student signs in on the mobile app → a row is created in
   `student_profiles` (Google `sub`, Gbox email, school ID).
2. They capture their face → `embedding` and `face_enrolled_at` are filled in.
3. At the queue area they tap **Join the Queue**, which arms their intent for a
   short window. Without this, recognition alone issues nothing.
4. The camera confirms their presence, then matches their face. Every attempt
   writes a row to `face_match_events`, accepted or rejected.
5. On an accepted match, the system mints a queue number and writes
   `queue_records` with `linked_via = 'face_only'` and `student_id` set, plus a
   `created` row in `queue_events`.
6. The full timing of that recognition is written to `recognition_metrics`.
7. Staff sign in using a `users` account and mark the number done:
   `queue_records.status` becomes `served`, `served_by_user_id` and `served_at`
   are stamped, and a `served` row is added to `queue_events`.
8. Walk-ins skip steps 1–6 entirely: staff type the number, producing a
   `queue_records` row with `is_walkin = 1` and `linked_via = 'manual'`.
9. Counter changes go to `counter_config_history`; area analytics go to
   `crowd_snapshots`.

## What the relationships let the system answer

- Which staff member served a ticket, and when?
- Which student was issued which number, and how was the link made — face
  recognition or manual entry?
- How often does face matching refuse rather than guess, and at what scores?
- How long does recognition take from first sighting to issued ticket?
- What fraction of frames containing a person actually yield a usable face?
- How did counter settings and crowd conditions change over a shift?

The last four are what make an evaluation chapter possible. They were added
specifically so accuracy could be reported from real operation instead of
estimated.
