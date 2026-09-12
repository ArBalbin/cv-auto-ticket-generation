-- QueueFlow student self-registration revision migration.
--
-- Run this once against an EXISTING database that was created from an earlier
-- version of schema_cloud_ready.sql / aiven_clean_full_schema.sql (i.e. before
-- self-service Gbox sign-up existed and student_profiles only supported
-- staff-operated enrollment). Fresh installs should use the updated schema
-- files instead — this script only exists to bring an already running
-- database up to date without dropping anything.
--
-- Context: per panel revision, face-recognition registration is self-service
-- (student ID + NCF Gbox Google account), not staff-operated. Account
-- creation now happens BEFORE face capture (sign up, then register your
-- face as a second step), so embedding/embedding_model_version can no
-- longer be NOT NULL.
--
-- Safe to run once. Re-running will fail on columns/keys that already exist
-- — that failure is expected and can be ignored if this script was already
-- applied.

ALTER TABLE student_profiles
    ADD COLUMN gbox_email       VARCHAR(255) DEFAULT NULL AFTER school_id,
    ADD COLUMN google_sub       VARCHAR(255) DEFAULT NULL AFTER gbox_email,
    ADD COLUMN last_login       DATETIME     DEFAULT NULL AFTER enrolled_at,
    ADD COLUMN face_enrolled_at DATETIME     DEFAULT NULL AFTER last_login;

ALTER TABLE student_profiles
    MODIFY COLUMN embedding               JSON        DEFAULT NULL,
    MODIFY COLUMN embedding_model_version VARCHAR(50) DEFAULT NULL,
    MODIFY COLUMN num_enrollment_samples  INT         NOT NULL DEFAULT 0;

ALTER TABLE student_profiles
    ADD UNIQUE KEY uq_gbox_email (gbox_email),
    ADD UNIQUE KEY uq_google_sub (google_sub);

-- linked_via stays face_ocr/manual only — QueueFlow never mints its own
-- queue number (a face match with no ticket-number read just stays
-- pending until staff resolve it via /api/queue/link-pending), per the
-- panel's directive to leave the kiosk's queue flow untouched.
ALTER TABLE queue_records
    MODIFY COLUMN linked_via ENUM('face_ocr', 'manual')
                              NOT NULL DEFAULT 'manual';
