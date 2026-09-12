-- QueueFlow face-recognition revision migration.
--
-- Run this once against an EXISTING database that was created from an earlier
-- version of schema_cloud_ready.sql / aiven_clean_full_schema.sql (i.e. before
-- student_profiles / face_match_events existed). Fresh installs should use the
-- updated schema files instead — this script only exists to bring an already
-- running database up to date without dropping anything.
--
-- Safe to run once. Re-running will fail on the ALTER TABLE statements (columns
-- already exist) — that failure is expected and can be ignored if this script
-- was already applied.

CREATE TABLE IF NOT EXISTS student_profiles (
    id                      INT           AUTO_INCREMENT PRIMARY KEY,
    school_id               VARCHAR(50)   NOT NULL,
    full_name                VARCHAR(255)  DEFAULT NULL,
    embedding                 JSON          NOT NULL,
    embedding_model_version   VARCHAR(50)   NOT NULL,
    num_enrollment_samples    INT           NOT NULL DEFAULT 1,
    enrolled_by_user_id      INT           DEFAULT NULL,
    is_active                 BOOLEAN       NOT NULL DEFAULT TRUE,
    enrolled_at               DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at                DATETIME      NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_school_id (school_id),
    INDEX idx_is_active (is_active),
    CONSTRAINT fk_student_profiles_enrolled_by
        FOREIGN KEY (enrolled_by_user_id) REFERENCES users(id)
        ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

ALTER TABLE queue_records
    ADD COLUMN student_id INT DEFAULT NULL AFTER served_by_user_id,
    ADD COLUMN is_walkin  BOOLEAN NOT NULL DEFAULT TRUE AFTER student_id,
    ADD COLUMN linked_via ENUM('face_ocr','manual') NOT NULL DEFAULT 'manual' AFTER is_walkin,
    ADD COLUMN linked_at  DATETIME DEFAULT NULL AFTER linked_via,
    ADD INDEX idx_student_id (student_id),
    ADD CONSTRAINT fk_queue_records_student
        FOREIGN KEY (student_id) REFERENCES student_profiles(id)
        ON DELETE SET NULL;

CREATE TABLE IF NOT EXISTS face_match_events (
    id            INT       AUTO_INCREMENT PRIMARY KEY,
    student_id    INT       DEFAULT NULL,
    track_id      INT       DEFAULT NULL,
    matched_score FLOAT     DEFAULT NULL,
    margin        FLOAT     DEFAULT NULL,
    accepted      BOOLEAN   NOT NULL,
    created_at    DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_created_at (created_at),
    INDEX idx_student_id (student_id),
    CONSTRAINT fk_face_match_student
        FOREIGN KEY (student_id) REFERENCES student_profiles(id)
        ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
