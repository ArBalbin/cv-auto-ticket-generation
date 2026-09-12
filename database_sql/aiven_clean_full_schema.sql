
USE defaultdb;

SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS counter_config_history;
DROP TABLE IF EXISTS queue_events;
DROP TABLE IF EXISTS face_match_events;
DROP TABLE IF EXISTS crowd_snapshots;
DROP TABLE IF EXISTS queue_records;
DROP TABLE IF EXISTS student_profiles;
DROP TABLE IF EXISTS users;

SET FOREIGN_KEY_CHECKS = 1;

CREATE TABLE users (
    id         INT          AUTO_INCREMENT PRIMARY KEY,
    username   VARCHAR(255) UNIQUE NOT NULL,
    password   VARCHAR(255) NOT NULL,
    email      VARCHAR(255) DEFAULT NULL,
    full_name  VARCHAR(255) DEFAULT NULL,
    role       VARCHAR(50)  DEFAULT 'staff',
    created_at TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP    NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
    last_login TIMESTAMP    NULL,
    is_active  BOOLEAN      DEFAULT TRUE,
    INDEX idx_username (username),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Student self-registration + face-recognition enrollment. Accounts are
-- created via self-service Sign-in-with-Google against the student's NCF
-- Gbox account (no password is ever stored here — Google is the credential
-- authority; google_sub is Google's stable per-account identifier, gbox_email
-- is display/lookup convenience). school_id is the student's school ID
-- number, entered once at sign-up to link the account to their school
-- record. Face embedding is captured as a SECOND step after account
-- creation, so embedding/embedding_model_version start out NULL — a row
-- with embedding IS NULL means "account exists, face not yet registered."
-- Only a derived embedding vector is ever persisted — never a face image —
-- mirroring the system's existing "no raw video stored" privacy stance,
-- extended to biometric data (Data Privacy Act consideration).
CREATE TABLE student_profiles (
    id                      INT          AUTO_INCREMENT PRIMARY KEY,
    school_id               VARCHAR(50)  NOT NULL,
    gbox_email               VARCHAR(255) DEFAULT NULL,
    google_sub                VARCHAR(255) DEFAULT NULL,
    full_name                VARCHAR(255) DEFAULT NULL,
    embedding                 JSON         DEFAULT NULL,
    embedding_model_version   VARCHAR(50)  DEFAULT NULL,
    num_enrollment_samples    INT          NOT NULL DEFAULT 0,
    enrolled_by_user_id      INT          DEFAULT NULL,
    is_active                 BOOLEAN      NOT NULL DEFAULT TRUE,
    enrolled_at               DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login                DATETIME     DEFAULT NULL,
    face_enrolled_at          DATETIME     DEFAULT NULL,
    updated_at                DATETIME     NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_school_id (school_id),
    UNIQUE KEY uq_gbox_email (gbox_email),
    UNIQUE KEY uq_google_sub (google_sub),
    INDEX idx_is_active (is_active),
    CONSTRAINT fk_student_profiles_enrolled_by
        FOREIGN KEY (enrolled_by_user_id) REFERENCES users(id)
        ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE queue_records (
    id                 INT          AUTO_INCREMENT PRIMARY KEY,
    service_date       DATE         NOT NULL,
    queue_number       INT          NOT NULL,
    short_code         VARCHAR(10)  DEFAULT NULL,
    jwt_token          TEXT         DEFAULT NULL,
    pdf_path           VARCHAR(512) DEFAULT NULL,
    created_by_user_id INT          DEFAULT NULL,
    served_by_user_id  INT          DEFAULT NULL,
    student_id         INT          DEFAULT NULL,
    is_walkin          BOOLEAN      NOT NULL DEFAULT TRUE,
    linked_via         ENUM('face_ocr','manual','face_only') NOT NULL DEFAULT 'manual',
    linked_at          DATETIME     DEFAULT NULL,
    status             ENUM('waiting','served','no_show','expired')
                                      NOT NULL DEFAULT 'waiting',
    expires_at         DATETIME     DEFAULT NULL,
    created_at         DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    served_at          DATETIME     DEFAULT NULL,
    INDEX idx_queue_number (queue_number),
    INDEX idx_service_date_queue (service_date, queue_number),
    INDEX idx_short_code (short_code),
    INDEX idx_status (status),
    INDEX idx_expires_at (expires_at),
    INDEX idx_queue_status_created (queue_number, status, created_at),
    INDEX idx_service_date_status (service_date, status),
    INDEX idx_created_at (created_at),
    INDEX idx_created_by_user_id (created_by_user_id),
    INDEX idx_served_by_user_id (served_by_user_id),
    INDEX idx_student_id (student_id),
    CONSTRAINT fk_queue_records_created_by
        FOREIGN KEY (created_by_user_id) REFERENCES users(id)
        ON DELETE SET NULL,
    CONSTRAINT fk_queue_records_served_by
        FOREIGN KEY (served_by_user_id) REFERENCES users(id)
        ON DELETE SET NULL,
    CONSTRAINT fk_queue_records_student
        FOREIGN KEY (student_id) REFERENCES student_profiles(id)
        ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE crowd_snapshots (
    id               INT      AUTO_INCREMENT PRIMARY KEY,
    recorded_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    person_count     INT      NOT NULL DEFAULT 0,
    avg_density      FLOAT    NOT NULL DEFAULT 0.0,
    max_density      FLOAT    NOT NULL DEFAULT 0.0,
    queue_length     INT      NOT NULL DEFAULT 0,
    active_counters  INT      NOT NULL DEFAULT 3,
    est_wait_minutes FLOAT    NOT NULL DEFAULT 0.0,
    INDEX idx_recorded_at (recorded_at),
    INDEX idx_recorded_at_counts (recorded_at, person_count, queue_length)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE queue_events (
    id              INT          AUTO_INCREMENT PRIMARY KEY,
    queue_record_id INT          DEFAULT NULL,
    service_date    DATE         DEFAULT NULL,
    queue_number    INT          DEFAULT NULL,
    event_type      ENUM('created','served','no_show','expired','reset')
                                  NOT NULL,
    actor_user_id   INT          DEFAULT NULL,
    event_note      VARCHAR(255) DEFAULT NULL,
    created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_queue_record_id (queue_record_id),
    INDEX idx_queue_number_created (queue_number, created_at),
    INDEX idx_event_type_created (event_type, created_at),
    INDEX idx_actor_user_id (actor_user_id),
    CONSTRAINT fk_queue_events_record
        FOREIGN KEY (queue_record_id) REFERENCES queue_records(id)
        ON DELETE SET NULL,
    CONSTRAINT fk_queue_events_actor
        FOREIGN KEY (actor_user_id) REFERENCES users(id)
        ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE counter_config_history (
    id                 INT      AUTO_INCREMENT PRIMARY KEY,
    old_counters       INT      DEFAULT NULL,
    new_counters       INT      NOT NULL,
    avg_service_time   FLOAT    NOT NULL DEFAULT 3.0,
    changed_by_user_id INT      DEFAULT NULL,
    created_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_created_at (created_at),
    INDEX idx_changed_by_user_id (changed_by_user_id),
    CONSTRAINT fk_counter_config_actor
        FOREIGN KEY (changed_by_user_id) REFERENCES users(id)
        ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Audit trail for face-match attempts (accepted and rejected). No images stored,
-- only scores, so this can be retained longer than any raw biometric capture.
CREATE TABLE face_match_events (
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

INSERT INTO users (username, password, email, full_name, role)
VALUES (
    'admin',
    'scrypt:32768:8:1$8BsPvqaskARQ0wBc$dfa77a503bd996658379bf49c582daaba735ffb374aad67194d0c3160cfacdd9439c6c9dc6a641e578654ad093080167cf39036afee9cbf407ec32c98f367a68',
    'admin@crowd-monitoring.com',
    'System Administrator',
    'admin'
);

SELECT
    TABLE_NAME,
    COLUMN_NAME,
    CONSTRAINT_NAME,
    REFERENCED_TABLE_NAME,
    REFERENCED_COLUMN_NAME
FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
WHERE TABLE_SCHEMA = DATABASE()
  AND REFERENCED_TABLE_NAME IS NOT NULL
ORDER BY TABLE_NAME, COLUMN_NAME;

SHOW TABLES;
