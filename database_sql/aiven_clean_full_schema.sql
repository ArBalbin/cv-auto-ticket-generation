-- QueueFlow clean Aiven schema.
--
-- Use this when you want to reset the Aiven database and recreate all
-- QueueFlow tables cleanly.
--
-- WARNING:
-- This deletes existing QueueFlow data in defaultdb.

USE defaultdb;

SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS counter_config_history;
DROP TABLE IF EXISTS queue_events;
DROP TABLE IF EXISTS crowd_snapshots;
DROP TABLE IF EXISTS queue_records;
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

CREATE TABLE queue_records (
    id                 INT          AUTO_INCREMENT PRIMARY KEY,
    service_date       DATE         NOT NULL,
    queue_number       INT          NOT NULL,
    short_code         VARCHAR(10)  DEFAULT NULL,
    jwt_token          TEXT         DEFAULT NULL,
    pdf_path           VARCHAR(512) DEFAULT NULL,
    created_by_user_id INT          DEFAULT NULL,
    served_by_user_id  INT          DEFAULT NULL,
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
    INDEX idx_created_by_user_id (created_by_user_id),
    INDEX idx_served_by_user_id (served_by_user_id),
    CONSTRAINT fk_queue_records_created_by
        FOREIGN KEY (created_by_user_id) REFERENCES users(id)
        ON DELETE SET NULL,
    CONSTRAINT fk_queue_records_served_by
        FOREIGN KEY (served_by_user_id) REFERENCES users(id)
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
    INDEX idx_recorded_at (recorded_at)
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
