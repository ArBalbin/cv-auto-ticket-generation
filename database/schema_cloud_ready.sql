-- QueueFlow cloud-ready schema.
--
-- Use this for a fresh cloud database. It intentionally does not DROP databases,
-- DROP tables, or CREATE MySQL users because many cloud providers manage those
-- outside the imported schema.

CREATE TABLE IF NOT EXISTS users (
    id         INT           AUTO_INCREMENT PRIMARY KEY,
    username   VARCHAR(255)  UNIQUE NOT NULL,
    password   VARCHAR(255)  NOT NULL,
    email      VARCHAR(255)  DEFAULT NULL,
    full_name  VARCHAR(255)  DEFAULT NULL,
    role       VARCHAR(50)   DEFAULT 'staff',
    created_at TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP     NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
    last_login TIMESTAMP     NULL,
    is_active  BOOLEAN       DEFAULT TRUE,
    INDEX idx_username (username),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS queue_records (
    id           INT           AUTO_INCREMENT PRIMARY KEY,
    service_date DATE          NOT NULL,
    queue_number INT           NOT NULL,
    short_code   VARCHAR(10)   DEFAULT NULL,
    jwt_token    TEXT          DEFAULT NULL,
    pdf_path     VARCHAR(512)  DEFAULT NULL,
    status       ENUM('waiting','served','no_show','expired')
                               NOT NULL DEFAULT 'waiting',
    expires_at   DATETIME      DEFAULT NULL,
    created_at   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    served_at    DATETIME      DEFAULT NULL,
    INDEX idx_queue_number (queue_number),
    INDEX idx_service_date_queue (service_date, queue_number),
    INDEX idx_short_code (short_code),
    INDEX idx_status (status),
    INDEX idx_expires_at (expires_at),
    INDEX idx_queue_status_created (queue_number, status, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS crowd_snapshots (
    id               INT       AUTO_INCREMENT PRIMARY KEY,
    recorded_at      DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    person_count     INT       NOT NULL DEFAULT 0,
    avg_density      FLOAT     NOT NULL DEFAULT 0.0,
    max_density      FLOAT     NOT NULL DEFAULT 0.0,
    queue_length     INT       NOT NULL DEFAULT 0,
    active_counters  INT       NOT NULL DEFAULT 3,
    est_wait_minutes FLOAT     NOT NULL DEFAULT 0.0,
    INDEX idx_recorded_at (recorded_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
