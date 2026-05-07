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
    created_by_user_id INT     DEFAULT NULL,
    served_by_user_id  INT     DEFAULT NULL,
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

CREATE TABLE IF NOT EXISTS queue_events (
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

CREATE TABLE IF NOT EXISTS counter_config_history (
    id                 INT       AUTO_INCREMENT PRIMARY KEY,
    old_counters       INT       DEFAULT NULL,
    new_counters       INT       NOT NULL,
    avg_service_time   FLOAT     NOT NULL DEFAULT 3.0,
    changed_by_user_id INT       DEFAULT NULL,
    created_at         DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_created_at (created_at),
    INDEX idx_changed_by_user_id (changed_by_user_id),
    CONSTRAINT fk_counter_config_actor
        FOREIGN KEY (changed_by_user_id) REFERENCES users(id)
        ON DELETE SET NULL
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
