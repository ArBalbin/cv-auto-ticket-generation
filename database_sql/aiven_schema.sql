-- ============================================================================
-- QueuEx — database schema
-- ============================================================================
--
-- The complete structure the running system expects: eight tables, in three
-- groups.
--
--   Identity     users, student_profiles
--   Queue ops    queue_records, queue_events, counter_config_history
--   Measurement  face_match_events, recognition_metrics, crowd_snapshots
--
-- The measurement group is what makes the evaluation chapter possible: it
-- records every face-match attempt and every recognition, so accuracy can be
-- reported from real operation instead of estimated.
--
-- HOW TO RUN (Aiven, from the project root):
--
--   mysql --host=<service>.aivencloud.com --port=<port> \
--         --user=avnadmin --password \
--         --ssl-mode=REQUIRED --ssl-ca=ca.pem \
--         defaultdb -e "source database_sql/aiven_schema.sql"
--
-- Then load the rows:
--
--   ... defaultdb -e "source database_sql/aiven_data.sql"
--
-- THIS FILE DROPS THE EIGHT TABLES BELOW before recreating them. Tables it
-- does not name are left alone. Re-running it is safe and repeatable, but it
-- destroys whatever those eight tables held.
--
-- Tables are dropped children-first and created parents-first, so foreign
-- keys resolve without ever disabling FOREIGN_KEY_CHECKS.
-- ============================================================================


-- ----------------------------------------------------------------------------
-- Drop: children before the tables they reference.
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS `crowd_snapshots`;
DROP TABLE IF EXISTS `recognition_metrics`;
DROP TABLE IF EXISTS `face_match_events`;
DROP TABLE IF EXISTS `counter_config_history`;
DROP TABLE IF EXISTS `queue_events`;
DROP TABLE IF EXISTS `queue_records`;
DROP TABLE IF EXISTS `student_profiles`;
DROP TABLE IF EXISTS `users`;


-- ============================================================================
-- IDENTITY
-- ============================================================================

-- Staff accounts for the dashboard. Passwords are scrypt hashes, never
-- plaintext.
CREATE TABLE `users` (
  `id`         int          NOT NULL AUTO_INCREMENT,
  `username`   varchar(255) NOT NULL,
  `password`   varchar(255) NOT NULL,
  `email`      varchar(255) DEFAULT NULL,
  `full_name`  varchar(255) DEFAULT NULL,
  `role`       varchar(50)  DEFAULT 'staff',
  `created_at` timestamp    NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp    NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
  `last_login` timestamp    NULL DEFAULT NULL,
  `is_active`  tinyint(1)   DEFAULT '1',
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`),
  -- Redundant: the UNIQUE key above already indexes this column. Kept so this
  -- file matches the live database exactly; drop both together if ever tidied.
  KEY `idx_username` (`username`),
  KEY `idx_is_active` (`is_active`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- Enrolled students. `embedding` holds a 512-number ArcFace vector, NOT a
-- photograph: no student image is ever stored, and the original face cannot be
-- reconstructed from it. `google_sub` is Google's stable account id and is what
-- sign-in trusts — an email address can be reassigned, a sub cannot.
CREATE TABLE `student_profiles` (
  `id`                      int          NOT NULL AUTO_INCREMENT,
  `school_id`               varchar(50)  NOT NULL,
  `gbox_email`              varchar(255) DEFAULT NULL,
  `google_sub`              varchar(255) DEFAULT NULL,
  `full_name`               varchar(255) DEFAULT NULL,
  `embedding`               json         DEFAULT NULL,
  `embedding_model_version` varchar(50)  DEFAULT NULL,
  `num_enrollment_samples`  int          NOT NULL DEFAULT '0',
  `enrolled_by_user_id`     int          DEFAULT NULL,
  `is_active`               tinyint(1)   NOT NULL DEFAULT '1',
  `enrolled_at`             datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `last_login`              datetime     DEFAULT NULL,
  `face_enrolled_at`        datetime     DEFAULT NULL,
  `updated_at`              datetime     DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_school_id`  (`school_id`),
  UNIQUE KEY `uq_gbox_email` (`gbox_email`),
  UNIQUE KEY `uq_google_sub` (`google_sub`),
  KEY `idx_is_active` (`is_active`),
  KEY `fk_student_profiles_enrolled_by` (`enrolled_by_user_id`),
  CONSTRAINT `fk_student_profiles_enrolled_by`
    FOREIGN KEY (`enrolled_by_user_id`) REFERENCES `users` (`id`)
    ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================================
-- QUEUE OPERATIONS
-- ============================================================================

-- Every issued ticket.
--
-- `linked_via` records how the ticket reached a person:
--   face_only  face recognition minted the number      (the normal path)
--   manual     staff typed a walk-in's kiosk number
--   face_ocr   dead value — the retired design that read a printed kiosk
--              ticket with a digit CNN. Kept so historical rows stay
--              readable; nothing writes it any more.
--
-- Every student_id here is ON DELETE SET NULL, so erasing a student under the
-- Data Privacy Act keeps the queue history intact and only removes the link.
CREATE TABLE `queue_records` (
  `id`                 int          NOT NULL AUTO_INCREMENT,
  `service_date`       date         NOT NULL,
  `queue_number`       int          NOT NULL,
  `short_code`         varchar(10)  DEFAULT NULL,
  `jwt_token`          text,
  `pdf_path`           varchar(512) DEFAULT NULL,
  `created_by_user_id` int          DEFAULT NULL,
  `served_by_user_id`  int          DEFAULT NULL,
  `student_id`         int          DEFAULT NULL,
  `is_walkin`          tinyint(1)   NOT NULL DEFAULT '1',
  `linked_via`         enum('face_ocr','manual','face_only') NOT NULL DEFAULT 'manual',
  `linked_at`          datetime     DEFAULT NULL,
  `status`             enum('waiting','served','no_show','expired') NOT NULL DEFAULT 'waiting',
  `expires_at`         datetime     DEFAULT NULL,
  `created_at`         datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `served_at`          datetime     DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_queue_number`          (`queue_number`),
  KEY `idx_service_date_queue`    (`service_date`,`queue_number`),
  KEY `idx_short_code`            (`short_code`),
  KEY `idx_status`                (`status`),
  KEY `idx_expires_at`            (`expires_at`),
  KEY `idx_queue_status_created`  (`queue_number`,`status`,`created_at`),
  KEY `idx_created_by_user_id`    (`created_by_user_id`),
  KEY `idx_served_by_user_id`     (`served_by_user_id`),
  KEY `idx_student_id`            (`student_id`),
  CONSTRAINT `fk_queue_records_created_by`
    FOREIGN KEY (`created_by_user_id`) REFERENCES `users` (`id`) ON DELETE SET NULL,
  CONSTRAINT `fk_queue_records_served_by`
    FOREIGN KEY (`served_by_user_id`) REFERENCES `users` (`id`) ON DELETE SET NULL,
  CONSTRAINT `fk_queue_records_student`
    FOREIGN KEY (`student_id`) REFERENCES `student_profiles` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- Audit trail of queue actions. `service_date` and `queue_number` are
-- denormalised on purpose: the common reports read them without joining back
-- to queue_records.
CREATE TABLE `queue_events` (
  `id`              int  NOT NULL AUTO_INCREMENT,
  `queue_record_id` int  DEFAULT NULL,
  `service_date`    date DEFAULT NULL,
  `queue_number`    int  DEFAULT NULL,
  `event_type`      enum('created','served','no_show','expired','reset') NOT NULL,
  `actor_user_id`   int          DEFAULT NULL,
  `event_note`      varchar(255) DEFAULT NULL,
  `created_at`      datetime     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_queue_record_id`      (`queue_record_id`),
  KEY `idx_queue_number_created` (`queue_number`,`created_at`),
  KEY `idx_event_type_created`   (`event_type`,`created_at`),
  KEY `idx_actor_user_id`        (`actor_user_id`),
  CONSTRAINT `fk_queue_events_actor`
    FOREIGN KEY (`actor_user_id`) REFERENCES `users` (`id`) ON DELETE SET NULL,
  CONSTRAINT `fk_queue_events_record`
    FOREIGN KEY (`queue_record_id`) REFERENCES `queue_records` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- How many counters were open, and when it changed. The M/M/c wait-time model
-- needs to know how many servers were active during a period, and that changes
-- during a shift.
CREATE TABLE `counter_config_history` (
  `id`                 int      NOT NULL AUTO_INCREMENT,
  `old_counters`       int      DEFAULT NULL,
  `new_counters`       int      NOT NULL,
  `avg_service_time`   float    NOT NULL DEFAULT '3',
  `changed_by_user_id` int      DEFAULT NULL,
  `created_at`         datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_created_at`         (`created_at`),
  KEY `idx_changed_by_user_id` (`changed_by_user_id`),
  CONSTRAINT `fk_counter_config_actor`
    FOREIGN KEY (`changed_by_user_id`) REFERENCES `users` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================================
-- MEASUREMENT
-- ============================================================================

-- One row per face-match attempt, accepted or not.
--
-- Recording the REJECTED attempts is the point: a rejection means the system
-- saw a face and refused to guess, which is the designed safe outcome. Without
-- these rows there is no way to show how often that safeguard fires, or to
-- justify the score and margin thresholds from live data.
CREATE TABLE `face_match_events` (
  `id`            int        NOT NULL AUTO_INCREMENT,
  `student_id`    int        DEFAULT NULL,
  `track_id`      int        DEFAULT NULL,
  `matched_score` float      DEFAULT NULL,
  `margin`        float      DEFAULT NULL,
  `accepted`      tinyint(1) NOT NULL,
  `created_at`    datetime   NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_created_at` (`created_at`),
  KEY `idx_student_id` (`student_id`),
  CONSTRAINT `fk_face_match_student`
    FOREIGN KEY (`student_id`) REFERENCES `student_profiles` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- One row per successful recognition: the whole story of a single link.
--
-- `embed_attempts` and `embed_successes` together give the live face-detection
-- rate — a missed detection and a rejected match are different failures with
-- different fixes, and only this pair separates them.
--
-- Deliberately has NO foreign key on student_id: these rows are evaluation
-- history and must survive a student record being deleted.
CREATE TABLE `recognition_metrics` (
  `id`                 int         NOT NULL AUTO_INCREMENT,
  `track_id`           int         DEFAULT NULL,
  `student_id`         int         DEFAULT NULL,
  `queue_number`       int         DEFAULT NULL,
  `linked_via`         varchar(20) DEFAULT NULL,
  `first_seen_at`      datetime    DEFAULT NULL,
  `confirmed_at`       datetime    DEFAULT NULL,
  `linked_at`          datetime    DEFAULT NULL,
  `seconds_to_confirm` float       DEFAULT NULL,
  `seconds_to_link`    float       DEFAULT NULL,
  `embed_attempts`     int         NOT NULL DEFAULT '0',
  `embed_successes`    int         NOT NULL DEFAULT '0',
  `match_score`        float       DEFAULT NULL,
  `match_margin`       float       DEFAULT NULL,
  `created_at`         datetime    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_created_at` (`created_at`),
  KEY `idx_student_id` (`student_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- Time-series of the queue area as a whole. No foreign key to a ticket: one
-- snapshot describes the whole area at a moment, not one person.
CREATE TABLE `crowd_snapshots` (
  `id`               int      NOT NULL AUTO_INCREMENT,
  `recorded_at`      datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `person_count`     int      NOT NULL DEFAULT '0',
  `avg_density`      float    NOT NULL DEFAULT '0',
  `max_density`      float    NOT NULL DEFAULT '0',
  `queue_length`     int      NOT NULL DEFAULT '0',
  `active_counters`  int      NOT NULL DEFAULT '3',
  `est_wait_minutes` float    NOT NULL DEFAULT '0',
  PRIMARY KEY (`id`),
  KEY `idx_recorded_at` (`recorded_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
