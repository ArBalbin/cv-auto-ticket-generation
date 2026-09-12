-- Per-recognition instrumentation for the evaluation chapter (SOP #2:
-- "how accurately can a YOLO-based AI system identify students and assign
-- queue numbers without requiring manual input").
--
-- face_match_events already records the *decision* (score, margin, accepted)
-- for every match attempt. What it cannot answer is how LONG recognition
-- took, and how many camera frames were spent getting there — the two
-- questions a live deployment is actually judged on. One row is written
-- here per successful link.
--
-- Timing is split into two stages because they have different causes:
--   seconds_to_confirm : first sighting -> presence confirmed. Governed by
--                        QUEUE_MIN_CONFIRM_FRAMES and how steadily the
--                        tracker holds the same track_id.
--   seconds_to_link    : first sighting -> queue number issued. Adds the
--                        face-match step, which can only run on frames that
--                        actually yielded an embedding.
--
-- embed_attempts / embed_successes give the live face-detection rate. A miss
-- is not a wrong answer, it is a retry: the frame cost (~260ms) is spent and
-- the student keeps waiting, so this ratio drives seconds_to_link directly.
CREATE TABLE IF NOT EXISTS recognition_metrics (
    id                 INT       AUTO_INCREMENT PRIMARY KEY,
    track_id           INT       DEFAULT NULL,
    student_id         INT       DEFAULT NULL,
    queue_number       INT       DEFAULT NULL,
    linked_via         VARCHAR(20) DEFAULT NULL,
    first_seen_at      DATETIME  DEFAULT NULL,
    confirmed_at       DATETIME  DEFAULT NULL,
    linked_at          DATETIME  DEFAULT NULL,
    seconds_to_confirm FLOAT     DEFAULT NULL,
    seconds_to_link    FLOAT     DEFAULT NULL,
    embed_attempts     INT       NOT NULL DEFAULT 0,
    embed_successes    INT       NOT NULL DEFAULT 0,
    match_score        FLOAT     DEFAULT NULL,
    match_margin       FLOAT     DEFAULT NULL,
    created_at         DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_created_at (created_at),
    INDEX idx_student_id (student_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
