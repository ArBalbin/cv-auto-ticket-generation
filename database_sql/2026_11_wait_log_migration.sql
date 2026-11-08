-- ============================================================================
-- 2026-11: log the wait-time estimate and the real wait for each ticket.
--
-- Records how many people were ahead and what wait the system estimated when
-- the number was issued, and when that person was called to a counter. This is
-- what checks the wait-time model against real waits and re-fits it on exact
-- queue counts (WAIT_TIME_MODEL.md).
--
-- The backend applies these on startup (ensure_wait_log_columns in
-- app/database/database_handler.py), so this file is for databases migrated
-- by hand and for the record. Run once; MySQL has no ADD COLUMN IF NOT EXISTS.
-- ============================================================================

ALTER TABLE `queue_records`
  -- At issue: the queue the person joined and what the system told them.
  ADD COLUMN `people_ahead_at_issue`    INT          DEFAULT NULL,
  ADD COLUMN `active_counters_at_issue` TINYINT      DEFAULT NULL,
  ADD COLUMN `predicted_wait_min`       DECIMAL(6,1) DEFAULT NULL,
  ADD COLUMN `wait_model_version`       VARCHAR(32)  DEFAULT NULL,
  -- When a counter was assigned to them; the wait is called_at - created_at.
  ADD COLUMN `called_at`                DATETIME     DEFAULT NULL;
