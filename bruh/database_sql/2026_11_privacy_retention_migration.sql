-- ============================================================================
-- 2026-11: consent before registration and the data-retention policy
-- (pre-final defense panel requirements).
--
-- The backend applies these on startup (ensure_privacy_columns in
-- app/database/database_handler.py), so this file is for databases migrated
-- by hand and for the record. Run once; MySQL has no ADD COLUMN IF NOT EXISTS.
-- ============================================================================

ALTER TABLE `student_profiles`
  -- Version of the data-privacy notice the student agreed to, and when.
  ADD COLUMN `consent_version` VARCHAR(32) DEFAULT NULL,
  ADD COLUMN `consented_at`    DATETIME    DEFAULT NULL,
  -- Latest sign-in or request to be queued; drives the retention clock.
  ADD COLUMN `last_active_at`  DATETIME    DEFAULT NULL,
  -- Set when the account is archived (deactivated, face data erased).
  ADD COLUMN `archived_at`     DATETIME    DEFAULT NULL;

CREATE INDEX `idx_last_active_at` ON `student_profiles` (`last_active_at`);
CREATE INDEX `idx_archived_at`    ON `student_profiles` (`archived_at`);
