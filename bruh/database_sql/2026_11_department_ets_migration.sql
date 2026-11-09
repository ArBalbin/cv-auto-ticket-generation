-- Additive Department and trained ETS deployment fields. Existing tickets and
-- student profiles remain intact; old accounts keep NULL until updated.
ALTER TABLE `student_profiles`
  ADD COLUMN `department` VARCHAR(20) DEFAULT NULL;

ALTER TABLE `queue_records`
  ADD COLUMN `department_at_issue` VARCHAR(20) DEFAULT NULL,
  ADD COLUMN `predicted_service_time_seconds` INT DEFAULT NULL,
  ADD COLUMN `prediction_at_issue` DATETIME DEFAULT NULL,
  ADD COLUMN `cashier_window` TINYINT DEFAULT NULL;
