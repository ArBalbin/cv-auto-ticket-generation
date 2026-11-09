-- Which cashier window served each ticket, written when staff mark it done or
-- no-show. Additive: existing rows keep NULL. The backend also adds this column
-- at startup (ensure_wait_log_columns), so running this by hand is optional.
ALTER TABLE `queue_records`
  ADD COLUMN `cashier_window` TINYINT DEFAULT NULL;
