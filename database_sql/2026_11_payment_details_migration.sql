-- What the student said they are paying for, and how much, from the app's
-- join form. Shown to the cashier. Additive: existing rows keep NULL. The
-- backend also adds these at startup (ensure_wait_log_columns); run this by
-- hand first to keep schema changes in your own hands.
ALTER TABLE `queue_records`
  ADD COLUMN `payment_type`   VARCHAR(60)   DEFAULT NULL,
  ADD COLUMN `payment_amount` DECIMAL(10,2) DEFAULT NULL,
  ADD COLUMN `payment_note`   VARCHAR(120)  DEFAULT NULL;
