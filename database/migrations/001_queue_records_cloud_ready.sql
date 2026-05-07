-- QueueFlow queue_records cloud-ready migration.
--
-- Purpose:
-- - Keep old ticket rows instead of overwriting Q001/Q002/etc.
-- - Add service_date for reporting/filtering.
-- - Remove the global UNIQUE(queue_number) rule.
--
-- Run as a MySQL admin/root user, not as the app user.
-- This does not drop tables or delete rows.

USE Crowd_Detection;

DELIMITER $$

DROP PROCEDURE IF EXISTS migrate_queue_records_cloud_ready $$
CREATE PROCEDURE migrate_queue_records_cloud_ready()
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'queue_records'
          AND COLUMN_NAME = 'service_date'
    ) THEN
        ALTER TABLE queue_records
            ADD COLUMN service_date DATE NULL AFTER id;
    END IF;

    UPDATE queue_records
    SET service_date = DATE(created_at)
    WHERE service_date IS NULL;

    ALTER TABLE queue_records
        MODIFY service_date DATE NOT NULL;

    SET @unique_queue_index := (
        SELECT INDEX_NAME
        FROM INFORMATION_SCHEMA.STATISTICS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'queue_records'
          AND COLUMN_NAME = 'queue_number'
          AND NON_UNIQUE = 0
          AND INDEX_NAME <> 'PRIMARY'
        LIMIT 1
    );

    IF @unique_queue_index IS NOT NULL THEN
        SET @sql := CONCAT('ALTER TABLE queue_records DROP INDEX `', @unique_queue_index, '`');
        PREPARE stmt FROM @sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM INFORMATION_SCHEMA.STATISTICS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'queue_records'
          AND INDEX_NAME = 'idx_service_date_queue'
    ) THEN
        CREATE INDEX idx_service_date_queue
            ON queue_records (service_date, queue_number);
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM INFORMATION_SCHEMA.STATISTICS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'queue_records'
          AND INDEX_NAME = 'idx_queue_status_created'
    ) THEN
        CREATE INDEX idx_queue_status_created
            ON queue_records (queue_number, status, created_at);
    END IF;
END $$

CALL migrate_queue_records_cloud_ready() $$
DROP PROCEDURE migrate_queue_records_cloud_ready $$

DELIMITER ;

SHOW COLUMNS FROM queue_records;
SHOW INDEX FROM queue_records;
