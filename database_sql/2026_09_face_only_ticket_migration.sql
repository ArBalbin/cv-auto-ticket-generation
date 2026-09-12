-- Restores a system-minted queue number for registered students, per SOP #2
-- as reconfirmed by the panel AFTER the pre-oral revision meeting: "How
-- accurately can a YOLO-based AI system identify students and assign queue
-- numbers without requiring manual input?" A registered student recognized
-- at the queue-zone camera now gets a queue number generated directly by
-- the system (mint_face_only_number() in queue_tracker.py) — no kiosk
-- ticket, no staff involvement. This supersedes the note in
-- 2026_09_student_self_registration_migration.sql, which reflected the
-- pre-oral-only reading before this SOP was reconfirmed.
--
-- This is purely additive for registered students; it does not touch the
-- kiosk's own walk-in flow (force_new_person()/manual linking), which is
-- unchanged, per "refrain from modifying the current queue flow."
ALTER TABLE queue_records
    MODIFY COLUMN linked_via ENUM('face_ocr', 'manual', 'face_only')
                              NOT NULL DEFAULT 'manual';
