# QueueFlow Database Entities and Relationships

This document explains the cloud-ready database structure used by QueueFlow.

## Main Entities

### users

Stores staff accounts that can log in to the QueueFlow dashboard.

Important columns:

- `id` - primary key
- `username` - unique staff login name
- `password` - hashed password
- `role` - staff/admin role label
- `is_active` - account status

### queue_records

Stores every generated queue ticket.

Important columns:

- `id` - primary key
- `service_date` - date the queue number belongs to
- `queue_number` - visible queue number shown to the customer
- `short_code` - customer access token shown on the ticket
- `jwt_token` - signed ticket token used for QR/status validation
- `pdf_path` - local PDF path or object storage URL/path
- `status` - `waiting`, `served`, `no_show`, or `expired`
- `created_by_user_id` - optional staff user that created the ticket
- `served_by_user_id` - staff user that marked the ticket as served

Relationships:

- `queue_records.created_by_user_id` references `users.id`
- `queue_records.served_by_user_id` references `users.id`

### queue_events

Stores an audit trail of queue actions.

Important columns:

- `id` - primary key
- `queue_record_id` - related ticket record
- `queue_number` - queue number at the time of the event
- `event_type` - `created`, `served`, `no_show`, `expired`, or `reset`
- `actor_user_id` - staff user that performed the action, if any
- `event_note` - short action description
- `created_at` - event time

Relationships:

- `queue_events.queue_record_id` references `queue_records.id`
- `queue_events.actor_user_id` references `users.id`

### counter_config_history

Stores a history of active counter changes from the dashboard.

Important columns:

- `id` - primary key
- `old_counters` - previous active counter count
- `new_counters` - new active counter count
- `avg_service_time` - service time value used with the counter change
- `changed_by_user_id` - staff user that changed the counter count
- `created_at` - change time

Relationship:

- `counter_config_history.changed_by_user_id` references `users.id`

### crowd_snapshots

Stores computer vision analytics snapshots.

Important columns:

- `id` - primary key
- `recorded_at` - snapshot timestamp
- `person_count` - detected people count
- `queue_length` - queue length during the snapshot
- `active_counters` - active counters during the snapshot
- `est_wait_minutes` - estimated wait time

This table is time-series analytics data. It does not need a direct foreign key to a queue ticket because one snapshot can describe the whole area, not one customer.

## ERD Summary

```text
users
  1 ---- many queue_records.created_by_user_id
  1 ---- many queue_records.served_by_user_id
  1 ---- many queue_events.actor_user_id
  1 ---- many counter_config_history.changed_by_user_id

queue_records
  1 ---- many queue_events.queue_record_id

crowd_snapshots
  standalone analytics/time-series table
```

## System Data Flow With Relationships

1. The detector sees a new person in the queue.
2. The backend generates a ticket PDF and creates a row in `queue_records`.
3. The backend records a `created` event in `queue_events`.
4. Staff logs in through the dashboard using a `users` account.
5. When staff marks a queue number as done, `queue_records.status` becomes `served`.
6. The same action stores `served_by_user_id` and creates a `served` event in `queue_events`.
7. If staff changes active counters, the change is stored in `counter_config_history`.
8. CV analytics are stored separately in `crowd_snapshots`.

## Why This Is Better

The original database already had entities, but most relationships were only logical. The updated version adds real foreign keys and audit tables, so the system can answer questions like:

- Which staff member served a ticket?
- When was a ticket created, served, or marked no-show?
- Who changed the number of active counters?
- How did counter settings change over time?
- What crowd conditions were recorded during operation?
