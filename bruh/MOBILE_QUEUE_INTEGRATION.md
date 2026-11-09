# Mobile QueueFlow integration

There is no mobile application source tree in this backend workspace. The API
contracts below support adding department setup and queue estimates to the
existing client without replacing its sign-in or ticket flow.

## Department profile

- During first Google sign-in, send `department` with the existing `school_id`
  and consent fields. Allowed values are `CCS`, `CBM`, `CHS`, `CCJE`, `COE`,
  and `CTED`.
- Existing accounts can set or change the value with authenticated
  `PATCH /api/students/me/department`, JSON `{"department":"CCS"}`.
- Google sign-in responses include `department` (nullable for legacy accounts).
- Department changes affect future ticket snapshots only; they do not change
  an issued ticket number or its queue position.

## Ticket ETS and EWT

Continue polling the existing public endpoint
`GET /api/queue/status?q={queue_number}&token={ticket_token}`. Its existing
fields remain. The `prediction` object now also includes:

- `estimated_service_time_seconds`: this student's ETS.
- `estimated_wait_time_seconds`: EWT until a cashier starts service.
- `estimated_completion_duration_seconds`: EWT + ETS.
- `active_cashier_windows`, `people_ahead`, `department`,
  `prediction_method`, `model_version`, and `prediction_state`.

Show ETS and EWT as separate values. When `prediction_state` is `paused`, all
cashier windows are closed; present the queue as paused rather than suggesting
that service is about to start. `prediction_method` identifies the deployed
model path or average fallback. Model inference runs from the backend, so the
mobile client does not load model artifacts.

The existing endpoint is polled by the client, so every poll reflects the
current queue and cashier configuration. There is no mobile app or push
notification transport in this repository.

## Admin data export

An existing staff session with database role `admin` or `administrator` can
download `GET /api/queue/training-export`. The CSV includes only completed
records and omits student identifiers. It can be used for a separately
reviewed offline retraining pipeline; the API never retrains models.

## Deployment

The deployed artifact is `models/ets_model.joblib`, with
`models/model_metadata.json`. Optionally set `QUEUEFLOW_ETS_MODEL_DIR` to a
directory containing those files. Install the pinned scikit-learn 1.8.0 plus
joblib and pandas dependencies from `requirements.txt`. Deploy a replacement
artifact as a reviewed release and retain the previous model directory for
rollback. The bundled model reports weak generalization on its supplied test
split, so its metadata version is returned for observability, not as an
accuracy guarantee.
