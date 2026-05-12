# QueueFlow Cloud Deployment Checklist

This checklist covers the app/cloud setup only. Database schema, users, and records are intentionally left for your own database setup.

## Current Progress Snapshot

Last updated: May 12, 2026

The backend is cloud-ready through `Dockerfile`, `Procfile`, `render.yaml`, and
`.env.production.example`. The cloud deployment hosts the FastAPI backend,
dashboard routes, queue APIs, ticket/status APIs, database access, optional Redis
cache, and optional object storage. The YOLO detector still runs on the local
camera-connected machine and pushes to the cloud backend through `API_BASE_URL`.

PDF ticket generation is currently the active demo output. Thermal printing is
the target production output and still needs hardware integration/testing.

## 1. Required Cloud Environment Variables

Copy `.env.example` into your cloud provider's environment variable settings and replace the placeholders.

Minimum required values:

```env
APP_ENV=production
PORTAL_BASE_URL=https://your-cloud-domain.com
CORS_ORIGINS=https://your-cloud-domain.com
TRUSTED_HOSTS=your-cloud-domain.com
JWT_SECRET_KEY=use-a-long-random-secret
CAM_TOKEN=use-a-long-random-camera-token
SESSION_COOKIE_SECURE=1
SESSION_COOKIE_SAMESITE=lax
STAFF_REGISTRATION_ENABLED=1
STAFF_REGISTRATION_CODE=use-a-private-code-for-staff-signup
```

Most cloud hosts provide `PORT` automatically. The app now reads `PORT` first, then falls back to `API_PORT`.

## 2. Database

Configure these with your own cloud MySQL values:

```env
DB_HOST=
DB_PORT=3306
DB_NAME=
DB_USERNAME=
DB_PASSWORD=
DB_SSL_MODE=REQUIRED
```

For Aiven MySQL, use the service values shown in the Aiven overview. Aiven usually requires TLS, so keep `DB_SSL_MODE=REQUIRED`. If you download the Aiven CA certificate and your host supports file secrets, set `DB_SSL_CA` to that certificate file path and use `DB_SSL_MODE=VERIFY_CA`.

The app expects your database to contain the required tables, including staff users and queue ticket records.

For a fresh cloud database, import:

```text
database_sql/schema_cloud_ready.sql
```

For a clean Aiven reset that drops and recreates all QueueFlow tables, use:

```text
database_sql/aiven_clean_full_schema.sql
```

Only use the clean reset file if you are okay deleting the current cloud database rows.

For staff registration/login, the `users` table must have at least:

```sql
username
password
```

Recommended extra columns:

```sql
is_active
full_name
role
created_at
updated_at
last_login
```

The registration code checks your table columns and only inserts optional fields that exist.

## 3. Optional Redis Cache

Redis is optional, but recommended in cloud:

```env
REDIS_URL=redis://...
```

When enabled, Redis stores recent detector state, snapshots, history, and staff sessions. If Redis is not configured, the app falls back to process memory.

## 4. Optional Ticket Object Storage

The prototype still generates a local PDF ticket first. In cloud, local files may disappear after restart/redeploy, so object storage is recommended if you need to keep ticket PDFs.

QueueFlow supports S3-compatible storage:

```env
OBJECT_STORAGE_ENABLED=1
OBJECT_STORAGE_ENDPOINT_URL=https://your-s3-compatible-endpoint
OBJECT_STORAGE_BUCKET=your-bucket-name
OBJECT_STORAGE_REGION=your-region
OBJECT_STORAGE_ACCESS_KEY_ID=your-access-key
OBJECT_STORAGE_SECRET_ACCESS_KEY=your-secret-key
OBJECT_STORAGE_PREFIX=tickets
OBJECT_STORAGE_PUBLIC_BASE_URL=
OBJECT_STORAGE_ADDRESSING_STYLE=auto
```

If `OBJECT_STORAGE_PUBLIC_BASE_URL` is set, the database stores URLs using that base URL. Otherwise it stores an endpoint-based URL, or an `s3://bucket/key` URL for AWS-style storage.

## 5. Ticket QR Codes

Set:

```env
PORTAL_BASE_URL=https://your-cloud-domain.com
```

After changing this value, generate new tickets. Old tickets may still point to the old local IP.

## 6. Detector / Camera Deployment

The cloud server cannot directly access your local webcam. Run `app/detector.py` on the computer connected to the camera and set this on that computer:

```env
API_BASE_URL=https://your-cloud-domain.com
CAM_TOKEN=the-same-token-used-by-the-cloud-api
```

The detector will push `/yolo/push-frame` and `/yolo/update` to the cloud backend.

## 7. Ticket Output

In the actual queue area, this system is intended to print tickets through a thermal printer. For the current prototype, tickets are generated as PDF files instead because no thermal printer is available.

By default, prototype PDF tickets are written to:

```env
TICKETS_OUTPUT_DIR=app/tickets
```

Many cloud platforms erase local files on restart/redeploy. Use persistent disk storage or the object storage settings above if you need to keep generated PDFs.

## 8. Start Command

Generic start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT --proxy-headers
```

For providers that support a `Procfile`, this repository includes:

```Procfile
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT --proxy-headers
```

## 9. Smoke Test

After deployment:

1. Open `https://your-cloud-domain.com/health`.
2. Confirm `"status": "ok"`.
3. Confirm `"db": true` after your database is configured.
4. If Redis is configured, confirm `"cache": {"configured": true, "available": true}`.
5. If ticket storage is configured, confirm `"object_storage": {"configured": true}`.
6. Login at `https://your-cloud-domain.com/login`.
7. Start the local detector and confirm `"snapshot": true` in `/health`.
8. Generate a new ticket and scan the QR with the mobile app.

## 10. GitHub Safety

Do not commit `.env`, generated ticket PDFs, logs, local virtual environments, or model weights. The `.gitignore` now excludes those runtime files.
