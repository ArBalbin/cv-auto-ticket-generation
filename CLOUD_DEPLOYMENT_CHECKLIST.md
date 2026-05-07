# QueueFlow Cloud Deployment Checklist

This checklist covers the app/cloud setup only. Database schema, users, and records are intentionally left for your own database setup.

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
database/schema_cloud_ready.sql
```

For an existing database created from the older local script, run this migration as a MySQL admin/root user:

```text
database/migrations/001_queue_records_cloud_ready.sql
```

The migration adds `service_date`, removes the risky global `UNIQUE(queue_number)`, and keeps existing rows.

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

## 4. Ticket QR Codes

Set:

```env
PORTAL_BASE_URL=https://your-cloud-domain.com
```

After changing this value, generate new tickets. Old tickets may still point to the old local IP.

## 5. Detector / Camera Deployment

The cloud server cannot directly access your local webcam. Run `app/detector.py` on the computer connected to the camera and set this on that computer:

```env
API_BASE_URL=https://your-cloud-domain.com
CAM_TOKEN=the-same-token-used-by-the-cloud-api
```

The detector will push `/yolo/push-frame` and `/yolo/update` to the cloud backend.

## 6. Ticket Output

In the actual queue area, this system is intended to print tickets through a thermal printer. For the current prototype, tickets are generated as PDF files instead because no thermal printer is available.

By default, prototype PDF tickets are written to:

```env
TICKETS_OUTPUT_DIR=app/tickets
```

Many cloud platforms erase local files on restart/redeploy. Use persistent disk storage if you need to keep generated PDFs, or keep PDF generation on the local queue-area computer that runs the detector.

## 7. Start Command

Generic start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT --proxy-headers
```

For providers that support a `Procfile`, this repository includes:

```Procfile
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT --proxy-headers
```

## 8. Smoke Test

After deployment:

1. Open `https://your-cloud-domain.com/health`.
2. Confirm `"status": "ok"`.
3. Confirm `"db": true` after your database is configured.
4. Login at `https://your-cloud-domain.com/login`.
5. Start the local detector and confirm `"snapshot": true` in `/health`.
6. Generate a new ticket and scan the QR with the mobile app.

## 9. GitHub Safety

Do not commit `.env`, generated ticket PDFs, logs, local virtual environments, or model weights. The `.gitignore` now excludes those runtime files.
