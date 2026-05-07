# QueueFlow Cloud Cache Setup

Use Redis for temporary live data when deploying QueueFlow to cloud.
Permanent records still belong in MySQL.

## What Redis Stores

- `queueflow:snapshot:latest`: latest annotated JPEG snapshot, short TTL
- `queueflow:snapshot:seq`: snapshot sequence number, short TTL
- `queueflow:state:latest`: latest crowd and prediction state, short TTL
- `queueflow:history:recent`: recent crowd count history, longer TTL

The live camera snapshot is not stored in MySQL.

## Required Cloud Environment

```env
REDIS_URL=redis://default:password@your-redis-host:6379/0
CACHE_KEY_PREFIX=queueflow
CACHE_STATE_TTL_SECONDS=30
CACHE_SNAPSHOT_TTL_SECONDS=10
CACHE_HISTORY_TTL_SECONDS=3600
REDIS_SOCKET_TIMEOUT=0.25
REDIS_CONNECT_TIMEOUT=0.25

PORTAL_BASE_URL=https://your-domain.com
API_BASE_URL=https://your-domain.com
JWT_SECRET_KEY=replace-with-one-fixed-long-secret
```

## Deployment Notes

- Install dependencies with `pip install -r requirements.txt`; this now includes `redis`.
- Keep one backend instance for the thesis prototype unless queue state is moved fully to Redis or MySQL.
- If you run multiple backend instances, Redis lets `/api/snapshot`, `/api/crowd/video`, and latest crowd data read the same live data.
- Generate new tickets after setting `PORTAL_BASE_URL`, because old QR codes keep the URL they were created with.

## Health Check

Open:

```text
https://your-domain.com/health
```

Expected cache section when Redis is working:

```json
{
  "cache": {
    "configured": true,
    "available": true
  }
}
```
