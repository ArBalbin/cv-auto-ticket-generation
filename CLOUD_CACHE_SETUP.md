# QueueFlow Cloud Cache Setup

Use Redis for temporary live data when deploying QueueFlow to cloud.
Permanent records still belong in MySQL.

## Current Progress Snapshot

Last updated: May 19, 2026

Redis cache support is implemented but optional. When `REDIS_URL` is set, live
queue state, recent history, and the latest annotated snapshot can be mirrored to
Redis so a cloud-hosted backend can serve dashboard data more reliably. When
Redis is not configured or unavailable, QueueFlow falls back to process memory.

## What Redis Stores

- `queueflow:snapshot:latest`: latest annotated JPEG snapshot, short TTL
- `queueflow:snapshot:seq`: snapshot sequence number, short TTL
- `queueflow:state:latest`: latest crowd and prediction state, short TTL
- `queueflow:history:recent`: recent crowd count history, longer TTL

The live camera snapshot is not stored in MySQL.

## Redis Persistence Requirement

QueueFlow treats Redis as a volatile cache, not as permanent storage. The
application writes live state and annotated camera snapshots using short TTL
values, but TTL alone does not prevent Redis from writing data to disk if Redis
RDB snapshots or AOF persistence are enabled on the Redis server.

For the privacy claim that live camera snapshots are temporary, Redis should run
in memory only:

```conf
save ""
appendonly no
```

This repository includes `redis.conf` with those settings for self-hosted Redis.
If you use a managed Redis provider, disable persistence/backups for the Redis
instance used by QueueFlow, or disclose provider-side persistence in the
manuscript and deployment notes.

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
