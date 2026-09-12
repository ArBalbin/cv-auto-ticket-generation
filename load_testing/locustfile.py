"""
Locust load/pressure test for QueueFlow's API endpoints (Phase 8,
panel-required "conduct load testing/pressure testing").

IMPORTANT — read before running:
  StatusPollUser and DisplayBoardUser are read-only and safe against any
  environment, including a live/shared database.

  CameraPushUser and StaffAnalyticsUser exercise real backend state: a
  confirmed presence can create in-memory pending queue entries, and
  StaffAnalyticsUser needs a real staff login. Neither issues raw SQL writes
  on their own, but both add real load to whatever backend/DB the target
  server is using. Point --host at a disposable/test deployment, not a
  live one, unless you intend that.

Usage (read-only scenarios, safe anywhere):
    locust -f load_testing/locustfile.py --host=http://localhost:5000 \\
        StatusPollUser DisplayBoardUser

Usage (full suite, test environment only):
    locust -f load_testing/locustfile.py --host=http://localhost:5000

Headless run with CSV export (for the thesis evaluation chapter):
    locust -f load_testing/locustfile.py --host=http://localhost:5000 \\
        --headless -u 50 -r 5 -t 2m --csv=results
"""

import os
import random

from locust import HttpUser, task, between


CAM_TOKEN = os.getenv("CAM_TOKEN", "detector-secret-token")
STAFF_USERNAME = os.getenv("LOAD_TEST_STAFF_USERNAME", "testadmin")
STAFF_PASSWORD = os.getenv("LOAD_TEST_STAFF_PASSWORD", "")

# A real queue number + access token, so the status poll exercises the
# genuine "found and serialised an active entry" path. Without these the
# endpoint short-circuits to 404/401, which is a cheaper code path and
# would understate the latency a real ticket-holder experiences.
REAL_QUEUE_NUMBER = os.getenv("LOAD_TEST_QUEUE_NUMBER", "")
REAL_ACCESS_TOKEN = os.getenv("LOAD_TEST_ACCESS_TOKEN", "")


class StatusPollUser(HttpUser):
    """Simulates many ticket-holders refreshing their status page. Read-only."""

    wait_time = between(2, 5)

    @task
    def poll_status(self):
        if REAL_QUEUE_NUMBER and REAL_ACCESS_TOKEN:
            params = {"q": REAL_QUEUE_NUMBER, "token": REAL_ACCESS_TOKEN}
            name = "/api/queue/status (real ticket)"
        else:
            # Fallback: still exercises routing and the lookup, but returns
            # not-found. Report it as a separate name so a run without seeded
            # credentials is never mistaken for the real-ticket measurement.
            params = {"q": random.randint(1, 999), "token": "TEST-TOKEN"}
            name = "/api/queue/status (not found)"

        self.client.get("/api/queue/status", params=params, name=name)


class DisplayBoardUser(HttpUser):
    """Simulates the public display board and prediction pollers. Read-only."""

    wait_time = between(1, 2)

    @task(2)
    def display(self):
        self.client.get("/api/queue/display", name="/api/queue/display")

    @task(1)
    def prediction(self):
        self.client.get("/api/queue/prediction", name="/api/queue/prediction")


class CameraPushUser(HttpUser):
    """Simulates the detector process pushing frames. Exercises the full
    queue_tracker per-frame path (candidate accumulation, dedup, remap);
    a sustained run can accumulate real pending entries in server memory.
    """

    wait_time = between(0.1, 0.3)

    def _synthetic_payload(self):
        return {
            "count": 1,
            "avg_density": 0.1,
            "max_density": 0.2,
            "active_counters": 3,
            "estimated_wait_time": 2.0,
            "arrival_rate": 1.0,
            "system_utilization": 0.3,
            "predicted_wait_5min": 2.0,
            "predicted_wait_15min": 3.0,
            "predicted_wait_30min": 4.0,
            "queue_length": 1,
            "timestamp": 0,
            "yolo_frame_idx": random.randint(1, 10_000),
            "tracked_persons": [
                {
                    "track_id": random.randint(100000, 999999),
                    "bbox": [100, 100, 200, 300],
                    "conf": 0.9,
                    "face_embedding": None,
                }
            ],
        }

    @task
    def push_frame(self):
        self.client.post(
            "/yolo/push-frame",
            json=self._synthetic_payload(),
            headers={"X-CAM-TOKEN": CAM_TOKEN},
            name="/yolo/push-frame",
        )


class StaffAnalyticsUser(HttpUser):
    """Simulates staff dashboards polling analytics under concurrent load.
    Requires LOAD_TEST_STAFF_PASSWORD to be set; skips login (and so gets
    401s, still a valid error-rate data point) if it isn't.
    """

    wait_time = between(2, 4)

    def on_start(self):
        if not STAFF_PASSWORD:
            return
        self.client.post(
            "/api/auth/login",
            json={"username": STAFF_USERNAME, "password": STAFF_PASSWORD},
            name="/api/auth/login",
        )

    @task
    def analytics(self):
        self.client.get("/api/queue/analytics", name="/api/queue/analytics")
