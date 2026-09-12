"""
Reproducible load/pressure test runner (panel-required).

Does the setup that makes the numbers meaningful, then drives Locust:

  1. Logs in as staff and seeds a real walk-in ticket, so the status-poll
     scenario measures the genuine "active entry found and serialised" path
     rather than a 404. Those are different code paths with different costs,
     and quoting the 404 latency as if it were a real ticket-holder's would
     understate the result.
  2. Runs the read-only scenarios (safe against any environment).
  3. Optionally runs the camera-ingest scenario, which mutates server state.
  4. Writes CSVs for the evaluation chapter.

Usage:
    python load_testing/run_load_test.py                 # read-only
    python load_testing/run_load_test.py --with-camera   # + detector ingest
    python load_testing/run_load_test.py -u 100 -t 3m

The camera scenario accumulates synthetic pending entries in the tracker.
Reset afterwards from the staff dashboard, or restart the backend, before
collecting real beta data.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
LOCUSTFILE = Path(__file__).parent / "locustfile.py"
RESULTS_DIR = Path(__file__).parent / "results"

READONLY_USERS = ["StatusPollUser", "DisplayBoardUser", "StaffAnalyticsUser"]
ALL_USERS = READONLY_USERS + ["CameraPushUser"]


def seed_ticket(host: str, username: str, password: str) -> tuple[str, str] | None:
    """Log in as staff, create a walk-in entry, return (queue_number, token)."""
    session = requests.Session()
    try:
        login = session.post(
            f"{host}/api/auth/login",
            json={"username": username, "password": password},
            timeout=10,
        )
        if login.status_code != 200:
            print(f"  staff login failed ({login.status_code}) — "
                  f"status poll will measure the not-found path instead")
            return None

        token = login.json().get("access_token")
        headers = {"Authorization": f"Bearer {token}"}

        # A high number, clear of both kiosk tickets and the face-only range,
        # so the seeded entry is obviously synthetic in any later report.
        queue_number = 9001
        created = session.post(
            f"{host}/api/queue/force-new",
            json={"queue_number": queue_number, "is_walkin": True},
            headers=headers,
            timeout=10,
        )
        if created.status_code not in (200, 201):
            print(f"  could not seed ticket ({created.status_code}: "
                  f"{created.text[:120]})")
            return None

        # /api/queue/status validates against the ticket's SHORT CODE, which
        # the background ticket worker mints a moment after the entry is
        # created. Reading immediately returns the raw access_token instead,
        # which that endpoint rejects with 401 — so poll until the real short
        # code appears rather than grabbing whatever is there first.
        import time

        for _ in range(20):
            data = session.get(
                f"{host}/api/queue/data", headers=headers, timeout=10
            ).json()
            for person in data.get("active_queue", []):
                if person.get("queue_number") != queue_number:
                    continue
                access = person.get("access_token")
                # The short code is formatted XXXX-XXXX; the fallback raw
                # token has no dash. Only the former will authenticate.
                if access and "-" in str(access):
                    return str(queue_number), str(access)
            time.sleep(0.5)

        print("  short code never appeared — status poll will measure the "
              "401 path, not a real ticket")
        return None
    except Exception as exc:
        print(f"  seeding failed: {exc}")
        return None


def cleanup_ticket(host: str, username: str, password: str,
                   queue_number: str, short_code: str) -> None:
    """
    Remove the synthetic ticket this run created.

    Without this, every load test leaves a Q9001 row sitting in queue_records
    with status 'waiting' — it shows up on the staff dashboard and the public
    display board as if a real person were queued, and the PDF worker writes a
    ticket file for it. Two runs on 2026-09-08 left exactly that behind and it
    had to be cleaned out by hand.

    Deletion is keyed on the SHORT CODE this run generated, not on the queue
    number, so it can only ever remove the row this script created. A run
    against a remote host will not have DB access; that is reported, not
    treated as a failure, because the load-test results are still valid.
    """
    # Take it out of the live in-memory tracker first, so the dashboard stops
    # showing it even if the DB step below cannot run.
    try:
        session = requests.Session()
        login = session.post(f"{host}/api/auth/login",
                             json={"username": username, "password": password},
                             timeout=10)
        if login.status_code == 200:
            token = login.json().get("access_token")
            session.post(f"{host}/api/queue/done",
                         json={"queue_number": int(queue_number)},
                         headers={"Authorization": f"Bearer {token}"},
                         timeout=10)
    except Exception as exc:
        print(f"  could not mark Q{queue_number} done: {exc}")

    try:
        sys.path.insert(0, str(ROOT / "app"))
        from database.database_handler import _ensure_db_pool

        pool = _ensure_db_pool()
        if pool is None:
            print("  no DB connection — remove the seeded row manually")
            return

        conn = pool.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM queue_records WHERE short_code = %s", (short_code,))
        ids = [row[0] for row in cursor.fetchall()]
        if ids:
            placeholders = ",".join(["%s"] * len(ids))
            cursor.execute(
                f"DELETE FROM queue_events WHERE queue_record_id IN ({placeholders})", ids)
            events = cursor.rowcount
            cursor.execute(
                f"DELETE FROM queue_records WHERE id IN ({placeholders})", ids)
            conn.commit()
            print(f"  removed seeded ticket Q{queue_number} "
                  f"({cursor.rowcount} record, {events} event rows)")
        else:
            print(f"  no row found for short code {short_code} — nothing to remove")
        cursor.close()
        conn.close()
    except Exception as exc:
        print(f"  DB cleanup skipped ({exc}) — remove Q{queue_number} manually")

    for pdf in (ROOT / "app" / "tickets").glob(f"ticket_Q{queue_number}_*.pdf"):
        try:
            pdf.unlink()
            print(f"  removed {pdf.name}")
        except Exception as exc:
            print(f"  could not remove {pdf.name}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # 127.0.0.1, NOT localhost. On Windows "localhost" resolves to ::1 (IPv6)
    # first, but uvicorn binds 0.0.0.0 (IPv4 only), so every new connection
    # waits ~2s for the IPv6 attempt to time out before falling back. Measured
    # first-request latency: localhost 2036-2072ms vs 127.0.0.1 7-9ms. Each
    # Locust user pays it once, which showed up as a fake ~2100ms p98/p99 tail
    # in earlier runs. It is a client-side DNS artifact, not server latency.
    parser.add_argument("--host", default="http://127.0.0.1:5000")
    parser.add_argument("-u", "--users", default="50")
    parser.add_argument("-r", "--spawn-rate", default="5")
    parser.add_argument("-t", "--run-time", default="2m")
    parser.add_argument("--with-camera", action="store_true",
                        help="Include CameraPushUser (mutates server state).")
    parser.add_argument("--keep-ticket", action="store_true",
                        help="Leave the seeded ticket in place after the run "
                             "(default is to delete it, so the queue and the "
                             "display board are not left holding a fake entry).")
    parser.add_argument("--staff-user", default=os.getenv("LOAD_TEST_STAFF_USERNAME", "testadmin"))
    parser.add_argument("--staff-pass", default=os.getenv("LOAD_TEST_STAFF_PASSWORD", ""))
    args = parser.parse_args()

    try:
        health = requests.get(f"{args.host}/health", timeout=5)
        health.raise_for_status()
    except Exception:
        sys.exit(f"Backend not reachable at {args.host} — start app/main.py first.")

    print(f"Target: {args.host}")
    print(f"Load  : {args.users} users, spawn {args.spawn_rate}/s, for {args.run_time}")

    env = os.environ.copy()
    env["LOAD_TEST_STAFF_USERNAME"] = args.staff_user
    if args.staff_pass:
        env["LOAD_TEST_STAFF_PASSWORD"] = args.staff_pass

    print("\nSeeding a real ticket for the status-poll scenario...")
    seeded = None
    if args.staff_pass:
        seeded = seed_ticket(args.host, args.staff_user, args.staff_pass)
        if seeded:
            env["LOAD_TEST_QUEUE_NUMBER"], env["LOAD_TEST_ACCESS_TOKEN"] = seeded
            print(f"  seeded Q{seeded[0]} token={seeded[1]}")
    else:
        print("  no --staff-pass given; status poll will measure the "
              "not-found path and staff analytics will return 401")

    users = ALL_USERS if args.with_camera else READONLY_USERS
    if args.with_camera:
        print("\n  WARNING: CameraPushUser will create synthetic pending entries.")
        print("  Reset the queue before collecting real data.")

    RESULTS_DIR.mkdir(exist_ok=True)
    prefix = RESULTS_DIR / ("full" if args.with_camera else "readonly")

    cmd = [
        sys.executable, "-m", "locust",
        "-f", str(LOCUSTFILE),
        "--host", args.host,
        "--headless",
        "-u", args.users,
        "-r", args.spawn_rate,
        "-t", args.run_time,
        "--csv", str(prefix),
        *users,
    ]

    print(f"\nRunning: {' '.join(users)}\n")
    subprocess.run(cmd, env=env, cwd=str(ROOT))
    print(f"\nCSVs written to {prefix}_stats.csv (and _failures, _stats_history)")

    if seeded and not args.keep_ticket:
        print("\nCleaning up the seeded ticket...")
        cleanup_ticket(args.host, args.staff_user, args.staff_pass, *seeded)

    if args.with_camera:
        print("\n  REMINDER: CameraPushUser created synthetic pending entries in")
        print("  server memory. Reset the queue from the staff dashboard, or")
        print("  restart the backend, before collecting real beta data.")


if __name__ == "__main__":
    main()
