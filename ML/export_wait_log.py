"""
Exports the system's own wait log as a CSV that ML/train_wait_model.py reads.

Each served ticket that has a logged estimate and a call time becomes one row:
how many people were ahead when the number was issued, how many counters were
open, what wait the system showed, and when the person was actually called.
This is the data the model needs and the October log lacked: a real head
count, and the system's estimate next to the wait that followed.

Read-only: it runs one SELECT against the database in .env.

Usage:
    .venv/Scripts/python.exe ML/export_wait_log.py
    .venv/Scripts/python.exe ML/train_wait_model.py --csv ML/data/wait_time/system_wait_log.csv --out ML/data/wait_time/candidate_model.json
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from database.database_handler import fetch_wait_observations  # noqa: E402

DEFAULT_OUT = ROOT / "ML" / "data" / "wait_time" / "system_wait_log.csv"
FIELDS = ["date", "ticket_number", "ticket_time", "called_time", "finished_time",
          "people_ahead", "windows_open", "predicted_wait_min", "wait_model_version"]


def to_row(record: dict) -> dict:
    created, called, served = record["created_at"], record["called_at"], record["served_at"]
    return {
        "date": created.strftime("%Y-%m-%d"),
        "ticket_number": record["queue_number"],
        "ticket_time": created.strftime("%H:%M:%S"),
        "called_time": called.strftime("%H:%M:%S"),
        "finished_time": served.strftime("%H:%M:%S") if served else "",
        "people_ahead": record["people_ahead_at_issue"],
        "windows_open": record["active_counters_at_issue"] or "",
        "predicted_wait_min": record["predicted_wait_min"] if record["predicted_wait_min"] is not None else "",
        "wait_model_version": record["wait_model_version"] or "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the system's wait log for training.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    records = fetch_wait_observations()
    if not records:
        raise SystemExit("No served tickets with a logged estimate and call time yet.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(to_row(r) for r in records)
    days = sorted({r["created_at"].date() for r in records})
    print(f"Wrote {len(records)} tickets over {len(days)} day(s) to {args.out}")


if __name__ == "__main__":
    main()
