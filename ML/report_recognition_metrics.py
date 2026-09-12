"""
Live face-recognition evaluation report (SOP #2).

Turns the rows captured during real use into the numbers the evaluation
chapter needs. Two tables feed this:

  recognition_metrics — one row per successful link: how long recognition
                        took end to end, and how many camera frames it cost.
  face_match_events   — one row per match attempt: score, margin, accepted.

The distinction that matters when reading the output: a *rejected* match and
a *missed detection* are different failures. A rejection means the decision
rule saw a face and refused to guess (the safe outcome, by design). A missed
detection means no face was found in that frame at all, so there was nothing
to decide on — the student simply waits for the next frame. Only the second
one is a speed problem.

Run: python ML/report_recognition_metrics.py [--since YYYY-MM-DD]
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from database.database_handler import _ensure_db_pool  # noqa: E402


def percentiles(values: list[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    if not ordered:
        return 0.0, 0.0, 0.0
    last = len(ordered) - 1
    return (
        ordered[min(int(len(ordered) * 0.50), last)],
        ordered[min(int(len(ordered) * 0.95), last)],
        ordered[min(int(len(ordered) * 0.99), last)],
    )


def stat_line(label: str, values: list[float], unit: str = "s") -> None:
    if not values:
        print(f"  {label:<22}: (no data)")
        return
    p50, p95, p99 = percentiles(values)
    print(
        f"  {label:<22}: p50 {p50:6.2f}{unit}   p95 {p95:6.2f}{unit}   "
        f"p99 {p99:6.2f}{unit}   mean {statistics.mean(values):6.2f}{unit}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", help="Only include rows on/after this date (YYYY-MM-DD)")
    args = parser.parse_args()

    pool = _ensure_db_pool()
    if pool is None:
        print("No database connection — cannot build report.")
        return

    where = "WHERE created_at >= %s" if args.since else ""
    params = (args.since,) if args.since else ()

    conn = pool.get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(f"SELECT * FROM recognition_metrics {where}", params)
    metrics = cursor.fetchall()

    cursor.execute(f"SELECT * FROM face_match_events {where}", params)
    events = cursor.fetchall()

    cursor.close()
    conn.close()

    scope = f" since {args.since}" if args.since else ""
    print(f"\nQueuEx live face-recognition report{scope}")
    print("=" * 62)

    # ---- Speed -------------------------------------------------------
    print(f"\n1. Time to recognition   ({len(metrics)} successful links)")
    if metrics:
        stat_line(
            "first seen -> confirmed",
            [m["seconds_to_confirm"] for m in metrics if m["seconds_to_confirm"] is not None],
        )
        stat_line(
            "ready -> linked",
            [m["seconds_to_link"] for m in metrics if m["seconds_to_link"] is not None],
        )
        print()
        print("  'ready' = the later of presence-confirmed and join-intent, i.e.")
        print("  the first moment the system could legitimately issue a number.")
        print("  Rows written before 2026-09-12 measured from FIRST SIGHTING")
        print("  instead, so they include however long the student stood in")
        print("  frame before opening the app — one such row reads 889s. Filter")
        print("  with --since 2026-09-12 for a clean figure, or recompute from")
        print("  the stored first_seen_at / confirmed_at / linked_at columns.")
    else:
        print("  (no links recorded yet)")

    # ---- Live detection rate ----------------------------------------
    print("\n2. Live face-detection rate")
    attempts = sum(m["embed_attempts"] or 0 for m in metrics)
    successes = sum(m["embed_successes"] or 0 for m in metrics)
    if attempts:
        print(f"  frames with a person  : {attempts}")
        print(f"  frames yielding a face: {successes}")
        print(f"  detection rate        : {successes / attempts * 100:.1f}%")
        print(f"  frames per recognition: {attempts / len(metrics):.1f} avg")
    else:
        print("  (no frames recorded yet)")

    # ---- Decision quality -------------------------------------------
    print(f"\n3. Match decisions        ({len(events)} attempts)")
    if events:
        accepted = [e for e in events if e["accepted"]]
        rejected = [e for e in events if not e["accepted"]]
        print(f"  accepted              : {len(accepted)}  ({len(accepted) / len(events) * 100:.1f}%)")
        print(f"  rejected (stayed safe): {len(rejected)}  ({len(rejected) / len(events) * 100:.1f}%)")

        acc_scores = [e["matched_score"] for e in accepted if e["matched_score"] is not None]
        if acc_scores:
            print(
                f"  accepted score range  : {min(acc_scores):.3f} - {max(acc_scores):.3f}"
                f"   (mean {statistics.mean(acc_scores):.3f})"
            )
        acc_margins = [e["margin"] for e in accepted if e["margin"] is not None]
        if acc_margins:
            print(f"  accepted margin min   : {min(acc_margins):.3f}")
    else:
        print("  (no match attempts recorded yet)")

    # ---- Correctness -------------------------------------------------
    print("\n4. Identification correctness")
    print(
        "  Not derivable from logs alone — the system cannot know it linked\n"
        "  the wrong student. Confirm against the beta roster: for each row in\n"
        "  recognition_metrics, check student_id matches who actually stood at\n"
        "  the camera. Any mismatch is a false accept and must be reported."
    )

    print()


if __name__ == "__main__":
    main()
