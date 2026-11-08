"""
Trains the wait-time model (Algorithm 6) and writes Model/wait_time_model.json.

The model is a median regression of the waiting time on the number of people
ahead when the ticket was issued:

    wait = base_minutes + minutes_per_person x people_ahead

The aiprediction repo tried time of day and department as inputs, and nothing
beat a constant 2 minutes, because those inputs say nothing about the queue.
The people ahead do, and the live system knows them.

Two input formats are accepted, told apart by their columns:

  - The October cashier log (dataset/ncf_cashier_department_clean.csv in the
    aiprediction repo): ticket times and waits, but no queue count. People
    ahead is reconstructed from ticket numbers, which overcounts.
  - Measured data, with a real head count per ticket: either recorded by hand
    at the cashier (ML/wait_data_collection/) or exported from the system's own
    log (ML/export_wait_log.py). Columns: date, ticket_time, called_time,
    people_ahead, and optionally ticket_number, finished_time, windows_open,
    predicted_wait_min. When windows_open is recorded, the people ahead are
    scaled to three windows, the same scaling the system applies.

The files stay out of this repo (ML/data/ is git-ignored); pass --csv.

Every predictor is scored by leave-one-day-out validation: fitted on the other
days and tested on a whole unseen day, because tickets from one day share that
day's congestion and a random split would leak it. --test-from adds a
chronological check: fit on the days before a date, test once on the rest.

Usage:  .venv/Scripts/python.exe ML/train_wait_model.py [--csv FILE] [--test-from YYYY-MM-DD]
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "ML" / "data" / "wait_time" / "ncf_cashier_department_clean.csv"
DEFAULT_OUT = ROOT / "Model" / "wait_time_model.json"

REFERENCE_COUNTERS = 3        # Windows 3, 4 and 5 serve the shared queue
ARRIVAL_WINDOW_MINUTES = 15
CAP_CANDIDATES = (8, 12, 20, 40, None)
RANGE_QUANTILES = (10, 90)    # the wait falls in this range about 80% of the time
BOOTSTRAP_SAMPLES = 4000
MIN_DAYS = 3                  # leave-one-day-out needs several days to mean anything
MEASURED_COLUMNS = {"date", "ticket_time", "called_time", "people_ahead"}


def _optional_float(value: str | None) -> float | None:
    value = (value or "").strip()
    return float(value) if value else None


def load_transactions(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        measured = MEASURED_COLUMNS <= set(reader.fieldnames or [])
        for index, r in enumerate(reader):
            if measured:
                created = datetime.fromisoformat(f"{r['date']} {r['ticket_time'].strip()}")
                called = datetime.fromisoformat(f"{r['date']} {r['called_time'].strip()}")
                finished = (r.get("finished_time") or "").strip()
                windows = _optional_float(r.get("windows_open"))
                rows.append({
                    "date": r["date"],
                    "number": int(r["ticket_number"]) if (r.get("ticket_number") or "").strip() else index,
                    "created": created,
                    # The system can call someone a few seconds before their
                    # ticket row is written; that is no wait, not a negative one.
                    "wait": max(0.0, (called - created).total_seconds() / 60),
                    "service": (
                        (datetime.fromisoformat(f"{r['date']} {finished}") - called).total_seconds() / 60
                        if finished else None
                    ),
                    "called": called,
                    "measured": True,
                    "people_ahead": float(r["people_ahead"]),
                    "windows_open": int(windows) if windows else None,
                    "predicted": _optional_float(r.get("predicted_wait_min")),
                })
            else:
                created = datetime.fromisoformat(r["ticket_creation_datetime"])
                wait = float(r["waiting_time_minutes"])
                rows.append({
                    "date": r["date"],
                    "number": int(r["transaction_number"]),
                    "created": created,
                    "wait": wait,
                    "service": _optional_float(r.get("service_time_minutes")),
                    "called": created + timedelta(minutes=wait),
                    "measured": False,
                    "people_ahead": None,
                    "windows_open": None,
                    "predicted": None,
                })
    rows.sort(key=lambda r: (r["created"], r["number"]))
    return rows


def add_queue_state(rows: list[dict]) -> None:
    """
    Adds what was knowable at the moment each ticket was issued, from strictly
    earlier events only.

    A measured head count is kept as it is. Without one:

    people_ahead = ticket number - highest number already called - 1

    This is a reconstruction, not a head count: tickets that were issued but
    never served at these windows are counted, so it overestimates. Tickets
    issued before any observed ticket was called that day get None.
    """
    for r in rows:
        same_day = [o for o in rows if o["date"] == r["date"] and o is not r]
        t = r["created"]

        if not r.get("measured"):
            called = [o["number"] for o in same_day if o["called"] < t]
            r["people_ahead"] = float(max(0, r["number"] - max(called) - 1)) if called else None

        recent = [o for o in same_day if t - timedelta(minutes=ARRIVAL_WINDOW_MINUTES) <= o["created"] < t]
        recent = recent or [o for o in same_day if o["created"] < t]
        r["arrival_rate"] = None
        if recent:
            first = min(recent, key=lambda o: o["created"])
            minutes = (t - first["created"]).total_seconds() / 60
            if minutes > 0:
                r["arrival_rate"] = max(0, r["number"] - first["number"]) / minutes

        finished = sorted(
            (o for o in same_day
             if o["service"] is not None and o["called"] + timedelta(minutes=o["service"]) < t),
            key=lambda o: o["called"],
        )
        r["recent_service"] = float(np.mean([o["service"] for o in finished[-10:]])) if len(finished) >= 3 else None


def queue_load(r: dict) -> float | None:
    """People ahead, scaled to three windows when the open windows were
    recorded: 6 ahead with 2 windows open counts as 9."""
    if r["people_ahead"] is None:
        return None
    windows = r.get("windows_open")
    return r["people_ahead"] * REFERENCE_COUNTERS / windows if windows else r["people_ahead"]


def fit_median_line(x, y) -> tuple[float, float]:
    """Least-absolute-deviation line, as a linear program: returns (intercept, slope)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    design = np.column_stack([np.ones(n), x])
    cost = np.r_[0.0, 0.0, np.ones(2 * n)]
    a_eq = np.hstack([design, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * 2 + [(0, None)] * (2 * n)
    solution = linprog(cost, A_eq=a_eq, b_eq=y, bounds=bounds, method="highs")
    if not solution.success:
        raise RuntimeError(f"median regression failed: {solution.message}")
    return float(solution.x[0]), float(solution.x[1])


def fit_on_people_ahead(rows: list[dict], cap: float | None = None) -> tuple[float, float]:
    known = [r for r in rows if r["people_ahead"] is not None]
    x = [queue_load(r) if cap is None else min(queue_load(r), cap) for r in known]
    return fit_median_line(x, [r["wait"] for r in known])


def leave_one_day_out(rows: list[dict], predictor) -> np.ndarray:
    predictions = np.zeros(len(rows))
    for day in sorted({r["date"] for r in rows}):
        train = [r for r in rows if r["date"] != day]
        held_out = [i for i, r in enumerate(rows) if r["date"] == day]
        predictions[held_out] = predictor(train, [rows[i] for i in held_out])
    return np.clip(predictions, 0, None)


# ── Predictors: each takes (training rows, rows to predict) ──────────────────

def constant_median(train, test):
    return np.full(len(test), np.median([r["wait"] for r in train]))


def people_ahead_line(cap=None):
    def predict(train, test):
        base, per_person = fit_on_people_ahead(train, cap)
        fallback = np.median([r["wait"] for r in train])
        return np.array([
            fallback if r["people_ahead"] is None
            else base + per_person * (queue_load(r) if cap is None else min(queue_load(r), cap))
            for r in test
        ])
    return predict


def people_ahead_line_nested_cap(train, test):
    """Picks the cap by an inner leave-one-day-out on the training days only."""
    y = np.array([r["wait"] for r in train])
    errors = {cap: np.abs(y - leave_one_day_out(train, people_ahead_line(cap))).mean() for cap in CAP_CANDIDATES}
    return people_ahead_line(min(errors, key=errors.get))(train, test)


def previous_position_rule(train, test):
    """The rule the system used before this model: whole batches ahead x service time."""
    service = np.median([r["service"] for r in train if r["service"] is not None])
    return np.array([((queue_load(r) or 0) // REFERENCE_COUNTERS) * service for r in test])


def previous_queueing_formula(train, test):
    """The manuscript's current-wait formula: Q/lambda + 1/mu when rho < 1, else Q/(mu c)."""
    default_service = np.mean([r["service"] for r in train if r["service"] is not None])
    predictions = []
    for r in test:
        queue = queue_load(r) or 0.0
        service = r["recent_service"] or default_service
        arrival = r["arrival_rate"] or 0.0
        mu = 1.0 / service
        if arrival / (REFERENCE_COUNTERS * mu) < 1:
            predictions.append(queue / max(arrival, 0.1) + 1 / mu)
        else:
            predictions.append(queue / (mu * REFERENCE_COUNTERS))
    return np.array(predictions)


PREDICTORS = {
    "Median regression on people ahead (deployed)": people_ahead_line(),
    "Same, with a cap chosen on the training days": people_ahead_line_nested_cap,
    "Constant median (aiprediction production model)": constant_median,
    "Previous position rule": previous_position_rule,
    "Previous queueing formula": previous_queueing_formula,
}
DEPLOYED = "Median regression on people ahead (deployed)"
CONSTANT = "Constant median (aiprediction production model)"
CURRENT = "Model in use now, applied unchanged"


def current_model_predictor(model_path: Path, csv_path: Path):
    """
    The model already deployed, applied as it is with no re-fitting: the fair
    yardstick for a re-fit on new data. None when there is no model yet, or
    when it was trained on this same file, which would score it on its own
    training data.
    """
    if not model_path.exists():
        return None
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if model.get("training_data", {}).get("sha256") == hashlib.sha256(csv_path.read_bytes()).hexdigest():
        return None
    base, per_person = model["base_minutes"], model["minutes_per_person"]

    def predict(train, test):
        return np.array([base + per_person * (queue_load(r) or 0.0) for r in test])
    return predict


def score(y: np.ndarray, predicted: np.ndarray) -> dict:
    errors = np.abs(y - predicted)
    nonzero = y > 0
    return {
        "MAE": round(float(errors.mean()), 3),
        "RMSE": round(float(np.sqrt(((y - predicted) ** 2).mean())), 3),
        "R2": round(float(1 - ((y - predicted) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 3),
        "MAPE_pct_nonzero_waits": (
            round(float((errors[nonzero] / y[nonzero]).mean() * 100), 1) if nonzero.any() else None
        ),
        "WAPE_pct": round(float(errors.sum() / y.sum() * 100), 1) if y.sum() > 0 else None,
        "within_2_min_pct": round(float((errors <= 2).mean() * 100), 1),
        "within_5_min_pct": round(float((errors <= 5).mean() * 100), 1),
    }


def improvement_over(baseline_errors, errors, dates) -> dict:
    """MAE reduction with a day-level bootstrap: days, not tickets, are the independent unit."""
    days = sorted(set(dates))
    index = {d: np.where(dates == d)[0] for d in days}
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        chosen = np.concatenate([index[d] for d in rng.choice(days, len(days))])
        samples.append(baseline_errors[chosen].mean() - errors[chosen].mean())
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "minutes": round(float(baseline_errors.mean() - errors.mean()), 3),
        "ci95": [round(float(low), 3), round(float(high), 3)],
    }


def chronological_test(rows: list[dict], test_from: str) -> dict:
    """Fit on the days before `test_from`, test once on the rest: the later days
    are unseen in time as well as in content."""
    train = [r for r in rows if r["date"] < test_from]
    test = [r for r in rows if r["date"] >= test_from]
    if not train or not test:
        raise SystemExit(f"--test-from {test_from} leaves no training or no test days")
    y = np.array([r["wait"] for r in test])
    base, per_person = fit_on_people_ahead(train)
    return {
        "train_days": sorted({r["date"] for r in train}),
        "test_days": sorted({r["date"] for r in test}),
        "test_transactions": len(test),
        "fit_on_training_days": {"base_minutes": round(base, 4), "minutes_per_person": round(per_person, 4)},
        "predictors": {
            name: score(y, np.clip(PREDICTORS[name](train, test), 0, None)) for name in (DEPLOYED, CONSTANT)
        },
    }


def print_table(results: dict) -> None:
    print(f"\n{'Predictor':50s} {'MAE':>7} {'RMSE':>7} {'R2':>8} {'<=2min':>7} {'<=5min':>7}")
    for name, m in results.items():
        print(f"{name:50s} {m['MAE']:7.3f} {m['RMSE']:7.3f} {m['R2']:8.3f} "
              f"{m['within_2_min_pct']:6.1f}% {m['within_5_min_pct']:6.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the wait-time model.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--test-from", help="also fit on the days before this date and test on the rest")
    parser.add_argument("--current", type=Path, default=DEFAULT_OUT,
                        help="the model in use, scored unchanged on new data")
    args = parser.parse_args()

    rows = load_transactions(args.csv)
    add_queue_state(rows)
    y = np.array([r["wait"] for r in rows])
    dates = np.array([r["date"] for r in rows])
    days = sorted(set(dates))
    measured = all(r["measured"] for r in rows)
    print(f"{len(rows)} transactions over {len(days)} days; people ahead "
          f"{'measured' if measured else 'reconstructed from ticket numbers'}")
    if len(days) < MIN_DAYS:
        raise SystemExit(f"Only {len(days)} day(s) of data; at least {MIN_DAYS} are needed. Nothing written.")

    predictors = dict(PREDICTORS)
    if not any(r["service"] is not None for r in rows):
        # Both previous rules need service times; a sheet without finish times has none.
        del predictors["Previous position rule"], predictors["Previous queueing formula"]
    current = current_model_predictor(args.current, args.csv)
    if current:
        predictors[CURRENT] = current
    predictions = {name: leave_one_day_out(rows, p) for name, p in predictors.items()}
    results = {name: score(y, p) for name, p in predictions.items()}
    print_table(results)
    if current:
        print(f"Replace the model in use only if the re-fit beats '{CURRENT}' and the constant.")

    errors = {name: np.abs(y - p) for name, p in predictions.items()}
    against_constant = improvement_over(errors[CONSTANT], errors[DEPLOYED], dates)
    print(f"\nMAE reduction against the constant: {against_constant['minutes']:+.3f} min, "
          f"95% CI {against_constant['ci95']}")

    # What the system itself told students, against what then happened.
    shown = [i for i, r in enumerate(rows) if r["predicted"] is not None]
    live = None
    if shown:
        live = score(y[shown], np.array([rows[i]["predicted"] for i in shown]))
        live["transactions"] = len(shown)
        print(f"\nEstimates the system showed ({len(shown)} tickets): MAE {live['MAE']:.3f}, "
              f"within 2 min {live['within_2_min_pct']:.1f}%, within 5 min {live['within_5_min_pct']:.1f}%")

    chronological = None
    if args.test_from:
        chronological = chronological_test(rows, args.test_from)
        print(f"\nChronological test, {chronological['test_days']}:")
        print_table(chronological["predictors"])

    per_day = {
        d: {
            "tickets": int((dates == d).sum()),
            "deployed_MAE": round(float(errors[DEPLOYED][dates == d].mean()), 3),
            "constant_MAE": round(float(errors[CONSTANT][dates == d].mean()), 3),
        }
        for d in days
    }

    # The usual spread around a prediction, from errors on unseen days.
    residuals = y - predictions[DEPLOYED]
    lower, upper = np.percentile(residuals, RANGE_QUANTILES)
    inside = (y >= np.clip(predictions[DEPLOYED] + lower, 0, None)) & (y <= predictions[DEPLOYED] + upper)

    base, per_person = fit_on_people_ahead(rows)
    print(f"\nFinal model, all days: wait = {base:.3f} + {per_person:.3f} x people_ahead")
    print(f"Usual range: {lower:+.2f} to {upper:+.2f} min (held-out coverage {inside.mean():.1%})")

    args.out.write_text(json.dumps({
        "model": "Median regression of waiting time on people ahead",
        "version": datetime.now().strftime("%Y-%m-%d"),
        "base_minutes": round(base, 4),
        "minutes_per_person": round(per_person, 4),
        "reference_counters": REFERENCE_COUNTERS,
        "usual_range": {
            "lower_offset_min": round(float(lower), 2),
            "upper_offset_min": round(float(upper), 2),
            "nominal_coverage": 0.8,
            "heldout_coverage": round(float(inside.mean()), 3),
        },
        "training_data": {
            "file": args.csv.name,
            "sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
            "people_ahead": "measured head count" if measured else "reconstructed from ticket numbers",
            "transactions": len(rows),
            "transactions_fitted": sum(r["people_ahead"] is not None for r in rows),
            "days": days,
        },
        "validation": {
            "scheme": f"leave-one-day-out, {len(days)} folds, pooled over all {len(rows)} held-out predictions",
            "predictors": results,
            "mae_reduction_against_constant": against_constant,
            "per_day": per_day,
            "estimates_shown_live": live,
            "chronological_test": chronological,
        },
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
