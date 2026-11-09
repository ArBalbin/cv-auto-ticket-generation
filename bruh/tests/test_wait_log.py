"""
The wait log that lets the wait-time model be checked against real waits and
re-fitted on exact queue counts: what the database stores, what the export
writes, and how the training script reads measured data. No real database.

Run:  .venv/Scripts/python.exe -m pytest tests -v
"""
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "ML"))

import export_wait_log  # noqa: E402
import train_wait_model  # noqa: E402
from database import database_handler  # noqa: E402
from services import queue_service, wait_model  # noqa: E402


# ---------------------------------------------------------------- database

class RecordingCursor:
    def __init__(self, log, row=None):
        self.log, self.row = log, row
        self.rowcount = 1
        self.lastrowid = 7

    def execute(self, sql, params=()):
        self.log.append((" ".join(sql.split()), tuple(params)))

    def fetchone(self):
        return self.row

    def close(self):
        pass


class RecordingPool:
    def __init__(self, row=None):
        self.log, self.row = [], row

    def get_connection(self):
        pool = self

        class Conn:
            def cursor(self, dictionary=False):
                return RecordingCursor(pool.log, pool.row)

            def commit(self):
                pass

            def close(self):
                pass
        return Conn()


@pytest.fixture
def database(monkeypatch):
    def install(columns, row=None):
        pool = RecordingPool(row)
        monkeypatch.setattr(database_handler, "_ensure_db_pool", lambda: pool)
        monkeypatch.setattr(database_handler, "_get_table_columns", lambda cursor, table: set(columns))
        monkeypatch.setattr(database_handler, "_lookup_user_id", lambda cursor, name: None)
        monkeypatch.setattr(database_handler, "_insert_queue_event", lambda *a, **k: None)
        return pool
    return install


def ticket(**extra):
    return {"queue_number": 5001, "short_code": "ABCD-EFGH", "jwt_token": "t", "pdf_path": "p.pdf",
            "expires_at": datetime(2026, 11, 8, 18), **extra}


def test_ticket_row_stores_the_estimate_shown_at_issue(database):
    pool = database({"service_date", "people_ahead_at_issue", "active_counters_at_issue",
                     "predicted_wait_min", "wait_model_version"})
    assert database_handler.save_ticket_record(ticket(
        people_ahead_at_issue=4, active_counters_at_issue=3, predicted_wait_min=1.7, wait_model_version="v1"))

    sql, params = pool.log[0]
    assert "people_ahead_at_issue, active_counters_at_issue, predicted_wait_min, wait_model_version" in sql
    assert params[-4:] == (4, 3, 1.7, "v1")


def test_ticket_row_still_saves_before_the_migration(database):
    pool = database({"service_date"})
    assert database_handler.save_ticket_record(ticket(people_ahead_at_issue=4, predicted_wait_min=1.7))
    assert "people_ahead_at_issue" not in pool.log[0][0]


def test_done_records_when_the_person_was_called(database):
    pool = database({"called_at", "served_by_user_id"}, row=(11, None))
    database_handler.update_queue_status(5001, "served", "staff", called_seconds_ago=270)

    sql, params = pool.log[-1]
    # Against the database's clock, the same one that sets created_at and
    # served_at; and a call time already stored is never overwritten.
    assert "called_at=COALESCE(called_at, NOW() - INTERVAL %s SECOND)" in sql
    assert params == ("served", None, 270, 11)


def test_done_without_a_call_time_leaves_the_column_alone(database):
    pool = database({"called_at"}, row=(11, None))
    database_handler.update_queue_status(5001, "served", None)
    assert "called_at" not in pool.log[-1][0]


# ---------------------------------------------------------------- export

def test_export_writes_the_measured_format():
    row = export_wait_log.to_row({
        "queue_number": 5003, "created_at": datetime(2026, 11, 8, 9, 0, 5),
        "called_at": datetime(2026, 11, 8, 9, 4, 35), "served_at": datetime(2026, 11, 8, 9, 6),
        "people_ahead_at_issue": 4, "active_counters_at_issue": 3,
        "predicted_wait_min": 1.7, "wait_model_version": "2026-11-08",
    })
    assert row["date"] == "2026-11-08" and row["ticket_time"] == "09:00:05" and row["called_time"] == "09:04:35"
    assert set(row) >= train_wait_model.MEASURED_COLUMNS


# ---------------------------------------------------------------- training on measured data

def write_sheet(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "ticket_number", "ticket_time", "called_time",
                                               "finished_time", "people_ahead", "windows_open",
                                               "predicted_wait_min"])
        writer.writeheader()
        writer.writerows(rows)


def sheet_rows(days=4, per_person=0.5):
    """Waits that follow 1 + per_person x people ahead (at three windows) exactly."""
    rows = []
    for d in range(days):
        for i, ahead in enumerate([0, 2, 4, 6, 8, 10]):
            start = datetime(2026, 11, 10 + d, 9, i * 10)
            wait = 1 + per_person * ahead
            rows.append({
                "date": start.strftime("%Y-%m-%d"), "ticket_number": i + 1,
                "ticket_time": start.strftime("%H:%M"),
                "called_time": (start + timedelta(minutes=wait)).strftime("%H:%M:%S"),
                "finished_time": "", "people_ahead": ahead, "windows_open": 3, "predicted_wait_min": 2.0,
            })
    return rows


def test_measured_head_count_is_used_not_reconstructed(tmp_path):
    path = tmp_path / "sheet.csv"
    write_sheet(path, [
        {"date": "2026-11-10", "ticket_number": 90, "ticket_time": "09:00", "called_time": "09:03",
         "finished_time": "09:05", "people_ahead": 2, "windows_open": 2, "predicted_wait_min": ""},
        {"date": "2026-11-10", "ticket_number": "", "ticket_time": "09:01:30", "called_time": "09:01:10",
         "finished_time": "", "people_ahead": 0, "windows_open": "", "predicted_wait_min": "1.1"},
    ])
    rows = train_wait_model.load_transactions(path)
    train_wait_model.add_queue_state(rows)

    first, second = rows
    assert first["people_ahead"] == 2.0, "the count on the sheet, not ticket numbers"
    assert first["wait"] == 3.0 and first["service"] == 2.0
    assert train_wait_model.queue_load(first) == 3.0, "2 ahead at 2 windows counts as 3 at three"
    assert second["wait"] == 0.0, "called before the row was written is no wait, not a negative one"
    assert second["predicted"] == 1.1


def test_training_on_measured_data_recovers_the_line_and_scores_live_estimates(tmp_path, monkeypatch):
    sheet, out = tmp_path / "sheet.csv", tmp_path / "candidate.json"
    write_sheet(sheet, sheet_rows())
    monkeypatch.setattr(sys, "argv", ["train", "--csv", str(sheet), "--out", str(out),
                                      "--current", str(wait_model.MODEL_PATH)])
    train_wait_model.main()

    model = json.loads(out.read_text(encoding="utf-8"))
    assert model["minutes_per_person"] == pytest.approx(0.5, abs=1e-6)
    assert model["base_minutes"] == pytest.approx(1.0, abs=1e-6)
    assert model["training_data"]["people_ahead"] == "measured head count"
    validation = model["validation"]
    assert validation["estimates_shown_live"]["transactions"] == 24
    assert train_wait_model.CURRENT in validation["predictors"], "the model in use is the yardstick"
    assert validation["predictors"][train_wait_model.DEPLOYED]["MAE"] == pytest.approx(0, abs=1e-6)


def test_too_few_days_writes_nothing(tmp_path, monkeypatch):
    sheet, out = tmp_path / "sheet.csv", tmp_path / "candidate.json"
    write_sheet(sheet, sheet_rows(days=2))
    monkeypatch.setattr(sys, "argv", ["train", "--csv", str(sheet), "--out", str(out)])
    with pytest.raises(SystemExit):
        train_wait_model.main()
    assert not out.exists()


# ---------------------------------------------------------------- what the app is told

def test_the_app_is_told_the_usual_range(monkeypatch):
    monkeypatch.setattr(queue_service, "USE_QUEUE_ENGINE", False)
    shown = queue_service.prediction_for_position(11, 2.0, 3)
    low, high = shown["estimated_wait_range_minutes"]
    assert low <= shown["estimated_wait_time_minutes"] <= high
    assert shown["estimated_wait_range_label"] == f"usually {low}–{high} mins"


def test_the_queue_engine_shows_no_unmeasured_range(monkeypatch):
    monkeypatch.setattr(queue_service, "USE_QUEUE_ENGINE", True)
    assert queue_service.prediction_for_position(11, 2.0, 3)["estimated_wait_range_label"] == ""


def test_no_range_for_a_ticket_out_of_the_line():
    shown = queue_service.prediction_for_position(0, 2.0, 3)
    assert shown["estimated_wait_range_label"] == ""


def test_usual_range_never_goes_below_zero():
    for minutes in np.linspace(0, 30, 31):
        low, high = wait_model.usual_range(minutes)
        assert 0 <= low <= minutes <= high
