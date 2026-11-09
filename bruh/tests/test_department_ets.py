import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from services.department_config import DEPARTMENTS, normalize_department
from services import ets_prediction
from services.ets_prediction import model_status, predict_service_time
from services.queue_simulator import simulate_department_wait_seconds
from routers import students
from services import queue_service
from core import security


def test_bundled_trained_artifact_loads_and_predicts_for_each_supported_department():
    assert model_status() == {"available": True, "version": "1.0.0"}
    for department in DEPARTMENTS:
        prediction = predict_service_time(department, 180, datetime(2026, 11, 9, 10, 30))
        assert prediction["method"] == "ml_department_fifo"
        assert prediction["model_version"] == "1.0.0"
        assert math.isfinite(prediction["seconds"]) and prediction["seconds"] >= 0


def test_adapter_matches_direct_artifact_inference_for_department_features():
    import pandas as pd

    pipeline, version = ets_prediction._load_model()
    now = datetime(2026, 11, 9, 10, 30)
    for department in DEPARTMENTS:
        row = pd.DataFrame([{"department": department, "hour": now.hour,
                             "day_of_week": now.weekday()}])
        direct_seconds = max(0.0, float(pipeline.predict(row)[0]) * 60.0)
        adapted = predict_service_time(department, 180, now)
        assert adapted["model_version"] == version
        assert math.isclose(adapted["seconds"], direct_seconds, rel_tol=0, abs_tol=1e-9)


def test_model_artifact_is_loaded_once_and_reused():
    ets_prediction._load_model.cache_clear()
    first = ets_prediction._load_model()
    second = ets_prediction._load_model()
    info = ets_prediction._load_model.cache_info()
    assert first is second
    assert info.misses == 1 and info.hits == 1


def test_timezone_aware_input_is_converted_to_training_school_time(monkeypatch):
    captured = []
    monkeypatch.setattr(ets_prediction, "SCHOOL_UTC_OFFSET_HOURS", 8)
    monkeypatch.setattr(ets_prediction, "_predict_cached",
                        lambda department, hour, weekday: captured.append((hour, weekday)) or (60, "qa"))
    result = predict_service_time("CCS", 180, datetime(2026, 11, 9, 2, 30, tzinfo=timezone.utc))
    assert captured == [(10, 0)]
    assert result["seconds"] == 60


def test_missing_model_uses_fallback_without_failing_prediction(tmp_path, monkeypatch):
    monkeypatch.setattr(ets_prediction, "MODEL_DIR", tmp_path)
    ets_prediction._load_model.cache_clear()
    try:
        assert model_status()["available"] is False
        prediction = predict_service_time("CCS", 121)
        assert prediction["seconds"] == 121
        assert prediction["method"] == "global_average_fallback"
    finally:
        ets_prediction._load_model.cache_clear()


def test_nonfinite_model_output_uses_fallback(monkeypatch):
    class InvalidPipeline:
        def predict(self, _row):
            return [float("inf")]

    monkeypatch.setattr(ets_prediction, "_load_model", lambda: (InvalidPipeline(), "qa-invalid"))
    ets_prediction._predict_cached.cache_clear()
    prediction = predict_service_time("CCS", 123, datetime(2026, 11, 9, 10, 30))
    assert prediction["seconds"] == 123
    assert prediction["method"] == "global_average_fallback"


def test_unknown_department_uses_safe_average_fallback():
    prediction = predict_service_time("UNKNOWN", 123)
    assert prediction["seconds"] == 123
    assert prediction["method"] == "global_average_fallback"
    assert normalize_department(None) is None


def test_department_profile_update_validates_and_updates_only_dependency_student(monkeypatch):
    updated = []
    monkeypatch.setattr(students, "update_student_department",
                        lambda student_id, department: updated.append((student_id, department)) or True)
    response = students.update_my_department(students.DepartmentBody(department="ccs"), student_id=41)
    assert response == {"success": True, "department": "CCS"}
    assert updated == [(41, "CCS")]
    with pytest.raises(HTTPException) as exc:
        students.update_my_department(students.DepartmentBody(department="NOT_A_DEPT"), student_id=41)
    assert exc.value.status_code == 422


def test_student_profile_response_includes_nullable_department():
    profile = students._student_public_profile({"id": 5, "school_id": "S-5", "department": "CTED"})
    assert profile["department"] == "CTED"


def test_training_export_admin_gate_rejects_staff_without_admin_role(monkeypatch):
    state = {"role": "admin"}

    class FakeCursor:
        def execute(self, _sql, _params):
            pass

        def fetchone(self):
            return dict(state)

        def close(self):
            pass

    class FakeConnection:
        def cursor(self, dictionary=False):
            return FakeCursor()

        def close(self):
            pass

    class FakePool:
        def get_connection(self):
            return FakeConnection()

    request = Request({"type": "http", "headers": [], "query_string": b""})
    monkeypatch.setattr(security, "require_staff", lambda _request: "qa-staff")
    monkeypatch.setattr(security, "get_db_pool", lambda: FakePool())
    assert security.require_admin(request) == "qa-staff"
    state["role"] = "staff"
    with pytest.raises(HTTPException) as exc:
        security.require_admin(request)
    assert exc.value.status_code == 403


def test_department_fifo_accounts_for_busy_windows_and_ignores_people_after_target():
    departments = {"CCS": 60, "CBM": 120, "CHS": 180}
    students_ahead = [{"department": "CCS"}, {"department": "CBM"}, {"department": "CHS"}]
    wait = simulate_department_wait_seconds(
        students_ahead,
        2,
        [{"residual_seconds": 30}],
        lambda student: departments[student["department"]],
    )
    # Busy window clears at 30s; CCS runs until 60, CBM until 150, and CHS
    # until 240. Both cashiers are then available for the target.
    assert wait == 150


def test_zero_active_cashiers_is_unavailable():
    assert simulate_department_wait_seconds([], 0, [], lambda _: 60) is None


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -1])
def test_fifo_rejects_invalid_service_durations(duration):
    with pytest.raises(ValueError):
        simulate_department_wait_seconds([{}], 1, [], lambda _: duration)


def _reference_fifo_wait(durations, cashiers):
    """Small event-scheduling oracle kept independent of production simulator."""
    available_at = [0.0] * cashiers
    for duration in durations:
        window = min(range(cashiers), key=lambda index: (available_at[index], index))
        available_at[window] += duration
    return min(available_at)


@pytest.mark.parametrize("cashiers", [1, 2, 3, 4, 5])
def test_mixed_department_fifo_matches_independent_reference_for_one_to_five_cashiers(cashiers):
    durations_by_department = {"CCS": 118, "CBM": 180, "CHS": 240, "CCJE": 150}
    departments = ["CCS", "CBM", "CBM", "CHS", "CCJE"]
    durations = [durations_by_department[department] for department in departments]
    actual = simulate_department_wait_seconds(
        [{"department": department} for department in departments],
        cashiers,
        [],
        lambda student: durations_by_department[student["department"]],
    )
    expected = _reference_fifo_wait(durations, cashiers)
    assert actual == expected
    if cashiers == 1:
        assert actual == 868


def test_fifo_occupied_windows_dispatch_to_earliest_available_cashier():
    durations = [50, 70, 80, 90]
    actual = simulate_department_wait_seconds(
        [{"duration": value} for value in durations],
        3,
        [{"residual_seconds": 120}, {"residual_seconds": 60}],
        lambda student: student["duration"],
    )
    # Starts are assigned at 0, 50, 60, and 120 seconds; the three windows
    # next become free at 140, 120, and 210 seconds.
    assert actual == 120


def test_ticket_prediction_keeps_ets_ewt_and_completion_distinct(monkeypatch):
    monkeypatch.setattr(queue_service, "USE_QUEUE_ENGINE", True)
    monkeypatch.setattr(queue_service, "live_or_db_active_queue", lambda: [
        {"queue_number": 7101, "position_in_line": 1, "status": "waiting"},
        {"queue_number": 7102, "position_in_line": 2, "status": "waiting"},
    ])
    records = [
        {"queue_number": 7101, "department_at_issue": "CCS"},
        {"queue_number": 7102, "department_at_issue": "CBM"},
    ]
    monkeypatch.setattr(queue_service, "_waiting_queue_records", lambda: records)
    monkeypatch.setattr(queue_service, "measure_department_service_time", lambda _: None)
    monkeypatch.setattr(queue_service, "counters_closed_until", lambda: None)
    monkeypatch.setattr(queue_service.ets_prediction, "predict_service_time", lambda dep, fallback, **kwargs: {
        "seconds": {"CCS": 100, "CBM": 70}[dep],
        "method": "ml_department_fifo",
        "model_version": "qa-test",
    })

    prediction = queue_service.prediction_for_position(2, 2.0, 1, "CBM")
    assert prediction["estimated_wait_time_seconds"] == 100
    assert prediction["estimated_service_time_seconds"] == 70
    assert prediction["estimated_completion_duration_seconds"] == 170
