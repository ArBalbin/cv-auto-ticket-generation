"""
Tests for the trained wait-time model (Algorithm 6): the committed coefficients,
what the queue service and the detector's forecasts do with them, and the
training script's rules against leaking the future into a prediction.
Nothing here touches the live database.

Run:  .venv/Scripts/python.exe -m pytest tests -v
"""
import json
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "ML"))

import train_wait_model  # noqa: E402
from services import prediction_service, queue_service, wait_model  # noqa: E402


def transaction(date, number, created, wait, service=2.0):
    start = datetime.fromisoformat(f"{date} {created}")
    return {
        "date": date,
        "number": number,
        "created": start,
        "wait": float(wait),
        "service": service,
        "called": start + timedelta(minutes=wait),
    }


def test_committed_model_records_its_validation():
    model = json.loads(wait_model.MODEL_PATH.read_text(encoding="utf-8"))
    assert model["base_minutes"] == wait_model.BASE_MINUTES
    assert model["minutes_per_person"] == wait_model.MINUTES_PER_PERSON
    assert model["minutes_per_person"] > 0, "a longer line must mean a longer wait"
    assert train_wait_model.DEPLOYED in model["validation"]["predictors"]


def test_three_counters_gives_the_trained_line_exactly():
    for ahead in (0, 1, 5, 30):
        expected = wait_model.BASE_MINUTES + wait_model.MINUTES_PER_PERSON * ahead
        assert wait_model.predict_minutes(ahead, 3) == pytest.approx(expected)


def test_wait_grows_with_the_line_and_never_goes_below_the_base():
    waits = [wait_model.predict_minutes(n, 3) for n in range(50)]
    assert waits == sorted(waits)
    assert wait_model.predict_minutes(-4, 3) == pytest.approx(wait_model.BASE_MINUTES)


def test_fewer_counters_means_a_longer_wait():
    one, three, five = (wait_model.predict_minutes(10, c) for c in (1, 3, 5))
    assert one > three > five
    assert wait_model.predict_minutes(0, 1) == pytest.approx(wait_model.predict_minutes(0, 5))
    wait_model.predict_minutes(10, 0)   # a bad counter value must not divide by zero


def test_position_means_people_ahead_plus_one(monkeypatch):
    monkeypatch.setattr(queue_service, "USE_QUEUE_ENGINE", False)
    first = queue_service.estimate_wait_for_position(1, 3)
    assert first == pytest.approx(round(wait_model.BASE_MINUTES, 1))
    assert queue_service.estimate_wait_for_position(11, 3) == pytest.approx(
        round(wait_model.predict_minutes(10, 3), 1)
    )


def test_a_ticket_out_of_the_line_has_no_wait():
    prediction = queue_service.prediction_for_position(0, 2.0, 3)
    assert prediction["estimated_wait_time_minutes"] == 0.0
    assert prediction["estimated_wait_time_label"] == "less than 1 min"


def test_service_finish_still_uses_the_measured_service_time():
    prediction = queue_service.prediction_for_position(1, 4.5, 3)
    assert prediction["estimated_service_time_min"] == 4.5


@pytest.fixture
def detector(monkeypatch):
    """A prediction service with empty history, driven by a fake clock, one update a second."""
    monkeypatch.setattr(prediction_service, "historical_counts", prediction_service.deque(maxlen=60))
    monkeypatch.setattr(prediction_service, "historical_timestamps", prediction_service.deque(maxlen=60))
    service = prediction_service.PredictionService({"avg_service_time": 2.0})
    data = {"service_rate": 0.5, "active_counters": 3}
    clock = {"now": 0.0}
    monkeypatch.setattr(prediction_service.time, "time", lambda: clock["now"])
    lock = threading.Lock()

    def run(counts):
        """Feeds one count per second, returning the data after each."""
        seen = []
        for count in counts:
            clock["now"] += 1.0
            service.update(count, data, lock)
            seen.append(dict(data))
        return seen

    return run


def test_detector_forecasts_use_the_model(detector):
    last = detector([6] * 900)[-1]

    # A steady line: every horizon is the wait for that same line.
    expected = round(wait_model.predict_minutes(6, 3), 1)
    assert last["estimated_wait_time"] == expected
    assert last["predicted_wait_5min"] == expected
    assert last["predicted_wait_15min"] == expected
    assert last["predicted_wait_30min"] == expected
    assert last["arrival_rate"] == 0.0


def test_one_person_walking_in_does_not_spike_the_forecast(detector):
    seen = detector([3] * 600 + [4] * 300)
    after = seen[600:]

    assert max(d["predicted_wait_5min"] for d in after) < 3.0
    assert max(d["predicted_wait_15min"] for d in after) < 3.0
    assert max(d["arrival_rate"] for d in after) < 1.0, "one person is not a rush"


def test_a_line_that_keeps_growing_raises_the_forecast(detector):
    growing = [3 + second // 30 for second in range(600)]    # two more people a minute
    last = detector(growing)[-1]

    assert last["predicted_wait_5min"] > last["estimated_wait_time"]
    assert last["arrival_rate"] > 1.0


def test_the_wait_now_follows_the_line_at_once(detector):
    assert detector([3] * 60 + [9])[-1]["estimated_wait_time"] == round(wait_model.predict_minutes(9, 3), 1)


def test_queue_prediction_response_uses_the_model(monkeypatch):
    monkeypatch.setattr(queue_service, "USE_QUEUE_ENGINE", False)
    line = [
        {"queue_number": 5001, "queue_label": "Q5001", "status": "waiting", "position_in_line": 1},
        {"queue_number": 5002, "queue_label": "Q5002", "status": "waiting", "position_in_line": 2},
    ]
    monkeypatch.setattr(queue_service, "live_or_db_active_queue", lambda: line)
    monkeypatch.setattr(queue_service, "runtime_config", lambda: {"active_counters": 3, "avg_service_time": 2.0})
    monkeypatch.setitem(queue_service.state.latest_state, "estimated_wait_time", 0.0)

    response = queue_service.build_queue_prediction()

    assert response["new_arrival"]["estimated_wait_time"] == round(wait_model.predict_minutes(2, 3), 1)
    assert response["active_queue"][0]["estimated_wait_time"] == round(wait_model.predict_minutes(0, 3), 1)
    assert response["active_queue"][1]["estimated_wait_time"] == round(wait_model.predict_minutes(1, 3), 1)


def test_median_fit_recovers_a_line_despite_an_outlier():
    x = np.arange(20.0)
    y = 1.5 + 0.25 * x
    y[3] = 200.0
    base, per_person = train_wait_model.fit_median_line(x, y)
    assert base == pytest.approx(1.5, abs=1e-6)
    assert per_person == pytest.approx(0.25, abs=1e-6)


def test_people_ahead_uses_only_what_had_already_happened():
    rows = [
        transaction("2026-10-05", 5, "09:00:00", 2),    # called 09:02
        transaction("2026-10-05", 8, "09:01:00", 30),   # called 09:31: not yet at 09:05
        transaction("2026-10-05", 9, "09:05:00", 1),
    ]
    train_wait_model.add_queue_state(rows)
    assert rows[0]["people_ahead"] is None, "nothing had been called yet that day"
    assert rows[2]["people_ahead"] == 3.0, "9 - 5 - 1, ticket 8 is still waiting"


def test_people_ahead_does_not_look_at_other_days():
    rows = [
        transaction("2026-10-05", 100, "09:00:00", 1),
        transaction("2026-10-06", 3, "09:30:00", 1),
    ]
    train_wait_model.add_queue_state(rows)
    assert rows[1]["people_ahead"] is None


def test_validation_never_trains_on_the_day_it_tests():
    rows = [transaction(f"2026-10-0{d}", n, "09:00:00", 1) for d in (5, 6, 7) for n in (1, 2)]
    seen = []

    def spy(train, test):
        assert {r["date"] for r in train}.isdisjoint({r["date"] for r in test})
        seen.append(test[0]["date"])
        return np.zeros(len(test))

    train_wait_model.leave_one_day_out(rows, spy)
    assert seen == ["2026-10-05", "2026-10-06", "2026-10-07"]
