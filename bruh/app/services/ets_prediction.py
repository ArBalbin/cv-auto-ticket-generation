"""Cached integration for the separately trained QueueFlow ETS pipeline."""
import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from core.config import SCHOOL_UTC_OFFSET_HOURS
from services.department_config import DEPARTMENT_SET, normalize_department

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.getenv("QUEUEFLOW_ETS_MODEL_DIR", _ROOT / "models"))
MODEL_VERSION = "department_average_fallback"


@lru_cache(maxsize=1)
def _load_model():
    import joblib
    artifact_path = MODEL_DIR / "ets_model.joblib"
    metadata_path = MODEL_DIR / "model_metadata.json"
    artifact = joblib.load(artifact_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("input_feature_names") != ["department", "hour", "day_of_week"]:
        raise ValueError("ETS artifact features do not match the deployed adapter")
    if not hasattr(artifact.get("pipeline"), "predict"):
        raise ValueError("ETS artifact does not contain a prediction pipeline")
    return artifact["pipeline"], str(metadata.get("model_version", "unknown"))


@lru_cache(maxsize=128)
def _predict_cached(department: str, hour: int, day_of_week: int) -> tuple[float, str]:
    import pandas as pd
    pipeline, version = _load_model()
    row = pd.DataFrame([{
        "department": department,
        "hour": hour,
        "day_of_week": day_of_week,
    }])
    seconds = float(pipeline.predict(row)[0]) * 60.0
    if not math.isfinite(seconds):
        raise ValueError("ETS model returned a non-finite prediction")
    return max(0.0, seconds), version


def predict_service_time(department: str | None, fallback_seconds: float,
                         transaction_time: datetime | None = None,
                         fallback_method: str = "global_average_fallback") -> dict:
    """Return nonnegative ETS seconds and its source; artifact targets minutes."""
    school_timezone = timezone(timedelta(hours=SCHOOL_UTC_OFFSET_HOURS))
    if transaction_time is None:
        now = datetime.now(school_timezone)
    elif transaction_time.tzinfo is not None:
        now = transaction_time.astimezone(school_timezone)
    else:
        # Training timestamps are naive school-local wall time.
        now = transaction_time
    try:
        code = normalize_department(department)
    except ValueError:
        logger.warning("Unknown department code; using measured service-time fallback")
        code = None
    if code is not None:
        try:
            seconds, version = _predict_cached(code, now.hour, now.weekday())
            return {"seconds": seconds, "method": "ml_department_fifo", "model_version": version}
        except Exception as exc:
            logger.warning("ETS model prediction unavailable; using measured average (%s)", type(exc).__name__)
    fallback = float(fallback_seconds)
    if not math.isfinite(fallback):
        fallback = 0.0
    return {"seconds": max(0.0, fallback),
            "method": fallback_method, "model_version": None}


def model_status() -> dict:
    try:
        _, version = _load_model()
        return {"available": True, "version": version}
    except Exception as exc:
        logger.warning("ETS model could not be loaded (%s)", type(exc).__name__)
        return {"available": False, "version": None}
