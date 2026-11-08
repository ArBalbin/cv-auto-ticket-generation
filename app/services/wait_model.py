"""
The trained wait-time model (Algorithm 6): a median regression of the observed
cashier wait on the number of people ahead, fitted by ML/train_wait_model.py.

The coefficients are read from JSON rather than a scikit-learn pickle. Loading
the pickled models and their pandas input cost 108 MB of resident memory on top
of numpy, measured locally, and the server has 512 MB in total. This one-line
model needs none of it, and a pickle would also tie the server to the exact
scikit-learn version it was saved with.

Imported by both the backend and the camera machine's detector.
"""
import json
import math
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[2] / "Model" / "wait_time_model.json"

_model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
BASE_MINUTES = float(_model["base_minutes"])
MINUTES_PER_PERSON = float(_model["minutes_per_person"])
REFERENCE_COUNTERS = int(_model["reference_counters"])
MODEL_VERSION = str(_model.get("version", "unknown"))
RANGE_LOWER = float(_model["usual_range"]["lower_offset_min"])
RANGE_UPPER = float(_model["usual_range"]["upper_offset_min"])


def predict_minutes(people_ahead: float, counters: int = REFERENCE_COUNTERS) -> float:
    """
    Expected wait for someone with `people_ahead` in front of them.

    The model was trained with three windows open. Staff can change the number
    of counters, and fewer counters clear the same line more slowly, so the
    per-person term scales with it. That scaling is a design assumption: the
    data cannot validate it. At three counters the output is the model exactly.
    """
    ahead = max(0.0, float(people_ahead))
    counter_scale = REFERENCE_COUNTERS / max(1, int(counters))
    return BASE_MINUTES + MINUTES_PER_PERSON * ahead * counter_scale


def usual_range(minutes: float) -> tuple[int, int]:
    """
    Where the wait usually falls around an estimate, in whole minutes: the
    spread of errors on unseen days, which held about 8 waits in 10. A single
    number hides how far off it can be; the range says so.
    """
    low = max(0, math.floor(minutes + RANGE_LOWER))
    return low, max(low, math.ceil(minutes + RANGE_UPPER))
