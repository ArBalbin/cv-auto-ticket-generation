"""FIFO scheduling approximation for heterogeneous department service times."""
import math


def simulate_department_wait_seconds(queue_students, active_cashier_windows,
                                     busy_cashier_states, service_time_predictor):
    """Replay only students ahead of the target against active cashier slots.

    Busy states carry a nonnegative ``residual_seconds`` estimate. Idle
    windows are available immediately. This returns the next time a cashier
    becomes available after all listed students ahead have been assigned.
    """
    if isinstance(active_cashier_windows, bool) or not isinstance(active_cashier_windows, int) or active_cashier_windows < 0:
        raise ValueError("active_cashier_windows must be a nonnegative integer")
    if active_cashier_windows == 0:
        return None
    states = list(busy_cashier_states or [])
    if len(states) > active_cashier_windows:
        raise ValueError("busy cashier states cannot exceed active windows")
    slots = [0.0] * (active_cashier_windows - len(states))
    for state in states:
        residual = float(state["residual_seconds"])
        if not math.isfinite(residual) or residual < 0:
            raise ValueError("busy residual time must be finite and nonnegative")
        slots.append(residual)
    for student in queue_students:
        slots.sort()
        duration = float(service_time_predictor(student))
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("predicted service duration must be finite and nonnegative")
        slots[0] += duration
    return float(min(slots))
