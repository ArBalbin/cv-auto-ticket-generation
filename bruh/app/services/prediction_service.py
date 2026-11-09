# app/services/prediction_service.py

import time
import numpy as np
from collections import deque

from services import wait_model


# ── Shared history (written by update_queue_metrics, read by predictors) ──────
# One sample every 10 seconds, so the 60 samples cover the last 10 minutes.
# Sampling every frame kept only about 12 seconds of history: one person walking
# in looked like a queue growing by the second, and the trend projected from it
# showed "in 5 min: 34 mins" for a few seconds on a line of four.
SAMPLE_INTERVAL_SECONDS = 10
historical_counts:     deque = deque(maxlen=60)
historical_timestamps: deque = deque(maxlen=60)


class PredictionService:

    def __init__(self, queue_config: dict):
        self.cfg = queue_config


    # DENSITY

    def calculate_density(self, persons: list, width: int, height: int,
                          grid=(3, 3)) -> tuple[float, float]:
        """
        Returns (avg_density, max_density) over a grid of cells.
        Density = number of person centres per grid cell.
        """
        if not persons:
            return 0.0, 0.0
        grid_h = max(height // grid[0], 1)
        grid_w = max(width  // grid[1], 1)
        counts = np.zeros(grid)
        for (x1, y1, x2, y2) in persons:
            r = min((y1 + y2) // 2 // grid_h, grid[0] - 1)
            c = min((x1 + x2) // 2 // grid_w, grid[1] - 1)
            counts[r, c] += 1
        return float(np.mean(counts)), float(np.max(counts))


    # ARRIVAL RATE


    def calculate_arrival_rate(self) -> float:
        """
        Estimate arrival rate in people/minute from the recent count history.
        Uses up to the last 30 samples.
        """
        n = min(30, len(historical_counts))
        if n < 2:
            return 0.0
        counts = list(historical_counts)
        dt = (historical_timestamps[-1] - historical_timestamps[-n]) / 60.0
        return max(0.0, (counts[-1] - counts[-n]) / dt) if dt > 0 else 0.0


    # TREND SLOPE (shared utility)


    def _trend_slope(self) -> float:
        """
        Linear regression slope over recent queue counts (people/sample).
        Positive → queue growing. Negative → queue shrinking.
        Returns 0.0 when insufficient history.
        """
        counts = list(historical_counts)
        if len(counts) < 5:
            return 0.0
        window = counts[-20:]          # cap at 20 samples for responsiveness
        m = len(window)
        x = np.arange(m, dtype=float)
        x -= x.mean()
        y = np.array(window, dtype=float)
        denom = np.dot(x, x)
        return float(np.dot(x, y) / denom) if denom > 1e-9 else 0.0

    def _samples_per_second(self) -> float:
        """Actual sampling rate from recent history timestamps."""
        times = list(historical_timestamps)
        if len(times) < 2:
            return 1.0 / SAMPLE_INTERVAL_SECONDS
        n       = min(30, len(times) - 1)
        elapsed = times[-1] - times[-n - 1]
        return n / elapsed if elapsed > 0 else 1.0 / SAMPLE_INTERVAL_SECONDS


    # SHORT-TERM  (5–15 min)


    def predict_short_term(self, base_wait: float, slope: float,
                           nc: int) -> float:
        """
        Projects queue size 10 minutes into the future using the current
        linear trend, then asks the wait model what that queue means.

        slope (people/sample) × samples/s × 600s = people expected to join
        in 10 min.
        """
        counts = list(historical_counts)
        if not counts:
            return base_wait

        projected_queue = max(0.0, counts[-1] + slope * self._samples_per_second() * 600)
        return wait_model.predict_minutes(projected_queue, nc)


    # MEDIUM-TERM  (15–30 min)


    def predict_medium_term(self, base_wait: float, slope: float,
                            nc: int) -> float:
        """
        Holt's double exponential smoothing (level + trend) over the full
        count history, projected 20 steps ahead with exponential damping.

        Damping prevents the trend from blowing up on a growing queue while
        still capturing direction better than simple smoothing.
        """
        counts = list(historical_counts)
        if len(counts) < 4:
            return base_wait

        alpha, beta = 0.4, 0.2
        level = float(counts[0])
        trend = float(counts[1] - counts[0])

        for c in counts[1:]:
            prev   = level
            level  = alpha * c + (1 - alpha) * (level + trend)
            trend  = beta * (level - prev) + (1 - beta) * trend

        # Damped horizon: sum of damping^1 … damping^horizon
        horizon        = 20
        damping        = 0.85
        damp_sum       = sum(damping ** i for i in range(1, horizon + 1))
        forecast_queue = max(0.0, level + trend * damp_sum)
        return wait_model.predict_minutes(forecast_queue, nc)


    # LONG-TERM  (1–4 hours

    def predict_long_term(self, base_wait: float, slope: float,
                          nc: int) -> float:
        """
        Compares early vs. recent queue averages to detect growth/decline phase.
        Blends the extrapolated trend with mean-reversion (queues stabilise
        over hours). Output is capped at 60 min — long forecasts are uncertain.
        """
        counts = list(historical_counts)
        if len(counts) < 8:
            return base_wait

        third  = max(len(counts) // 3, 1)
        early  = np.mean(counts[:third])
        recent = np.mean(counts[-third:])

        # Growth ratio: > 1 = growing, < 1 = shrinking
        growth_ratio = recent / max(early, 1.0)

        # 70 % follow the trend, 30 % revert to recent mean (ratio = 1.0)
        long_queue = max(0.0, recent * (0.7 * growth_ratio + 0.3))
        return min(wait_model.predict_minutes(long_queue, nc), 60.0)

    # MAIN UPDATE — called once per video frame


    def update(self, count: int, current_data: dict, data_lock) -> None:
        """
        Sample the latest frame count, recalculate all metrics, and
        write results back into current_data (thread-safe via data_lock).

        Call this from gen_frames() after detection is complete.
        """
        now = time.time()
        if not historical_timestamps or now - historical_timestamps[-1] >= SAMPLE_INTERVAL_SECONDS:
            historical_counts.append(count)
            historical_timestamps.append(now)

        arrival_rate = self.calculate_arrival_rate()

        with data_lock:
            sr    = current_data['service_rate']
            nc    = current_data['active_counters']
            ew    = wait_model.predict_minutes(count, nc)
            util  = min(arrival_rate / (nc * sr) if sr > 0 else 0.0, 1.0)
            slope = self._trend_slope()

            current_data.update({
                'queue_length':         count,
                'arrival_rate':         round(arrival_rate, 2),
                'system_utilization':   round(util, 2),
                'estimated_wait_time':  round(ew, 1),
                'predicted_wait_5min':  round(self.predict_short_term(ew, slope, nc), 1),
                'predicted_wait_15min': round(self.predict_medium_term(ew, slope, nc), 1),
                'predicted_wait_30min': round(self.predict_long_term(ew, slope, nc), 1),
            })
