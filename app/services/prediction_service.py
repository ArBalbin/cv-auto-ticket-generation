# app/services/prediction_service.py

import time
import numpy as np
from collections import deque


# ── Shared history (written by update_queue_metrics, read by predictors) ──────
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

    
    # M/M/c BASELINE

    def mmch_wait(self, arrival_rate: float, service_rate: float,
                  num_counters: int, queue: int) -> float:
        """
        M/M/c queuing formula — current-snapshot wait time in minutes.
        This is the ground-truth baseline; the three horizon models
        adjust it based on projected future queue depth.
        """
        avg_svc = self.cfg['avg_service_time']
        nc      = max(num_counters, 1)
        if arrival_rate <= 0:
            return queue * avg_svc / nc
        rho = arrival_rate / max(nc * service_rate, 1e-9)
        if rho >= 1.0:
            return queue * avg_svc / nc
        return max(0.0, (queue / max(arrival_rate, 0.1)) + avg_svc)

    
    # TREND SLOPE (shared utility)
    

    def _trend_slope(self) -> float:
        """
        Linear regression slope over recent queue counts (people/frame).
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

    def _estimated_fps(self) -> float:
        """Estimate camera fps from recent frame timestamps."""
        times = list(historical_timestamps)
        if len(times) < 2:
            return 15.0
        n       = min(30, len(times) - 1)
        elapsed = times[-1] - times[-n - 1]
        return n / elapsed if elapsed > 0 else 15.0

    
    # SHORT-TERM  (5–15 min)
    

    def predict_short_term(self, base_wait: float, slope: float,
                           nc: int) -> float:
        """
        Projects queue size 10 minutes into the future using the current
        linear trend, then recalculates M/M/c wait for that projected size.

        slope (people/frame) × fps × 600s = people expected to join in 10 min.
        """
        counts = list(historical_counts)
        if not counts:
            return base_wait

        fps             = self._estimated_fps()
        projected_queue = max(0.0, counts[-1] + slope * fps * 600)
        avg_svc         = self.cfg['avg_service_time']
        arrival         = self.calculate_arrival_rate()

        if arrival <= 0:
            return max(1.0, projected_queue * avg_svc / max(nc, 1))
        return max(1.0, (projected_queue / max(arrival, 0.1)) + avg_svc)

    
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

        avg_svc = self.cfg['avg_service_time']
        arrival = self.calculate_arrival_rate()
        if arrival <= 0:
            return max(1.0, forecast_queue * avg_svc / max(nc, 1))
        return max(1.0, (forecast_queue / max(arrival, 0.1)) + avg_svc)

    
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

        avg_svc = self.cfg['avg_service_time']
        arrival = self.calculate_arrival_rate()
        if arrival <= 0:
            raw = long_queue * avg_svc / max(nc, 1)
        else:
            raw = (long_queue / max(arrival, 0.1)) + avg_svc

        return max(1.0, min(raw, 60.0))
    
    # MAIN UPDATE — called once per video frame


    def update(self, count: int, current_data: dict, data_lock) -> None:
        """
        Append the latest frame count, recalculate all metrics, and
        write results back into current_data (thread-safe via data_lock).

        Call this from gen_frames() after detection is complete.
        """
        historical_counts.append(count)
        historical_timestamps.append(time.time())

        arrival_rate = self.calculate_arrival_rate()

        with data_lock:
            sr    = current_data['service_rate']
            nc    = current_data['active_counters']
            ew    = self.mmch_wait(arrival_rate, sr, nc, count)
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
