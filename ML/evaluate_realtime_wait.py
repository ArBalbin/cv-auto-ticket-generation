"""
Old engine (deployed until 20 Nov 2026) vs the real-time engine, on replayed lines.

Run:  .venv/Scripts/python.exe ML/evaluate_realtime_wait.py   (several minutes)
Results: ML/realtime_wait_results.md

Students arrive at random and are served first come first served at whichever
window frees first. Service lengths are drawn from the October 2026 cashier
log: each logged whole minute v is drawn as a time between v - 0.5 and v + 0.5
minutes, because real transactions are not whole minutes.

Each engine is asked what the live system knows at that moment: who is at
each window and since when, how many are still waiting ahead, and (new engine
only) how long the services finished in the past two hours took.

Two accuracy measures:
  at join     the estimate given when the student joins vs the wait they had
  real time   the estimate shown every 30 s while they wait vs the wait they
              still had from that moment (what the app shows, live)

Scenarios: 1-3 windows x 60/85/95 % busy on a normal day, then a slow day
(every transaction 40 % longer than October) and a fast day (25 % shorter).
The old engine is scored twice: with the October average it starts from, and
given each day's true average (generous: the deployed one re-measured it from
gaps between Done presses every five minutes, when it had five of them).
"""
import bisect
import heapq
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from services import queue_engine as qe  # noqa: E402

HIST = {1: 191, 2: 238, 3: 56, 4: 12, 5: 11, 6: 2, 8: 1}
LOGGED = [v for v, n in HIST.items() for _ in range(n)]
MEAN = qe.HISTORY_MEAN_MIN


def draw(rng, scale):
    v = rng.choice(LOGGED)
    return max(0.05, (v - 0.5 + rng.random())) * scale


# ---- the engine deployed until 20 Nov 2026 (queue_engine.py at 0b0791b)
def old_wait(t, occupied_since, ahead, svc):
    free = []
    for since in occupied_since:
        if since is None:
            free.append(t)
        else:
            e = t - since
            free.append(t + max(svc - max(0.0, e), 0.75 * svc))
    heapq.heapify(free)
    last = t
    for _ in range(ahead + 1):
        last = heapq.heappop(free)
        heapq.heappush(free, last + svc)
    return max(0.0, last - t)


# ---- the real-time engine (app/services/queue_engine.py now)
def new_wait(t, occupied_since, ahead, recent, prior):
    at_window = [t - s for s in occupied_since if s is not None]
    model = qe.service_model(recent, at_window, prior_min=prior)
    free = []
    for since in occupied_since:
        free.append(t if since is None else t + model.remaining(t - since))
    heapq.heapify(free)
    last = t
    for _ in range(ahead + 1):
        last = heapq.heappop(free)
        heapq.heappush(free, last + model.mean_min)
    return max(0.0, last - t)


def simulate(windows, util, scale=1.0, n=2500, seed=0, poll=0.5):
    rng = random.Random(seed)
    lam = util * windows / (MEAN * scale)
    t, arrivals = 0.0, []
    for _ in range(n):
        t += rng.expovariate(lam)
        arrivals.append((t, draw(rng, scale)))
    free = [(0.0, w) for w in range(windows)]
    heapq.heapify(free)
    starts, ends, win = [], [], []
    for arr, svc in arrivals:
        f, w = heapq.heappop(free)
        s = max(arr, f)
        starts.append(s), ends.append(s + svc), win.append(w)
        heapq.heappush(free, (s + svc, w))

    by_window = {w: [] for w in range(windows)}
    for i in range(n):
        by_window[win[i]].append(i)
    win_starts = {w: [starts[i] for i in idx] for w, idx in by_window.items()}
    order_end = sorted(range(n), key=lambda i: ends[i])
    end_sorted = [ends[i] for i in order_end]

    def state(t, i):
        occ = []
        for w in range(windows):
            k = bisect.bisect_right(win_starts[w], t) - 1
            j = by_window[w][k] if k >= 0 else None
            occ.append(starts[j] if j is not None and ends[j] > t else None)
        first_waiting = bisect.bisect_right(starts, t)
        ahead = max(0, i - first_waiting)
        hi = bisect.bisect_right(end_sorted, t)
        lo = bisect.bisect_left(end_sorted, t - qe.RECENT_WINDOW_MIN)
        done = order_end[max(lo, hi - qe.RECENT_SERVICES):hi]
        recent = [ends[j] - starts[j] for j in done]
        return occ, ahead, recent

    join, live = [], []
    for i in range(200, n):        # skip the empty line at the start
        arr = arrivals[i][0]
        actual = starts[i] - arr
        occ, ahead, recent = state(arr, i)
        join.append((old_wait(arr, occ, ahead, MEAN), old_wait(arr, occ, ahead, MEAN * scale),
                     new_wait(arr, occ, ahead, recent, MEAN), actual))
        k = 0
        while arr + k * poll < starts[i]:
            tq = arr + k * poll
            occ, ahead, recent = state(tq, i)
            live.append((old_wait(tq, occ, ahead, MEAN), old_wait(tq, occ, ahead, MEAN * scale),
                         new_wait(tq, occ, ahead, recent, MEAN), starts[i] - tq))
            k += 1
    return join, live


def r2(pred, act):
    m = st.mean(act)
    tot = sum((a - m) ** 2 for a in act)
    return 1 - sum((a - p) ** 2 for a, p in zip(act, pred)) / tot


def line(label, rows):
    act = [r[3] for r in rows]
    mae = lambda p: st.mean(abs(a - b) for a, b in zip(p, act)) * 60
    cells = []
    for k in range(3):
        pred = [r[k] for r in rows]
        cells.append(f"{r2(pred, act):.3f} | {mae(pred):.0f} s")
    return f"| {label} | {st.mean(act):.1f} | " + " | ".join(cells) + f" | {len(rows)} |"


if __name__ == "__main__":
    seeds = range(3)
    head = ("| Scenario | Mean wait (min) | Old R² | Old MAE | Old, true avg R² | Old, true avg MAE "
            "| New R² | New MAE | Estimates |\n|---|---|---|---|---|---|---|---|---|")
    pooled_join, pooled_live = [], []
    blocks = {"join": [], "live": []}
    for scale, day in ((1.0, "normal day"), (1.4, "slow day"), (0.75, "fast day")):
        for w in (1, 2, 3):
            for u in (0.6, 0.85, 0.95):
                if scale != 1.0 and u != 0.85:
                    continue
                J, L = [], []
                for s in seeds:
                    j, l = simulate(w, u, scale, seed=s)
                    J += j
                    L += l
                label = f"{w} window{'s' if w > 1 else ''}, {int(u * 100)}% busy, {day}"
                blocks["join"].append(line(label, J))
                blocks["live"].append(line(label, L))
                pooled_join += J
                pooled_live += L
                print(label, "done", flush=True)
    blocks["join"].append(line("**All pooled**", pooled_join))
    blocks["live"].append(line("**All pooled**", pooled_live))
    print("\nAT JOIN (estimate given on joining vs the wait they had)")
    print(head)
    print("\n".join(blocks["join"]))
    print("\nREAL TIME (estimate shown every 30 s while waiting vs the wait still ahead)")
    print(head)
    print("\n".join(blocks["live"]))
