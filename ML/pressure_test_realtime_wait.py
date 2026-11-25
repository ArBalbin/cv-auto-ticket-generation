"""
Pressure test of the real-time wait estimate, on the real backend code.

Run:  .venv/Scripts/python.exe ML/pressure_test_realtime_wait.py   (about a minute)
Results: ML/realtime_wait_results.md

The queue tracker, queue service and queue engine run unchanged. Only two
things are replaced: the database (every write is a no-op, nothing reaches the
live system) and the clock (a simulated one, so an hour of queue runs in a
minute). Students arrive at random and get a number (the staff manual-entry
path, which logs the estimate shown at join exactly as the live system does).
Staff at each window press Call next, serve for a time drawn from the October
2026 cashier log, press Done, and call again a few seconds later. One
transaction is made to run 9 minutes: the panel's case.

Every waiting student's phone asks for its estimate every 5 seconds, as the
app does, and every answer is checked against when that student was really
called. The old engine (deployed until 20 Nov) is asked the same question at
the same moment, from the same line.
"""
import random
import statistics as st
import sys
import threading
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import database.database_handler as dbh  # noqa: E402

dbh._ensure_db_pool = lambda: None           # no database at all

from services import queue_engine, queue_service, queue_tracker as qt  # noqa: E402
from services.queue_tracker import QueueTracker, QueueZone  # noqa: E402

# ------------------------------------------------------------------ simulated clock
BASE = datetime(2026, 11, 22, 9, 0, 0)


class Clock:
    t = 0.0


class SimDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        moment = BASE + timedelta(seconds=Clock.t)
        return moment if tz is None else moment.astimezone(tz)


class SimTime:
    @staticmethod
    def monotonic():
        return 1_000_000.0 + Clock.t

    @staticmethod
    def time():
        return 1_763_000_000.0 + Clock.t

    sleep = staticmethod(lambda s: None)


queue_service.datetime = SimDateTime
qt.datetime = SimDateTime
queue_service.time = SimTime

# ------------------------------------------------------------------ no database
queue_service.update_queue_status = lambda *a, **k: None
queue_service.invalidate_ticket_cache = lambda *a, **k: None
queue_service.counters_closed_until = lambda: None
logged = {}
queue_service.ticket_service.enqueue_ticket = lambda **k: logged.__setitem__(k["queue_number"], k)
threading.Thread.start = lambda self: None   # the DB-write threads would only no-op
import builtins  # noqa: E402

_print = builtins.print

HIST = {1: 191, 2: 238, 3: 56, 4: 12, 5: 11, 6: 2, 8: 1}
LOGGED = [v for v, n in HIST.items() for _ in range(n)]
MEAN = queue_engine.HISTORY_MEAN_MIN


def old_wait(now, counters, ahead, svc=MEAN):
    """The engine deployed until 20 Nov 2026 (queue_engine.py at 0b0791b)."""
    import heapq
    free = []
    for c in counters:
        f = now
        if c.busy_since is not None:
            e = (now - c.busy_since).total_seconds() / 60
            f = now + timedelta(minutes=max(svc - max(0, e), 0.75 * svc))
        free.append(f)
    heapq.heapify(free)
    last = now
    for _ in range(ahead + 1):
        last = heapq.heappop(free)
        heapq.heappush(free, last + timedelta(minutes=svc))
    return max(0.0, (last - now).total_seconds())


_learned_call_delay = queue_service.current_call_delay_minutes


def run(students, windows, util, seed, overrun_index=5, overrun_min=9.0, poll=5, call_gap=5,
        learn_call=True):
    """`call_gap`: seconds staff take to press Call next once they could (after
    Done, or after a student arrives at a free window). `learn_call` False runs
    the engine without the learned call delay, as it was before 25 Nov 2026."""
    rng = random.Random(seed)
    Clock.t = 0.0
    builtins.print = lambda *a, **k: None
    tracker = QueueTracker(zone=QueueZone(0, 0, 640, 480))
    tracker.auto_call = False
    tracker._num_counters = windows
    tracker.on_number_linked = queue_service._on_number_linked
    queue_service.queue_tracker = tracker
    queue_service._window_claims = {}
    queue_service._recent_services = deque(maxlen=50)
    queue_service._recent_call_gaps = deque(maxlen=50)
    queue_service._window_free_since = {}
    queue_service._window_opened_at = {}
    queue_service.current_call_delay_minutes = (
        _learned_call_delay if learn_call else (lambda: 0.0))
    queue_service.runtime_config = lambda: {"avg_service_time": MEAN, "active_counters": windows}
    logged.clear()

    lam = util * windows / (MEAN * 60)          # arrivals per second
    arrivals, t = [], 0.0
    for i in range(students):
        t += rng.expovariate(lam)
        arrivals.append(t)
    number = {i: 5000 + i for i in range(students)}

    joined, called, served = {}, {}, {}
    at_window = {w: None for w in range(1, windows + 1)}   # (queue_number, ends_at)
    freed_at = {w: 0.0 for w in range(1, windows + 1)}     # last Done at the window
    polls = []           # (t, number, new_s, old_s, moving)
    panel = []           # minute-by-minute during the long transaction
    overrun_number = number[overrun_index]
    i_arr = 0
    step = 1.0
    while len(served) < students and Clock.t < 6 * 3600:
        now_t = Clock.t
        while i_arr < students and arrivals[i_arr] <= now_t:
            tracker.force_new_person(number[i_arr])
            joined[number[i_arr]] = now_t
            i_arr += 1
        for w in at_window:                         # staff
            if int(now_t) % 30 == 0:
                queue_service.claim_window(f"staff{w}", w)
            current = at_window[w]
            if current and current[1] <= now_t:
                queue_service.mark_done(current[0])
                served[current[0]] = now_t
                at_window[w] = None
                freed_at[w] = now_t
            waiting = [joined[q] for q in joined if q not in called]
            # Staff call `call_gap` seconds after they could: after Done, or
            # after the first student waiting arrived at a free window.
            if at_window[w] is None and waiting and now_t >= max(freed_at[w], min(waiting)) + call_gap:
                outcome, person = queue_service.call_next(w, f"staff{w}")
                if outcome == "called":
                    q = person["queue_number"]
                    length = (overrun_min if q == overrun_number
                              else max(0.1, rng.choice(LOGGED) - 0.5 + rng.random())) * 60
                    at_window[w] = (q, now_t + length)
                    called[q] = now_t
        if int(now_t) % poll == 0:                   # phones
            state = queue_service.line_state(windows)
            for q in list(joined):
                if q in called:
                    continue
                pos = tracker.get_position(q)
                if pos <= 0:
                    continue
                pred = queue_service.prediction_for_position(pos, windows, state)
                ahead = queue_service._waiting_ahead(state.line, pos)
                o = old_wait(state.now, state.counters, ahead or 0)
                polls.append((now_t, q, pred["estimated_wait_seconds"], o, pred["line_moving"]))
            if overrun_number in called and overrun_number not in served:
                minute = int((now_t - called[overrun_number]) // 60)
                if not panel or panel[-1][0] != minute:
                    behind = [p for p in polls if p[0] == now_t][:2]
                    panel.append((minute, behind))
        Clock.t += step
    builtins.print = _print

    # ---------------------------------------------------------------- scoring
    def r2(pred, act):
        m = st.mean(act)
        tot = sum((a - m) ** 2 for a in act)
        return 1 - sum((a - p) ** 2 for a, p in zip(act, pred)) / tot if tot else float("nan")

    actual = [called[q] - t for t, q, *_ in polls]
    new = [p[2] for p in polls]
    old = [p[3] for p in polls]
    join_act = [called[q] - joined[q] for q in joined]
    join_new = [logged[q]["est_wait_min"] * 60 for q in joined]

    # Frozen: a waiting student's estimate identical for 60 s or more while
    # someone is at a window (the line is moving), as the panel saw.
    def frozen(index):
        by_q, worst = {}, 0
        for p in polls:
            by_q.setdefault(p[1], []).append(p)
        count = 0
        for q, ps in by_q.items():
            run_len = 0
            for a, b in zip(ps, ps[1:]):
                same = round(a[index]) == round(b[index]) and b[4] and round(b[index]) > 0
                run_len = run_len + poll if same else 0
                if run_len == 60:
                    count += 1
        return count

    waits = [called[q] - joined[q] for q in joined]
    return {
        "students": students, "windows": windows,
        "served": len(served), "minutes": Clock.t / 60,
        "mean_wait": st.mean(waits) / 60, "max_wait": max(waits) / 60,
        "max_line": max((sum(1 for q in joined if joined[q] <= t and called.get(q, 1e9) > t)
                         for t in range(0, int(Clock.t), 15)), default=0),
        "polls": len(polls),
        "rt_new": (r2(new, actual), st.mean(abs(a - b) for a, b in zip(new, actual))),
        "rt_old": (r2(old, actual), st.mean(abs(a - b) for a, b in zip(old, actual))),
        "join_new": (r2(join_new, join_act), st.mean(abs(a - b) for a, b in zip(join_new, join_act))),
        "frozen_new": frozen(2), "frozen_old": frozen(3),
        "panel": panel, "overrun_number": overrun_number,
        "rows": list(zip(actual, new, old)),
        "join_rows": list(zip(join_act, join_new)),
    }


if __name__ == "__main__":
    scenarios = [(24, 1, 0.85), (30, 2, 0.90), (40, 3, 0.95)]
    pooled = {"rt_new": [], "rt_old": []}
    print("| Students | Windows | Busy | Mean wait | Longest wait | Longest line | Phone checks "
          "| Live R² new | Live MAE new | Live R² old | Live MAE old | At-join R² new | At-join MAE new "
          "| Frozen 60 s+ new | Frozen 60 s+ old |")
    print("|---" * 15 + "|")
    panel_example = None
    for students, windows, util in scenarios:
        for seed in range(3):
            r = run(students, windows, util, seed)
            print(f"| {students} | {windows} | {int(util*100)}% | {r['mean_wait']:.1f} min | {r['max_wait']:.1f} min "
                  f"| {r['max_line']} | {r['polls']} | {r['rt_new'][0]:.3f} | {r['rt_new'][1]:.0f} s "
                  f"| {r['rt_old'][0]:.3f} | {r['rt_old'][1]:.0f} s | {r['join_new'][0]:.3f} "
                  f"| {r['join_new'][1]:.0f} s | {r['frozen_new']} | {r['frozen_old']} |", flush=True)
            if windows == 1 and seed == 0:
                panel_example = r
            pooled["rt_new"] += r["rows"]
            pooled.setdefault("join", []).extend(r["join_rows"])
    rows = pooled["rt_new"]
    act = [a for a, _, _ in rows]

    def r2(pred, act):
        m = st.mean(act)
        return 1 - sum((a - p) ** 2 for a, p in zip(act, pred)) / sum((a - m) ** 2 for a in act)

    newp = [n for _, n, _ in rows]
    oldp = [o for _, _, o in rows]
    mae = lambda pred, act: st.mean(abs(a - b) for a, b in zip(act, pred))
    print(f"\nPOOLED, 9 runs, {len(rows)} phone checks: live R2 new {r2(newp, act):.3f} "
          f"(MAE {mae(newp, act):.0f} s), old {r2(oldp, act):.3f} (MAE {mae(oldp, act):.0f} s)")
    ja = [a for a, _ in pooled["join"]]
    jn = [n for _, n in pooled["join"]]
    print(f"At join, {len(ja)} students: R2 new {r2(jn, ja):.3f} (MAE {mae(jn, ja):.0f} s)")
    print("\nPANEL CASE (24 students, 1 window, seed 0): the 9-minute transaction of "
          f"Q{panel_example['overrun_number']}. Waits of the next two in line, min:sec.")
    print("| Minutes at window | Next in line: new | old | Second in line: new | old |")
    print("|---|---|---|---|---|")
    f = lambda s: "{}:{:02d}".format(*divmod(int(round(s)), 60))
    for minute, behind in panel_example["panel"]:
        if len(behind) >= 2:
            a, b = behind[0], behind[1]
            print(f"| {minute} | {f(a[2])} | {f(a[3])} | {f(b[2])} | {f(b[3])} |")

    # Staff slow to call, as in the test sessions of 20-24 Nov 2026 (about
    # 71 s): the same runs with and without the learned call delay.
    print("\nSTAFF TAKE 60 S TO CALL (same 9 runs):")
    print("| Engine | Live R² | Live MAE | At-join R² | At-join MAE | Students |")
    print("|---|---|---|---|---|---|")
    for learn, label in ((False, "Without the call delay (until 25 Nov)"), (True, "Learning the call delay")):
        live, join = [], []
        for students, windows, util in scenarios:
            for seed in range(3):
                r = run(students, windows, util, seed, call_gap=60, learn_call=learn)
                live += [(a, n) for a, n, _ in r["rows"]]
                join += r["join_rows"]
        la, lp = [a for a, _ in live], [p for _, p in live]
        ja, jp = [a for a, _ in join], [p for _, p in join]
        print(f"| {label} | {r2(lp, la):.3f} | {mae(lp, la):.0f} s | {r2(jp, ja):.3f} "
              f"| {mae(jp, ja):.0f} s | {len(ja)} |", flush=True)
