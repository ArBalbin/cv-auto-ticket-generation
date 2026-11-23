# The Wait-Time Model: Real-Time Queue Engine

This document describes how QueuEx estimates each student's wait and time of
service (Algorithm 6): how the engine works, the formulas it uses, the constants
behind them, and how accurate it is.

**Date:** 2026-11 (engine rewritten 20 Nov 2026, after the final defense)
**Engine:** `app/services/queue_engine.py` (formulas), `app/services/queue_service.py` (`line_state()`, `prediction_for_position()`)
**Version logged with each ticket:** `queue-engine-v2`
**Accuracy scripts:** `ML/evaluate_realtime_wait.py` (replayed lines), `ML/pressure_test_realtime_wait.py` (pressure test on the real backend code)

---

## 1. What it was built to do

At the final defense on 20 Nov 2026 the panel asked for:

- a **real-time** estimate that changes with the situation in the line, for
  example when the student at a window takes longer than usual;
- the wait in **minutes and seconds**;
- an **exact estimated time of service** (a clock time);
- the **average service time** on the student's screen.

Their test case: Q1 is called to window 1 and takes long to be marked done. Q2's
and Q3's estimates must move later while it does, not stay where they were.

## 2. What happens on every request

The app asks every few seconds (every 4 s on the waiting screen). Each request
reads the line once and works out every estimate in the response from that one
reading (`line_state()`):

1. Read the line from the tracker: order, who is at which window, and since when.
2. Work out the **current average service time** μ (formula 2).
3. For each window, work out **when it frees up** (formula 1).
4. **Play the line forward** to find when each person reaches a window (formula 3).
5. Send each student their wait in seconds, the clock time of their turn, μ, and
   what the estimate rests on. The app counts the seconds down until the next
   answer replaces them.

Nothing is added or taken away in fixed steps. Every figure moves with the line.

## 3. Inputs

| Input | Where it comes from |
|---|---|
| Line order, who is at which window | The live tracker (`line_snapshot()`). Only people the tracker holds count; a ticket left "waiting" in the database by a restart is not in the line, since nobody can call it. |
| Time at the window, *e* | Now − `called_at`, set when the window is assigned |
| Windows counted | Automatic calling: every open window. Call next: the windows a staff member holds plus any where someone is being served (`calling_windows()`). If none, one is assumed and the app says no window is calling. |
| Services just finished | Every Done is timed from Call next to Done and kept in memory (`record_service_time()`); after a restart the last ones are read from the database (`fetch_recent_service_seconds()`). |
| Long-run average μ₀ | `AVG_SERVICE_TIME`, default 1.87 min, the mean of the October 2026 cashier log |
| Lunch closure | "Close all counters until" on the dashboard |

## 4. The formulas

To tell a student when their turn will come, the engine has to answer three
questions, one formula each:

| Formula | Question it answers | Takes | Gives |
|---|---|---|---|
| 1. Time left at a busy window | How much longer will the student now at each window take? | How long they have been there (*e*), the average service time (μ) | Minutes until that window is free |
| 2. Current average service time | How long does one student take at the window, at today's pace? | The long-run average, the services just finished, any still running long | μ, in minutes |
| 3. Playing the line forward | Given those, when does each person in line reach a window? | When each window is free (formula 1), μ (formula 2), the order of the line | Each student's wait in seconds and their time of service |

Formula 2 feeds formulas 1 and 3, and formula 1 feeds formula 3. All three are
worked out again on every request, so the answer changes as soon as the line
does.

### Formula 1: time left at a busy window (mean residual life)

**What it is for.** It tells the engine when each busy window will be free. A
student who has been at a window for *e* minutes has some time still to go, and
that is how long everyone behind them must wait for that window. This is the
formula that makes the estimate real-time: *e* grows every second, so the answer
is different on every request. Simply taking "average minus time so far" would
reach zero after an average-length service and say the window is about to free
up even when the student is clearly taking long. This formula instead uses how
real cashier transactions behave: one that has already run long usually still
has time to go.

A transaction that has lasted *e* minutes is expected to need

```
R(e) = μ · m(e / μ)

m(x) = Φ( (σ²/2 − ln x) / σ )  /  Φ( (−σ²/2 − ln x) / σ )  −  x        (m(0) = 1)
```

more minutes, where Φ is the standard normal distribution function and
σ = 0.425. `m(x)` is E[X − x | X > x] for a lognormal X with mean 1: the shape of
a cashier transaction's length, in units of its own average. It was fitted by
maximum likelihood to the 511 timed transactions of the October 2026 cashier
log, which recorded whole minutes (191 took 1, 238 took 2, 56 took 3, 12 took 4,
11 took 5, 2 took 6, 1 took 8), each read as an interval v − 0.5 to v + 0.5. The
fit has mean 1.87 min and median 1.71 min. Service times are commonly found to
be lognormal (Brown et al., 2005).

| Time at the window | Expected time left, fitted (μ = 1.87) | October log |
|---:|---:|---:|
| 0:00 | 1.87 min | 1.87 |
| 1:00 | 0.99 | 1.13 |
| 2:00 | 0.74 | 0.78 |
| 3:00 | 0.72 | 0.96 |
| 4:00 | 0.74 | 1.02 |
| 10:00 | 1.02 | (no data) |
| 20:00 | 1.49 | (no data) |

The time left falls while a transaction runs normally and rises again once it
runs long: a transaction that has gone on unusually long is an unusual one.
`e + R(e)`, the expected total length given it has lasted *e*, never decreases,
so a student's time of service can only move later while the person ahead of
them is still at the window.

| Symbol | Meaning |
|---|---|
| *e* | How long the student now at the window has been there, in minutes |
| μ | The current average service time (formula 2) |
| *x* = *e*/μ | The time so far as a multiple of the average (x = 1 means one average service so far) |
| σ = 0.425 | How spread out cashier transaction times are, from the October log |
| ln | Natural logarithm |
| Φ | Standard normal distribution function (the area under the bell curve up to a value) |
| m(*x*) | The time left as a multiple of the average |
| R(*e*) | The time left, in minutes |

**Example.** With μ = 1.87 min, a student who has been at the window for 2
minutes is expected to need R(2) = 0.74 min more, so that window is free in
about 45 seconds.

### Formula 2: the current average service time

**What it is for.** It is how long one student takes at the window at today's
pace. Formula 3 gives each person still waiting this much time at the window,
formula 1 uses it as the scale for the time left, and students see it as
"Average service time". The cashier's pace is not fixed: some days, and some
hours, transactions take longer. So instead of a fixed number the engine starts
from the long-run average and moves toward the times it has just measured. A
student taking unusually long right now also pushes it up at once, so that
everyone behind them is warned before that service ends.

```
μ = ( k·μ₀ + ΣSᵢ + Σeⱼ ) / ( k + n + L )
```

| Symbol | Meaning |
|---|---|
| μ₀ | The long-run average, 1.87 min |
| k = 10 | How many services μ₀ counts as |
| Sᵢ | The last n services finished in the past two hours, n ≤ 20; shorter than 5 s (a mis-click) or longer than 30 min (a forgotten Done) are ignored |
| eⱼ | Students still at a window who have already taken longer than the average of the above, counted at their time so far (they will take at least that long); L is how many |

With nothing timed, μ = μ₀. As services are timed it follows the cashier's real
pace that day. A window running long raises μ at once, so everyone behind it is
told so.

The long-run average counts as 10 services so that a single fast or slow
transaction early in the day does not swing every estimate; after 20 services
it carries about a third of the weight.

**Example.** Four services of 2:00 were just finished and no one is running
long: μ = (10 × 1.87 + 4 × 2.0) / (10 + 4) = 1.907 min, shown as 1:54.

### Formula 3: playing the line forward

**What it is for.** It turns the window times into each student's wait. This is
the queue itself, worked out step by step the way the cashier line actually
moves: first come, first served, each person going to whichever window frees up
first. Each student gets their own answer from their place in line, the number
of windows calling, and how long the people already at the windows have been
there. It also gives the clock time of their turn (the estimated time of
service) and handles a lunch closure.

Each counted window *w* is free at

```
F_w = now                       if no one is at it
F_w = now + R(e_w)              if someone has been at it e_w minutes
F_w = max(F_w, closed_until)    during a lunch closure
```

The people still waiting are taken in queue order. Each goes to the window that
frees first, which is then busy for μ more minutes. The moment a person is taken
is their start:

```
wait                        = start − now
estimated time of service   = now + wait
estimated finish            = start + μ
```

With one window and *p* people waiting ahead (not counting the one at the
window):

```
wait = R(e) + p · μ
```

With *c* windows, the line is shared out to whichever window frees first, so
each window takes about every *c*-th person.

| Symbol | Meaning |
|---|---|
| F_w | The moment window *w* can take the next person |
| *e_w* | How long the student at window *w* has been there |
| closed_until | The time the windows reopen after a lunch closure, if any |
| start | The moment a person is taken by a window |
| *p* | People still waiting ahead (not counting those already at a window) |
| *c* | Windows calling |

**Example.** One window; Q1 has been at it 5 minutes and μ = 2.113 min. The
window is free in R(5) = 0.85 min, so Q2 (next) waits 0:51. Q3 goes after Q2:
0.85 + 2.113 = 2.97 min, so Q3 waits 2:58 (section 5).

### Is the engine trained?

Not the way a machine-learning model is. It is not trained on past waits and
does not learn a mapping from inputs to a wait. It works out each wait by
playing the line forward from what the system knows at that moment, so it needs
no training set of waits, and its accuracy is checked by validation on lines it
has never seen (section 8). Two of its values do come from real data:

| Part | Where its value comes from | Kind |
|---|---|---|
| σ = 0.425, the shape of service times | Fitted once by maximum likelihood to 511 real transactions in the October 2026 cashier log | Estimated from data (a statistical fit) |
| μ, the average service time | Updated live from every service staff finish, starting from the October mean | Learned continuously from the cashier's own data |
| k = 10 and the last 20 services | Chosen by trying settings on replayed lines (section 8.4) | Tuned |
| First come first served, windows, lunch closure | How the cashier line works | Not learned: a model of the process |

In short, it is a queue model calibrated on real cashier data: the data sets
how long transactions take, and the queue rules turn that into each student's
wait.

## 5. Worked example: the panel's case

One window. Four services of 2:00 were timed earlier. Q1 has been at the window
for 5 minutes; Q2 is next, Q3 after Q2.

```
μ        = (10 × 1.87 + 4 × 2.0 + 5) / (10 + 4 + 1)   = 2.113 min
m(5/μ)   = m(2.366) = 0.0348 / 0.0126 − 2.366        ≈ 0.40
Q2 wait  = R(5) = 2.113 × 0.40                       ≈ 0.85 min = 0:51
Q3 wait  = R(5) + 1 × μ = 0.85 + 2.113              ≈ 2.97 min = 2:58
```

The same case, minute by minute:

| Q1 at the window | Q2 wait | Q3 wait | Average shown |
|---|---|---|---|
| 0:00 | 1:54 | 3:49 | 1:54 |
| 0:30 | 1:25 | 3:19 | 1:54 |
| 1:00 | 1:01 | 2:56 | 1:54 |
| 1:30 | 0:50 | 2:45 | 1:54 |
| 2:00 | 0:46 | 2:41 | 1:55 |
| 3:00 | 0:46 | 2:44 | 1:59 |
| 5:00 | 0:51 | 2:58 | 2:07 |
| 8:00 | 1:03 | 3:22 | 2:19 |
| 10:00 | 1:11 | 3:38 | 2:27 |
| 15:00 | 1:31 | 4:18 | 2:47 |
| 20:00 | 1:52 | 4:58 | 3:07 |

The waits count down while Q1 takes a usual time, then grow for as long as Q1
stays, and Q2's and Q3's times of service move later the whole time.

## 6. How the mobile app shows the prediction

The engine runs on the server; the app (`queueflow_mobile`) asks for its answer,
shows it, and keeps its clocks running between answers.

### 6.1 What the app asks for

Each refresh makes two requests (`QueueApi.fetchQueueSnapshot()`):

| Request | Gives the app |
|---|---|
| `GET /api/queue/status?q=<number>&token=<code>` | The student's own ticket: position, window when called, how long ago they were called, waiting time so far, and their prediction: wait in seconds, time of service, average service time, people ahead, windows calling, whether the line is moving, and the note when a window runs long |
| `GET /api/queue/prediction` | The whole line: how many are in line, who is at each window and for how long, the average service time, and the wait for someone getting a number now |

How often each screen asks:

| Screen | Asks every |
|---|---|
| Home (dashboard) | 5 s, or 2 s right after asking for a number |
| My Queue Status (waiting) | 4 s |
| You're next (first in line) | 5 s |
| It's your turn (called to a window) | 4 s |

Each answer is worked out on the server at that moment (section 2), so every
refresh brings a new estimate, not the old one again.

### 6.2 How the clocks run between answers

The server sends the wait as a number of **seconds**; the app never compares a
server time with its own clock. The app remembers the moment the answer arrived on
the phone and counts from there (`LiveCountdown`, `LiveCountUp` in
`lib/widgets/live_wait.dart`):

```
shown wait      = seconds sent − seconds since the answer arrived     (never below 0:00)
shown count-up  = seconds sent + seconds since the answer arrived
```

So a phone whose clock is a few minutes off still shows the right wait. The
display changes once a second, and each new answer (every 4 to 5 s) replaces
the figure it counts from. When the person at a window runs long, the new
answer is larger than what the clock had reached, and the clock moves back up:
that is the estimate moving later, as the panel asked.

- **Always a clock.** The wait is always shown as minutes and seconds, such as
  3:05, down to 0:00; never as words.
- **Live or Paused.** While someone is at a window (`line_moving` true) the clock
  counts down and the screen shows **Live**. While nobody is at a window the line
  is not moving, so the clock holds the server's figure still and the screen
  shows **Paused**; it starts counting as soon as a window calls the next number.
- **Time of service** is shown as a clock time from the server, on the school's
  clock (for example "2:41:30 PM"), so it is the same on every phone.

### 6.3 Screen by screen

**Home (dashboard).** Under "Queue right now", with a Live or Paused tag beside it:

| Shows | Meaning |
|---|---|
| In line | People in the line now |
| Est. wait (min:sec) | The wait for someone getting a number now, as a running clock |
| Windows calling | Windows taking people now |
| Now serving | Each window's number and how long that student has been at it, counting up; marked "taking longer than usual" past the average |
| Average service time | μ, as min:sec |

**My Queue Status (waiting in line).**

| Shows | Meaning |
|---|---|
| Your queue number, Waiting in line | The ticket |
| Ahead of you | People in line before this student |
| Est. wait (min:sec) | This student's wait, as a running clock |
| Windows calling | Windows taking people now |
| Your turn at about 2:41:30 PM, Live / Paused | The estimated time of service |
| Waiting so far | Time since the number was issued, counting up |
| Average service time | μ, as min:sec |
| Amber note | Only when a window runs long, e.g. "Window 1 has been serving Q001 for 5 min 12 sec, longer than the average of 2 min 07 sec, so your time moved later." |
| Now serving | As on the home screen |
| Queue ahead of you | Each person ahead and their own estimated wait |

**You're next (first in line).** The same three figures (ahead, est. wait as a
running clock, windows calling), the time-of-service card, and Now serving,
with "Stay near the queue area. This screen shows your window when you are
called."

**It's your turn (called to a window).** The phone vibrates, and the screen shows
the window number, the queue number and the payment to have ready. A clock
counts down the time to reach the window from the moment the number was called
(`arrive_within_seconds`, 60 s by default, set with `ARRIVE_WITHIN_SECONDS`): green,
amber in the last 15 s, then red "Late: go now" counting the time over. A
full-width **I'm at the window** button turns it into a clock of the time at the
window. Skipping a number that does not come stays a staff decision; the app
never skips anyone.

**Moving between screens.** After every answer the app picks the screen from
the ticket's status (`routeForQueueSnapshot()`): called to a window → It's your
turn; first in line → You're next; otherwise → My Queue Status; done → finished;
skipped by staff → skipped.

### 6.4 Alerts

| Alert | When | How |
|---|---|---|
| The line is long right now | Joining when the wait for a new number is 15 min or more | A question before joining ("Get a number anyway?") and a banner on the home screen |
| This will be a long wait | The student's own estimate reaches 15 min or more, once per ticket | Vibration, a message, and a banner on the waiting screen |
| Your turn is near | The estimate drops to 5 min or less with people still ahead, once per ticket | Phone notification, strong vibration and a message, e.g. "Q005 should be called in about 4 min (2 people ahead of you). Head back to the cashier queue area now." |
| You're next | The student becomes first in line | Phone notification |
| It's your turn | The number is called to a window | Strong vibration and the It's your turn screen |

### 6.5 What a student sees in the panel's case

One window, as in section 5. Q3 joins just as Q1 is called; Q2 is next.

1. **On joining**, My Queue Status shows Ahead of you 2, Est. wait 3:49 counting
   down, Live, "Your turn at about" a clock time 3 min 49 s away, and the average
   service time 1:54.
2. **While Q1 takes a usual time**, the clock counts down second by second. For
   the first half minute each new answer matches it; after that each answer is a
   few seconds above where the clock had reached, since a transaction already a
   minute in is expected to need a little more than the average minus a minute
   (formula 1). At 1:00, Q3 is told 2:56 rather than 2:49.
3. **When Q1 runs past the average**, each new answer is a little larger than
   where the clock had reached: the clock moves back up, the time of service moves
   later, the average service time rises, and the amber note says Window 1 is
   taking longer than usual. Q1 at 10 minutes: Q3 is told 3:38.
4. **When Q1 is done and Q2 is called**, Q3 becomes first in line: You're next
   appears with a notification, and the clock counts down Q2's expected time.
5. **When Q3 is called**, the phone vibrates and It's your turn shows the window
   and the time to reach it.

## 7. Constants

| Constant | Value | Source |
|---|---|---|
| σ (shape) | 0.425 | Lognormal fitted by maximum likelihood to the October 2026 log |
| μ₀ | 1.87 min | Mean of the October 2026 log (`AVG_SERVICE_TIME`) |
| k | 10 | Chosen on replayed lines (section 8.4) |
| n | last 20 services | Chosen on replayed lines (section 8.4) |
| Look-back | 2 hours | A service from another day says little about today's pace |
| Timed service kept | 5 s to 30 min | Shorter is a mis-click, longer a forgotten Done |
| Recalculation | every request (every 4 s on the waiting screen) | |

## 8. Accuracy

R² is the share of the variation in real waits that the estimate explains: 1.0
is perfect, 0 is no better than always predicting the average wait. MAE is the
typical miss.

All figures below come from **simulation** that uses the real service times of
the October 2026 cashier log. They are not a measurement at the cashier: the
October log does not record who was at each window and since when, so the engine
cannot be scored on it directly. Live accuracy is measured as described in
section 9.

### 8.1 Summary

| Test | R² | MAE |
|---|---:|---:|
| Pressure test on the real backend code, estimate shown every 5 s while waiting (9 runs, 282 students, 16,241 checks) | **0.834** | **53 s** |
| Same runs, estimate given when the student joined | **0.818** | **67 s** |
| Replayed lines, estimate given when the student joined (15 scenarios, 103,500 students) | **0.967** | **81 s** |
| Same replays, estimate shown every 30 s while waiting (1,509,230 checks) | **0.961** | **123 s** |

Pooled R² mixes scenarios whose waits differ tenfold, which raises it; the
per-scenario figures follow.

### 8.2 Replayed lines

Students arrive at random and are served first come first served at 1 to 3
windows, with service lengths drawn from the October log (each logged whole
minute v drawn between v − 0.5 and v + 0.5). Normal days are at the October pace,
slow days 40% slower, fast days 25% faster. Arrivals are sized so each day is as
busy as labelled. 2,500 students per run, three runs per scenario, the first 200
of each run skipped while the line fills (`ML/evaluate_realtime_wait.py`).

| Scenario | Mean wait | R² at join | MAE at join | R² while waiting | MAE while waiting |
|---|---:|---:|---:|---:|---:|
| 1 window, 60% busy, normal day | 2.0 min | 0.684 | 80 s | 0.766 | 60 s |
| 1 window, 85% busy, normal day | 8.9 min | 0.933 | 110 s | 0.912 | 111 s |
| 1 window, 95% busy, normal day | 29.5 min | 0.965 | 210 s | 0.954 | 200 s |
| 2 windows, 60% busy, normal day | 0.8 min | 0.599 | 38 s | 0.685 | 38 s |
| 2 windows, 85% busy, normal day | 4.2 min | 0.927 | 58 s | 0.898 | 61 s |
| 2 windows, 95% busy, normal day | 14.5 min | 0.964 | 107 s | 0.951 | 104 s |
| 3 windows, 60% busy, normal day | 0.4 min | 0.483 | 25 s | 0.586 | 31 s |
| 3 windows, 85% busy, normal day | 2.7 min | 0.916 | 41 s | 0.882 | 44 s |
| 3 windows, 95% busy, normal day | 9.5 min | 0.963 | 73 s | 0.948 | 72 s |
| 1 window, 85% busy, slow day | 12.5 min | 0.925 | 158 s | 0.904 | 158 s |
| 2 windows, 85% busy, slow day | 5.9 min | 0.925 | 79 s | 0.897 | 84 s |
| 3 windows, 85% busy, slow day | 3.8 min | 0.919 | 55 s | 0.888 | 60 s |
| 1 window, 85% busy, fast day | 6.7 min | 0.903 | 97 s | 0.869 | 99 s |
| 2 windows, 85% busy, fast day | 3.2 min | 0.888 | 53 s | 0.844 | 56 s |
| 3 windows, 85% busy, fast day | 2.0 min | 0.871 | 37 s | 0.817 | 41 s |
| **All pooled** | 7.1 min | **0.967** | **81 s** | **0.961** | **123 s** |

"Mean wait" is at join; the while-waiting columns score the estimate shown every
30 s against the wait still ahead at that moment. R² is lowest when lines are
short (0.4 to 2 minutes), where waits barely vary and a few seconds' miss is a
large share of that variation; the MAE there is 25 to 80 s.

### 8.3 Pressure test on the real backend code

The tracker, queue service and engine run unchanged with the database switched
off and a simulated clock (`ML/pressure_test_realtime_wait.py`). Students arrive
at random and get a number; staff at each window press Call next, serve for a
time drawn from the October log, press Done, and call again five seconds later.
One transaction per run takes 9 minutes. Every waiting student's phone asks for
its estimate every 5 seconds, and each answer is checked against when that
student was really called. Three runs per setting.

| Students | Windows | Busy | Mean wait | Longest wait | Longest line | Checks | R² while waiting | MAE | R² at join | MAE at join | Unchanged 60 s+ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 1 | 85% | 5.1 min | 11.6 min | 5 | 1,478 | 0.704 | 53 s | 0.651 | 85 s | 1 |
| 24 | 1 | 85% | 8.8 min | 15.0 min | 7 | 2,545 | 0.848 | 57 s | 0.751 | 99 s | 0 |
| 24 | 1 | 85% | 6.2 min | 13.1 min | 6 | 1,800 | 0.864 | 49 s | 0.775 | 72 s | 2 |
| 30 | 2 | 90% | 2.8 min | 7.2 min | 7 | 1,013 | 0.680 | 49 s | 0.713 | 58 s | 1 |
| 30 | 2 | 90% | 5.8 min | 9.8 min | 11 | 2,091 | 0.715 | 59 s | 0.681 | 78 s | 0 |
| 30 | 2 | 90% | 4.7 min | 9.2 min | 9 | 1,678 | 0.719 | 54 s | 0.677 | 77 s | 2 |
| 40 | 3 | 95% | 0.9 min | 4.2 min | 5 | 422 | 0.367 | 41 s | 0.701 | 25 s | 0 |
| 40 | 3 | 95% | 7.1 min | 14.6 min | 18 | 3,398 | 0.852 | 59 s | 0.763 | 97 s | 0 |
| 40 | 3 | 95% | 3.8 min | 8.7 min | 12 | 1,816 | 0.826 | 37 s | 0.848 | 39 s | 0 |
| **Pooled** | | | | | | **16,241** | **0.834** | **53 s** | **0.818** | **67 s** | |

"Unchanged 60 s+" counts each time a waiting student's estimate stayed identical
for a minute or more while someone was at a window. During the 9-minute
transaction in the first run, the next two in line were told:

| Minutes at the window | Next in line | Second in line |
|---:|---:|---:|
| 2 | 0:43 | 2:33 |
| 3 | 0:44 | 2:37 |
| 4 | 0:46 | 2:43 |
| 5 | 0:49 | 2:51 |
| 6 | 0:53 | 2:58 |
| 7 | 0:57 | 3:06 |
| 8 | 1:01 | 3:13 |

**Load.** A local server with the real queue router, 25 students in line and
two windows calling, with 25 phones each asking for their status and the queue
prediction every 2 seconds (twice the app's rate) for 60 seconds: 1,500
requests, 0 errors, median 5.5 ms (status) and 10.0 ms (prediction), slowest
215 ms. Every student seen waiting had an estimate that changed during the run.
Measured on a laptop; Render's free instance is slower and adds network time.

### 8.4 Choosing n and k

The live average was tried with three settings on the same replayed lines. Ten
services with k = 3 swung too much on normal days (R² while waiting 0.926 at one
window and 95% busy, against 0.968 with the average fixed at October's). Thirty
services with k = 20 trailed on fast days and on a day that turns 50% slower at
midday. Twenty with k = 10 did best on fast days, came within 0.003 of the best
on the day that turns slower, and was close to the best on normal days.

## 9. Measuring accuracy at the cashier

Every ticket logs what the engine told the student and what then happened, in
`queue_records`:

| Column | Set when | Holds |
|---|---|---|
| `people_ahead_at_issue` | The number is issued | Position in line minus one |
| `active_counters_at_issue` | The number is issued | Windows counted at that moment |
| `predicted_wait_min` | The number is issued | The estimate the student was shown |
| `wait_model_version` | The number is issued | `queue-engine-v2` for this engine |
| `called_at` | Staff mark the ticket done or no-show | When a window was first assigned to the person |
| `cashier_window` | Staff mark the ticket done or no-show | The window that served the person |

The real wait is `called_at − created_at`.

- **On the dashboard**, the Analytics tab's "Are the Wait Estimates Right?"
  panel scores the last 30 days of served tickets: how many were checked, the
  average miss, the share within 2 minutes, whether estimates run short or long,
  and the latest tickets as "app said X, called after Y". It does not compute
  R², and it includes tickets issued before this engine.
- **For R²**, download the service records (the panel's "Download service
  records (CSV)" button, or `ML/export_wait_log.py`), keep the rows with
  `wait_model_version = queue-engine-v2`, and compare `predicted_wait_min` with
  the real wait. This scores the estimate given at join; the estimates shown
  while waiting are not stored.

A session of 20 or more students, with staff holding windows and pressing Call
next and Done, gives the engine's first R² measured at the cashier.

## 10. Limitations

1. **Not yet measured at the cashier.** Every accuracy figure in section 8 is
   from simulation with real October service times (section 9).
2. **The shape comes from one month logged in whole minutes.** σ = 0.425 was
   fitted to 511 transactions; none took longer than 8 minutes, so the time
   left after 8 minutes is the lognormal's extrapolation.
3. **A forgotten Done inflates the estimates.** A transaction still open counts
   in μ at its length so far, so a window left open for an hour makes everyone's
   estimate long until staff press Done. Once Done is pressed, a service over
   30 minutes is ignored.
4. **Gaps between people are not modelled.** The time between Done and the next
   Call next, and the walk to the window, are inside the measured service times
   only when staff press Call next promptly.
5. **First come, first served is assumed.** Skipped numbers, staff calling out
   of order and students who leave are not predicted; the estimate corrects
   itself on the next request.
6. **The 5-, 15- and 30-minute forecasts on the dashboard** run this engine on a
   line grown at the camera's current rate of growth. They are not validated.

## 11. Questions the panel may ask

**What is the formula?**
Section 4. A busy window's time left is R(e) = μ · m(e/μ), the mean residual
life of a lognormal fitted to the October cashier log; μ is the live average of
the last 20 services; the line is then played forward, one μ per person, and the
wait is when you reach a window. With one window: wait = R(e) + people ahead × μ.

**Why does Q2's time move later while Q1 takes long?**
Because the longer a transaction has run, the more it is expected to need, and a
long one also raises the average everyone behind it is given (section 5).

**Is it trained? Is this machine learning?**
It is not trained on past waits the way a machine-learning model is. It is a
queue model calibrated on real cashier data: the shape σ was fitted by maximum
likelihood to 511 real transactions, and the average μ is learned continuously
from every service staff finish. The rest is the queue itself, played forward
from what the system knows at that moment (section 4, "Is the engine trained?").
Because it does not fit itself to past waits, its accuracy comes from validation
on lines it has never seen.

**What is its R²?**
In simulation with real October service times, run through the real backend
code: 0.834 for the estimate shown while waiting (MAE 53 s) and 0.818 at join
(MAE 67 s); on replayed lines 0.961 and 0.967 (section 8). It has not yet been
measured at the cashier; every served ticket is logged for that (section 9).

**Why is R² lower for short lines?**
When most waits are under a minute or two, they barely vary, so even a 30-second
miss is a large share of the variation. The MAE there is 25 to 80 seconds.

**What does the student see if no window is calling?**
The estimate assumes one window opens now, the app shows "Windows calling: 0"
and "Paused", and the clock holds still until a window calls the next number.
