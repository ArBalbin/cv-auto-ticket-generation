# Real-time wait estimate: replayed accuracy

Produced by `ML/evaluate_realtime_wait.py` on 20 Nov 2026, after the final
defense panel's comments on the wait-time prediction.

**What this is.** Lines replayed in simulation, with service lengths drawn from
the 511 transactions in the October 2026 cashier log (each logged whole minute
v drawn between v - 0.5 and v + 0.5). The system knows the queue exactly here:
who is at each window and since when, and how many wait ahead. It is not a
measurement at the cashier. Live accuracy comes from the wait log
(`wait_model_version = queue-engine-v2`, `/api/queue/prediction_accuracy`).

**Engines.**
- *Old*: deployed until 20 Nov 2026. A window's remaining time is
  max(average - elapsed, 0.75 x average), so every estimate behind a long
  transaction froze. Scored with the October average it starts from.
- *Old, true avg*: the same engine handed each day's true average, which no
  live system knows. A ceiling for the old engine, not something it achieved.
- *New*: `app/services/queue_engine.py`. Remaining time from a lognormal fitted
  to the October log, scaled by a live average (last 20 services in two hours,
  October average counted as 10 more, a transaction running past the average
  counted at its length so far).

**Days.** Normal (as October), slow (every transaction 40% longer), fast
(25% shorter). Arrivals are sized so each day is as busy as labelled.

Pooled R² mixes scenarios whose waits differ tenfold, which inflates it; read
the per-scenario rows and the MAE.


AT JOIN (estimate given on joining vs the wait they had)
| Scenario | Mean wait (min) | Old R² | Old MAE | Old, true avg R² | Old, true avg MAE | New R² | New MAE | Estimates |
|---|---|---|---|---|---|---|---|---|
| 1 window, 60% busy, normal day | 2.0 | 0.686 | 83 s | 0.686 | 83 s | 0.684 | 80 s | 6900 |
| 1 window, 85% busy, normal day | 8.9 | 0.948 | 103 s | 0.948 | 103 s | 0.933 | 110 s | 6900 |
| 1 window, 95% busy, normal day | 29.5 | 0.980 | 164 s | 0.980 | 164 s | 0.965 | 210 s | 6900 |
| 2 windows, 60% busy, normal day | 0.8 | 0.411 | 49 s | 0.411 | 49 s | 0.599 | 38 s | 6900 |
| 2 windows, 85% busy, normal day | 4.2 | 0.929 | 62 s | 0.929 | 62 s | 0.927 | 58 s | 6900 |
| 2 windows, 95% busy, normal day | 14.5 | 0.979 | 88 s | 0.979 | 88 s | 0.964 | 107 s | 6900 |
| 3 windows, 60% busy, normal day | 0.4 | -0.088 | 38 s | -0.088 | 38 s | 0.483 | 25 s | 6900 |
| 3 windows, 85% busy, normal day | 2.7 | 0.898 | 48 s | 0.898 | 48 s | 0.916 | 41 s | 6900 |
| 3 windows, 95% busy, normal day | 9.5 | 0.976 | 65 s | 0.976 | 65 s | 0.963 | 73 s | 6900 |
| 1 window, 85% busy, slow day | 12.5 | 0.809 | 239 s | 0.948 | 144 s | 0.925 | 158 s | 6900 |
| 2 windows, 85% busy, slow day | 5.9 | 0.825 | 115 s | 0.929 | 86 s | 0.925 | 79 s | 6900 |
| 3 windows, 85% busy, slow day | 3.8 | 0.831 | 76 s | 0.898 | 68 s | 0.919 | 55 s | 6900 |
| 1 window, 85% busy, fast day | 6.7 | 0.735 | 168 s | 0.948 | 77 s | 0.903 | 97 s | 6900 |
| 2 windows, 85% busy, fast day | 3.2 | 0.663 | 98 s | 0.929 | 46 s | 0.888 | 53 s | 6900 |
| 3 windows, 85% busy, fast day | 2.0 | 0.576 | 74 s | 0.898 | 36 s | 0.871 | 37 s | 6900 |
| **All pooled** | 7.1 | 0.955 | 98 s | 0.977 | 77 s | 0.967 | 81 s | 103500 |

REAL TIME (estimate shown every 30 s while waiting vs the wait still ahead)
| Scenario | Mean wait (min) | Old R² | Old MAE | Old, true avg R² | Old, true avg MAE | New R² | New MAE | Estimates |
|---|---|---|---|---|---|---|---|---|
| 1 window, 60% busy, normal day | 3.1 | 0.773 | 65 s | 0.773 | 65 s | 0.766 | 60 s | 29352 |
| 1 window, 85% busy, normal day | 9.9 | 0.932 | 101 s | 0.932 | 101 s | 0.912 | 111 s | 126292 |
| 1 window, 95% busy, normal day | 27.3 | 0.971 | 162 s | 0.971 | 162 s | 0.954 | 200 s | 409735 |
| 2 windows, 60% busy, normal day | 1.6 | 0.552 | 50 s | 0.552 | 50 s | 0.685 | 38 s | 12450 |
| 2 windows, 85% busy, normal day | 4.9 | 0.908 | 62 s | 0.908 | 62 s | 0.898 | 61 s | 61155 |
| 2 windows, 95% busy, normal day | 13.7 | 0.968 | 88 s | 0.968 | 88 s | 0.951 | 104 s | 202721 |
| 3 windows, 60% busy, normal day | 1.1 | 0.123 | 48 s | 0.123 | 48 s | 0.586 | 31 s | 7173 |
| 3 windows, 85% busy, normal day | 3.3 | 0.869 | 51 s | 0.869 | 51 s | 0.882 | 44 s | 39642 |
| 3 windows, 95% busy, normal day | 9.1 | 0.963 | 65 s | 0.963 | 65 s | 0.948 | 72 s | 133703 |
| 1 window, 85% busy, slow day | 13.8 | 0.761 | 247 s | 0.932 | 142 s | 0.904 | 158 s | 175563 |
| 2 windows, 85% busy, slow day | 6.9 | 0.773 | 121 s | 0.907 | 87 s | 0.897 | 84 s | 84494 |
| 3 windows, 85% busy, slow day | 4.6 | 0.780 | 82 s | 0.868 | 71 s | 0.888 | 60 s | 54451 |
| 1 window, 85% busy, fast day | 7.4 | 0.656 | 173 s | 0.932 | 76 s | 0.869 | 99 s | 95455 |
| 2 windows, 85% busy, fast day | 3.7 | 0.548 | 104 s | 0.908 | 46 s | 0.844 | 56 s | 46590 |
| 3 windows, 85% busy, fast day | 2.5 | 0.414 | 82 s | 0.870 | 38 s | 0.817 | 41 s | 30454 |
| **All pooled** | 14.1 | 0.958 | 130 s | 0.974 | 107 s | 0.961 | 123 s | 1509230 |

## The panel's case, minute by minute

One window. Q1 has been at it for the time in the first column; Q2 and Q3 wait.
Four services of 2:00 were timed earlier. Times are min:sec of waiting left.

| Q1 at the window | Old Q2 | Old Q3 | New Q2 | New Q3 | Average shown |
|---|---|---|---|---|---|
| 0:00 | 1:52 | 3:44 | 1:54 | 3:49 | 1:54 |
| 0:30 | 1:24 | 3:16 | 1:25 | 3:19 | 1:54 |
| 1:00 | 1:24 | 3:16 | 1:01 | 2:56 | 1:54 |
| 1:30 | 1:24 | 3:16 | 0:50 | 2:45 | 1:54 |
| 2:00 | 1:24 | 3:16 | 0:46 | 2:41 | 1:55 |
| 3:00 | 1:24 | 3:16 | 0:46 | 2:44 | 1:59 |
| 5:00 | 1:24 | 3:16 | 0:51 | 2:58 | 2:07 |
| 8:00 | 1:24 | 3:16 | 1:03 | 3:22 | 2:19 |
| 10:00 | 1:24 | 3:16 | 1:11 | 3:38 | 2:27 |
| 15:00 | 1:24 | 3:16 | 1:31 | 4:18 | 2:47 |
| 20:00 | 1:24 | 3:16 | 1:52 | 4:58 | 3:07 |

The old estimates stop moving after 30 seconds. The new ones count down while
Q1 takes a usual time, then grow while Q1 keeps going, and each student's
estimated time of service moves later the whole time.

## Pressure test on the real backend code (20 Nov 2026)

`ML/pressure_test_realtime_wait.py` runs the real queue tracker, queue
service and engine with the database switched off and a simulated clock.
Students arrive at random and get a number; staff at each window press Call
next, serve for a time drawn from the October log, press Done, and call again
five seconds later. One transaction per run takes 9 minutes. Every waiting
student's phone asks for its estimate every 5 seconds, and each answer is
checked against when that student was really called. The old engine is asked
the same question at the same moment. Three runs per row.

| Students | Windows | Busy | Mean wait | Longest wait | Longest line | Phone checks | Live R² new | Live MAE new | Live R² old | Live MAE old | At-join R² new | At-join MAE new | Frozen 60 s+ new | Frozen 60 s+ old |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 24 | 1 | 85% | 5.1 min | 11.6 min | 5 | 1478 | 0.704 | 53 s | 0.700 | 69 s | 0.651 | 85 s | 1 | 43 |
| 24 | 1 | 85% | 8.8 min | 15.0 min | 7 | 2545 | 0.848 | 57 s | 0.796 | 75 s | 0.751 | 99 s | 0 | 70 |
| 24 | 1 | 85% | 6.2 min | 13.1 min | 6 | 1800 | 0.864 | 49 s | 0.832 | 58 s | 0.775 | 72 s | 2 | 56 |
| 30 | 2 | 90% | 2.8 min | 7.2 min | 7 | 1013 | 0.680 | 49 s | 0.676 | 51 s | 0.713 | 58 s | 1 | 28 |
| 30 | 2 | 90% | 5.8 min | 9.8 min | 11 | 2091 | 0.715 | 59 s | 0.802 | 50 s | 0.681 | 78 s | 0 | 60 |
| 30 | 2 | 90% | 4.7 min | 9.2 min | 9 | 1678 | 0.719 | 54 s | 0.734 | 52 s | 0.677 | 77 s | 2 | 48 |
| 40 | 3 | 95% | 0.9 min | 4.2 min | 5 | 422 | 0.367 | 41 s | 0.240 | 47 s | 0.701 | 25 s | 0 | 12 |
| 40 | 3 | 95% | 7.1 min | 14.6 min | 18 | 3398 | 0.852 | 59 s | 0.875 | 53 s | 0.763 | 97 s | 0 | 115 |
| 40 | 3 | 95% | 3.8 min | 8.7 min | 12 | 1816 | 0.826 | 37 s | 0.795 | 42 s | 0.848 | 39 s | 0 | 55 |

Pooled over the 9 runs (282 students, 16,241 phone checks): live R² 0.834
(MAE 53 s) new, 0.829 (MAE 56 s) old; at join, R² 0.818 (MAE 67 s) new.
These days run at exactly October's pace, which the old engine assumes, so
the two are about equally accurate here; the difference is that the old
estimates froze. "Frozen 60 s+" counts each time a waiting student's
estimate stayed identical for a minute or more while someone was at a window.

The panel's case inside the first run: the 9-minute transaction of Q5005 at
the only window, and the waits of the next two in line (min:sec).

| Minutes at window | Next in line: new | old | Second in line: new | old |
|---|---|---|---|---|
| 2 | 0:43 | 1:24 | 2:33 | 3:16 |
| 3 | 0:44 | 1:24 | 2:37 | 3:16 |
| 4 | 0:46 | 1:24 | 2:43 | 3:16 |
| 5 | 0:49 | 1:24 | 2:51 | 3:16 |
| 6 | 0:53 | 1:24 | 2:58 | 3:16 |
| 7 | 0:57 | 1:24 | 3:06 | 3:16 |
| 8 | 1:01 | 1:24 | 3:13 | 3:16 |

## Pressure test with the time staff take to call (25 Nov 2026)

The test sessions of 20 to 24 November logged 20 served tickets; 14 had nobody
ahead and were told 0:00, and staff pressed Call next about 71 s later on
average. The engine now learns that time from the calls (formula 4 in
`WAIT_TIME_MODEL.md`). The pressure test was changed so that staff call a set
time after they can, also when a student arrives at a free window (before, a
student arriving at a free window was called at once), and run with and
without the learned call time. Same 9 runs, 282 students.

| Staff take | Engine | Live R² | Live MAE | At-join R² | At-join MAE | Frozen 60 s+ |
|---|---|---|---|---|---|---|
| 5 s | without the call time | 0.834 | 53 s | 0.813 | 70 s | 6 |
| 5 s | with it | 0.826 | 55 s | 0.811 | 71 s | 10 |
| 30 s | without the call time | 0.834 | 76 s | 0.749 | 121 s | 10 |
| 30 s | with it | 0.896 | 60 s | 0.834 | 96 s | 10 |
| 60 s | without the call time | 0.672 | 168 s | 0.500 | 270 s | not counted |
| 60 s | with it | 0.919 | 73 s | 0.844 | 131 s | not counted |

Per run with staff taking 5 s and the call time learned (the table in
`WAIT_TIME_MODEL.md` section 8.3): pooled over 16,523 phone checks, live R² 0.826
(MAE 55 s); at join R² 0.811 (MAE 71 s). The "frozen" counts with the call time
are the student next in line held at 45 to 59 s while the transaction ahead is
two to four minutes in, where the expected time left is nearly flat, as in the
runs without it, plus one held at 3:16 and called 3 s later.

The panel's case inside the first run, with the call time:

| Minutes at window | Next in line | Second in line |
|---|---|---|
| 2 | 0:47 | 2:40 |
| 3 | 0:47 | 2:44 |
| 4 | 0:49 | 2:50 |
| 5 | 0:53 | 2:57 |
| 6 | 0:56 | 3:05 |
| 7 | 1:00 | 3:12 |
| 8 | 1:04 | 3:20 |

## Load test (20 Nov 2026)

A local server with the real queue router (database off), 25 students in
line, two windows calling and finishing a transaction every 12 seconds, and
25 phones each asking for their status and the queue prediction every 2
seconds (twice the app's rate) for 60 seconds, all at once.

| Requests | Errors | Median | 95th percentile | Slowest |
|---|---|---|---|---|
| 750 status | 0 | 5.5 ms | 35.3 ms | 214.9 ms |
| 750 prediction | 0 | 10.0 ms | 39.6 ms | 175.7 ms |

All 23 students seen waiting had an estimate that changed during the run;
the 7 called to a window were sent the 60-second arrival time. On this
laptop, not on Render, whose free instance is slower and adds network time.
