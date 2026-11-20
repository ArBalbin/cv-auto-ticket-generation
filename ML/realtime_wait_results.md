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
