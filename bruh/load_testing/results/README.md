# Load-test results — read this before quoting any number

## 2026-11 scalability runs (pre-final panel requirement)

Run after the scalability changes: in-memory face-embedding index, a
ticket-validation cache with stale-while-revalidate, a background refresh of
the waiting list, the connection-leak fix, and busy (503) instead of crash
(500) when the pool is exhausted. Backend on the development laptop, connected
to the cloud MySQL and Redis, retention scheduler off, a real seeded ticket.

The "realistic" profile (`--profile realistic`) is mostly students polling
their ticket, with about one display board and one staff dashboard per 20
users. Display boards and staff screens do not grow with enrolment, so the
original equal split (as many staff as students) overstates them at scale.

| Files | Users | Profile | Requests | HTTP errors | Connection errors | req/s | p50 | p95 | p99 |
|---|---|---|---|---|---|---|---|---|---|
| `scale_050_equal_*` | 50 | equal | 2,400 | 0 | 0 | 20.5 | 9 ms | 110 ms | 410 ms |
| `scale_200_realistic_*` | 200 | realistic | 6,678 | 0 | 9 (0.13 %) | ~61 | 11 ms | 93 ms | 280 ms |
| `scale_400_realistic_*` | 400 | realistic | 13,025 | 0 | 75 (0.58 %) | 109.9 | 21 ms | 330 ms | 720 ms |

Before the changes, the same 200-user run produced over 1,000 "pool
exhausted" errors, 401s and 500s, and a steady-state p95 of 1.3 to 3 seconds.

Reading these numbers:

- **Connection errors** are `ConnectionAbortedError`/`ConnectionResetError`
  between Locust and the server on the same Windows laptop. No request
  received an HTTP error.
- In `scale_050_equal_*` the status poll alone improved on `final_readonly_*`
  (p50 8 ms vs 13 ms, p95 18 ms vs 26 ms). The higher aggregate p95 comes from
  staff login and analytics, which were as slow with no load at all on the day
  (login about 2 s, analytics about 75 ms, both waiting on the remote MySQL and
  Redis services), so it reflects the network that day rather than the code.
- `face_match_scaling.csv` is the in-process face-matching benchmark
  (`load_testing/bench_face_match.py`): 3.2 ms per match against 10,000
  enrolled students, down from 10.2 s.

## Earlier runs

Of the runs before 2026-11, only **`final_readonly_*`** is a valid measurement of this system.

The other files in this directory are diagnostic runs kept as a record of how
the valid result was arrived at. Quoting them would misrepresent the system,
and in one case would misrepresent it as far worse than it is.

| Files | Requests | Failures | p50 | p99 | Use? |
|---|---|---|---|---|---|
| `final_readonly_*` | 2,536 | **0.0 %** | 12 ms | **85 ms** | ✅ **Quote this one** |
| `readonly_*` | 2,450 | 0.0 % | 12 ms | 2,100 ms | ❌ measurement artifact |
| `student_only_*` | 2,019 | 0.0 % | 11 ms | 2,100 ms | ❌ measurement artifact |
| `low_concurrency_*` | 267 | 0.0 % | 9 ms | 2,100 ms | ❌ measurement artifact |
| `../results_readonly_*` | 252 | **30.6 %** | 7 ms | 2,100 ms | ❌ broken test setup |

## Why the invalid runs are invalid

**The ~2,100 ms p99 was never server latency.** Those runs targeted
`http://localhost:5000`. On Windows, `localhost` resolves to `::1` (IPv6)
first, while uvicorn binds `0.0.0.0` (IPv4 only), so every new connection
waits about two seconds for the IPv6 attempt to time out before falling back.
Each simulated user pays that once, on its first request, which lands exactly
in the p98/p99 bucket.

Measured directly, five fresh connections each way:

```
http://localhost:5000    first-request ms: [2059, 2036, 2072, 2043, 2041]
http://127.0.0.1:5000    first-request ms: [8, 7, 8, 7, 9]
```

`load_testing/run_load_test.py` now defaults to `127.0.0.1` for this reason.

**The 30.6 % failure rate was a broken test, not a broken server.** The seeder
read the ticket's raw `access_token` instead of its short code, so every
`/api/queue/status` request authenticated with the wrong credential and was
correctly rejected with 401. The server behaved properly; the test asked the
wrong question. The seeder now polls until the real short code appears.

## Reproducing the valid run

```bash
python load_testing/run_load_test.py
```

Defaults: 50 concurrent users, 2 minutes, three user classes (student status
poll, public display board, staff analytics), against a real seeded ticket
rather than a 404 path — those are different code paths with different costs,
and quoting the 404 latency would understate what a real ticket-holder
experiences. The seeded ticket is removed automatically afterwards.
