# Load-test results — read this before quoting any number

Only **`final_readonly_*`** is a valid measurement of this system.

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
