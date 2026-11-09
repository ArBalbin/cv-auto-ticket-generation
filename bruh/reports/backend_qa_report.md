# QueueFlow Backend QA Report

**Execution date:** 2026-11-09 (Asia/Manila)  
**Assessment:** **NOT VERIFIED** for production release  
**Automated suite:** 125 passed, 0 failed, 0 skipped, 0 collection errors

## A. Executive Summary

The complete discovered Pytest suite passed. Coverage includes mocked student profile and access checks, queue and ticket lifecycle regressions, model integration with the supplied ETS artifact, department-aware FIFO simulation, wait-log behavior, and in-memory scale checks.

Two defects found during review were fixed and covered by regression tests: timezone-aware model inputs now use the configured school timezone, and non-finite model output now falls back safely. Direct model inference matches the backend adapter for all six supported departments.

No production or development database credentials, `.env` file, disposable MySQL instance, or isolated running API was available. Consequently, database persistence and migration behavior, database-backed concurrency, authenticated end-to-end flows, and API load/latency were not verified. The release should wait for those checks. The cashier adjustment endpoint also accepts up to ten windows while this QA prompt specifies a maximum of five; that existing behavior is recorded for product/configuration resolution.

## B. Test Environment

| Item | Observed |
|---|---|
| Python | 3.14.3 |
| FastAPI | 0.162.2 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.2 |
| Pytest | 9.2.0 |
| Database type | MySQL through `mysql-connector-python`; no database connection was configured or attempted |
| Model | Bundled trusted ETS artifact, version 1.0.0 |
| Test isolation | Unit tests and mocked database/session services; no live database mutations |
| Coverage tool | Not installed; coverage percentage not measured |
| API load environment | None configured; no API load test run |

The workspace had no `.env` file and no database host/name environment values. Tests did not connect to Aiven, localhost MySQL, or any production system. The existing top-level `aiven_schema.sql` contains destructive schema setup statements; it was inspected but never executed.

## C. Test Results

Full command: `python -m pytest -q`  
Result: **125 passed in 5.76 seconds**. `python -m compileall -q app tests` also completed successfully. An OpenAPI import check confirmed the department update and training-export routes are registered.

Category counts below are mapped from the passing suite and overlap where a test covers more than one category. They are not additive.

| Test Category | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Authentication and authorization checks (mocked sessions/roles) | 9 | 0 | 0 |
| Department profile validation and response | 2 | 0 | 0 |
| Ticket lifecycle, cache, and queue engine regression | 46 | 0 | 0 |
| Trained ETS artifact and fallback integration | 7 | 0 | 0 |
| ETS/EWT compatibility and completion-duration calculation | 17 | 0 | 0 |
| FIFO simulation, mixed departments, busy and zero-window cases | 18 | 0 | 0 |
| Database-related unit tests (mocked/fake database only) | 32 | 0 | 0 |
| In-memory scale check for face matching at 10,000 embeddings | 1 | 0 | 0 |
| Real MySQL persistence, migrations, and transaction integrity | Not run | — | — |
| Concurrent ticket creation with MySQL (5/10/25/50 clients) | Not run | — | — |
| API load, latency, throughput, CPU, and memory | Not run | — | — |
| Live polling/client update behavior | Not run end-to-end | — | — |

The performance unit test confirms the face matching implementation meets its test’s `<50 ms` per-match assertion for a 10,000-entry in-memory gallery. It is not an API or concurrent-user performance result. No API average or P95 latency was measured.

## D. Bug Report

### QA-001 — Host timezone could shift ETS model time features

- **Severity:** Medium
- **Status:** Fixed and regression-tested
- **Affected code:** `app/services/ets_prediction.py`
- **Reproduction:** Pass a timezone-aware UTC datetime for a time represented in school-local time. Previously the adapter read the datetime’s original `.hour` without converting it.
- **Expected:** `hour` and `day_of_week` match the school-local timestamps used by the training pipeline.
- **Actual before fix:** A UTC hour could be supplied where the trained model expects the school-local hour. This is especially relevant on UTC-hosted deployments.
- **Root cause:** The adapter used `datetime.now()` / the input object's raw hour rather than the configured school timezone.
- **Fix:** Convert timezone-aware input to `SCHOOL_UTC_OFFSET_HOURS`; naive input remains interpreted as school-local training time.
- **Verification:** A regression test converts 02:30 UTC to 10:30 at UTC+8 and checks the feature received by inference.

### QA-002 — Non-finite ETS output could escape prediction validation

- **Severity:** Medium
- **Status:** Fixed and regression-tested
- **Affected code:** `app/services/ets_prediction.py`, `app/services/queue_simulator.py`
- **Expected:** Predictions and FIFO durations are finite and nonnegative; invalid model output selects a fallback.
- **Actual before fix:** Positive infinity from a pipeline could be returned as ETS. The simulator also did not reject NaN or infinite durations/residuals.
- **Root cause:** Validation checked negativity but not finiteness.
- **Fix:** Reject non-finite model output so the configured fallback is used, sanitize a non-finite fallback, and reject non-finite FIFO durations and cashier residuals.
- **Verification:** Regression tests cover an infinite model result and NaN/infinite/negative FIFO service durations.

### QA-003 — Cashier limit differs from the QA specification

- **Severity:** Medium
- **Status:** Open; behavior preserved pending configuration decision
- **Affected code:** `app/routers/queue.py` (`adjust_counters`)
- **Reproduction:** The endpoint validates `counters` from 1 through 10, while this QA prompt describes a maximum of five cashier windows.
- **Expected:** A maximum of five if the prompt’s limit is authoritative.
- **Actual:** Values 6 through 10 are accepted.
- **Root cause:** The existing QueueFlow endpoint and runtime configuration allow ten.
- **Fix:** None applied. Clamping this would remove existing supported configuration; confirm the deployed cashier limit before changing it.
- **Verification:** Source inspection. No live API request was made.

### QA-004 — Production validation does not require database and trusted-host configuration

- **Severity:** Medium
- **Status:** Open; requires deployment configuration review
- **Affected code:** `app/core/config.py`, `app/main.py`, `render.yaml`
- **Finding:** Production validation requires several auth/storage settings but does not require database environment values or a restricted `TRUSTED_HOSTS` value. The application configuration has development database defaults, and the default wildcard trusted-host setting disables `TrustedHostMiddleware`. `render.yaml` leaves database values for operator configuration and does not define `TRUSTED_HOSTS`.
- **Risk:** A misconfigured production deployment may start with unsafe/default database settings or without host-header filtering rather than failing fast.
- **Fix:** None applied because the correct production database and allowed hostnames are environment-specific and were unavailable for verification.
- **Verification:** Static configuration review only; no deployment or secret values were accessed or printed.

## E. ML Prediction Verification

- **Artifact:** `models/ets_model.joblib` loads successfully; metadata and artifact report version `1.0.0`.
- **Required features:** `department`, `hour`, `day_of_week`. The saved pipeline retains its own preprocessing and department encoding.
- **Units:** The artifact predicts minutes; the backend adapter converts to seconds and enforces finite, nonnegative values.
- **Departments:** CCS, CBM, CHS, CCJE, COE, and CTED were all evaluated with the real artifact. Adapter results matched direct pipeline inference for each department using the same feature row.
- **Example feature row:** 2026-11-09 10:30 school-local time. Results in seconds: CCS 106.091; CBM 106.699; CHS 105.683; CCJE 99.008; COE 95.939; CTED 105.683. Outputs are not required to differ for every pair; in this sample CHS and CTED match.
- **Fallbacks:** Unknown departments, missing artifact, and non-finite output return a global-average fallback in unit tests. The application also queries completed, department-specific service records for department-average fallback when available. No live MySQL measurements were available to test that query.
- **Caching:** Model loading is cached process-wide; inference is cached by department/hour/day. Functional load-time savings were not benchmarked.
- **EWT:** The backend uses FIFO simulation operationally for department-aware ticket predictions. The experimental EWT regression model was not made operational. The training project reports EWT FIFO baseline performance better than its experimental regressor on the supplied split.
- **Statistical accuracy:** Training metadata reports ETS test MAE 0.673 minutes and R² about -0.013 on 355 supplied test rows. The training report says the test rows are generated and source records are not independently verified. These values do not establish real-world accuracy.

Functional correctness of loading, features, units, and fallbacks was tested. Prediction accuracy on real transactions was not.

## F. Performance Results

| Load | Result |
|---|---|
| In-memory face gallery, 10,000 embeddings | Test passed its `<50 ms` per-match threshold |
| API clients, 10/25/50/100 | Not run; no isolated API server configured |
| Ticket throughput / prediction latency | Not measured |
| Median/P95 latency, error rate, CPU, memory | Not measured |
| Database query latency/pool pressure | Not measured |

The existing Locust runner can drive an API host, but no isolated host or test credentials were available. It was not pointed at a potentially live service.

## G. Coverage and Limitations

- No coverage percentage was produced because `coverage.py` was unavailable.
- MySQL student department persistence, ticket snapshots, foreign keys, unique constraints, rollback, reconnection, and migration data preservation were not tested against a disposable MySQL instance.
- Ticket creation concurrency at 5/10/25/50 requests was not tested with database locking or transaction semantics. In-memory queue tests do not prove database-level uniqueness under concurrency.
- Valid/invalid Google login, expired sessions, account-to-database department persistence, and the complete student-to-cashier workflow were not exercised against external auth or MySQL.
- Admin export role enforcement was unit-tested with a mocked database role; export SQL, CSV contents from MySQL, and deployed authorization were not integration-tested.
- Individual cashier closure/temporary unavailability is not represented by a persistent per-window database state in the tested flow. Existing engine tests cover scheduled closures and the department simulator covers occupied windows, but not a live cashier dashboard operation.
- Current real-time behavior is polling-based; no mobile app source was found in this workspace. Client polling and network failure behavior were not tested end-to-end.
- No rate-limiting middleware was found in the inspected API configuration. Rate-limit behavior and dedicated SQL-injection fuzz testing were not verified.
- The actual production settings for database SSL, allowed hosts, CORS origins, and staff roles are unavailable. No secret values or personal data were included in this report.

## H. Final QA Assessment

**Classification: NOT VERIFIED.**

Evidence supports a passing automated unit/regression suite and successful loading of the supplied ETS model. The two confirmed prediction defects were corrected. Production readiness remains unverified because the required database integration, database-backed concurrency, authenticated end-to-end workflows, and API load/performance tests could not run safely in this environment. Before release, run those checks against an isolated MySQL schema and staging API, confirm whether the cashier maximum is five or ten, and validate production database and host settings.
