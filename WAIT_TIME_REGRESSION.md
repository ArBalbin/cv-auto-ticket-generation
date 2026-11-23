# The Wait-Time Regression (benchmark and fallback)

The live system estimates waits with the real-time queue engine, documented in
`WAIT_TIME_MODEL.md`. This file keeps the record of the median regression that
came before it: how it was trained on the October 2026 cashier log, how it was
validated, and its limits. It still runs if `WAIT_ESTIMATOR=regression` is set.

**Training:** `ML/train_wait_model.py`
**Artifact:** `Model/wait_time_model.json` (coefficients plus the full validation results)
**Runtime:** `app/services/wait_model.py`

Section numbers are kept from the earlier combined document, so references such
as "section 5" still point to the right place.

---

## 1. The regression (benchmark and fallback)

```
wait = 1.1214 + 0.1445 × people_ahead          (minutes, at 3 counters)
```

It is a **median regression**: the line that minimises the absolute error,
which suits waits that are usually short with occasional very long ones.

| People ahead | Estimate shown |
|---|---|
| 0 | 1 min |
| 5 | 2 mins |
| 10 | 3 mins |
| 20 | 4 mins |
| 30 | 6 mins |

## 2. The data

528 real transactions observed at Windows 3, 4 and 5 on 14 days, 2–19 October
2026 (`dataset/ncf_cashier_department_clean.csv` in the `oshiiffs/aiprediction`
repository, commit `59eeec5`). The cleaning is documented in that repository's
`REPORT.md`. The file stays out of this repo; its SHA-256 is recorded in the
model artifact.

**People ahead is a reconstruction.** The log does not record the queue length,
so it is computed from what was known when the ticket was issued:

```
people_ahead = ticket number − highest ticket number already called − 1
```

It could be computed for 511 of the 528 transactions. The other 17 were issued
before any observed ticket had been called that day, so they are left out of
the fit.

## 3. Why this model and not the one in the aiprediction repository

That repository's production model is a **constant**: a scikit-learn
`DummyRegressor` that predicts 2 minutes for every ticket. This is the correct
conclusion from its inputs. It was given only the time of day and the
department, and on unseen days no machine-learning model using those inputs
beat the constant (Ridge MAE 3.63, Random Forest 3.71, constant 2.99).

The time and department carry no information about how long the line is. The
live system knows the line, so QuEx uses it. A constant would show "about 2
minutes" to the 30th person in line, and it would not change when staff open or
close a counter.

The repository's pickle files are also not deployed directly: loading them with
scikit-learn and pandas costs 108 MB of memory, and the server has 512 MB.

## 4. Validation

**Leave-one-day-out:** each predictor is fitted on 13 days and tested on the
14th, for every day, and the errors are pooled. Whole days are held out because
tickets from the same day share its congestion, so a random split would leak it.

| Predictor | MAE (min) | RMSE (min) | R² | MAPE* | WAPE |
|---|---:|---:|---:|---:|---:|
| **Median regression on people ahead (deployed)** | **3.00** | **5.90** | **−0.04** | **72.2%** | **75.2%** |
| Same, with a cap chosen on the training days | 2.91 | 5.98 | −0.07 | 61.9% | 72.8% |
| Constant 2 minutes (aiprediction production model) | 2.99 | 6.13 | −0.12 | 59.7% | 74.7% |
| Previous position rule (batches ahead × service time) | 5.33 | 10.40 | −2.22 | 232.2% | 133.5% |
| Previous queueing formula (Q/λ + 1/μ) | 16.58 | 46.89 | −64.48 | 772.5% | 414.8% |

\* MAPE excludes the 56 zero-minute waits.

**How to read this honestly:**

- **The deployed model ties the constant on MAE.** The difference is −0.02
  minutes, 95% CI −0.28 to +0.17 (bootstrap over days), and it wins on 7 of the
  14 days. It is not more accurate on average. The reason to deploy it is that
  it responds to the queue at the same accuracy, with slightly lower RMSE and R².
- **It is far better than what the system ran before.** The previous formulas
  are penalised here by the overcounted people-ahead values (section 5), so the
  exact factors should not be quoted as their live accuracy. The direction agrees
  with the manuscript's September evaluation, where the same formula over-predicted
  75% of tickets.
- **R² is slightly negative:** on unseen days, its squared error is a little
  larger than always predicting that data's average. Most of the variation in
  these waits comes from things the log does not capture.
- **The capped variant has the lowest MAE, but it was not chosen.** Its estimate
  stops rising at 3 to 4 minutes however long the line gets, which is not a
  credible thing to show a student.
- **One day dominates the error.** 19 October (109 tickets) has an MAE of 7.20;
  no other day is above 2.67. Waits that day reached 45 minutes. Every predictor
  tested was poor on it.
- **The constant is close more often.** It lands within 2 minutes of the real
  wait for 75.0% of tickets, against 66.3% for the deployed model. Both land
  within 5 minutes for about 86%.

**Chronological test.** The aiprediction repo's update (commit `01ed9b1`) holds
out the last two days instead: fit on 2–16 October, test once on 17 and 19
October. `--test-from 2026-10-17` reproduces it.

| Predictor | MAE (min) | RMSE (min) | R² | Within 2 min | Within 5 min |
|---|---:|---:|---:|---:|---:|
| **Median regression on people ahead (deployed)** | **6.69** | **11.07** | **−0.23** | 48.7% | 65.5% |
| Constant 2 minutes | 7.00 | 11.66 | −0.36 | 55.5% | 65.5% |

109 of the 119 test tickets come from 19 October, so this is mostly a test on
one busy day. **Fitted on 2–16 October alone, the slope is only 0.033 minutes
per person**, against 0.145 on all 14 days: most of the people-ahead effect in
this data comes from that one day.

The usual range around an estimate is **−1.86 to +6.87 minutes**, from errors on
unseen days (79.9% coverage). The API sends it with every estimate
(`estimated_wait_range_label`), and the app shows it beside the student's own
estimate, for example "3 mins · usually 0–10 mins".

**Check on the method.** The same script run on the manuscript's September data
(166 transactions, 3 days) gives 2.07 for the constant and 1.86 for the capped
line, against 2.07 and 1.85 in the manuscript. On that data the capped line
beat the constant. The uncapped line deployed here scored 2.04.

## 5. Limitations

1. **Long lines are under-estimated.** People ahead is overcounted in the
   training data: the log observed only 6–31% of each day's ticket numbers, and
   tickets never served at these windows are counted, reaching 186 "ahead". This
   flattens the slope to 0.14 minutes per person, and without 19 October it
   drops to 0.033. In a real line of 30 at three windows with 2-minute service,
   the wait would be about 20 minutes, while the model says 6. The live count is
   exact and the training count is not, so **the slope must be re-fitted on
   measured counts** (section 6).
2. **It has not been checked against live waits.** All figures are offline, on
   logged transactions. The system now logs what that check needs (section 6),
   but no comparison has been run yet.
3. **The training period is short:** 14 days in one month, with one extreme day.
4. **Counter scaling is an assumption.** The per-person term is multiplied by
   `3 / active counters` so that staff changes affect the estimate. The data
   cannot validate this because nearly all of it comes from the same three
   windows. At three counters the model is exactly the fitted line.
5. **The 5-, 15- and 30-minute forecasts are not validated.** They project the
   camera count forward (trend, Holt, growth ratio) and then apply the model to
   the projected count. Only the mapping from count to wait is trained.

## 6. Measuring real waits

The October log lacks the one input the model depends on, a real count of the
people ahead. Two sources now provide it, in the same CSV format, and
`ML/train_wait_model.py` reads either.

**The system's own log.** Each ticket row in `queue_records` now stores:

| Column | Set when | Holds |
|---|---|---|
| `people_ahead_at_issue` | The number is issued | Position in line minus one |
| `active_counters_at_issue` | The number is issued | Counters open at that moment |
| `predicted_wait_min` | The number is issued | The estimate the student was shown |
| `wait_model_version` | The number is issued | Which model made it |
| `called_at` | Staff mark the ticket done or no-show | When a counter was first assigned to the person |
| `cashier_window` | Staff mark the ticket done or no-show | The window that served the person |

The real wait is `called_at − created_at`. A person counts as called when the
system assigns them a counter, which happens when the person before them at that
counter is marked done. The backend adds the columns at startup; for databases
migrated by hand the same change is `database_sql/2026_11_wait_log_migration.sql`.

```powershell
.venv/Scripts/python.exe ML/export_wait_log.py
```

This writes `ML/data/wait_time/system_wait_log.csv` (read-only query). Training
on it also reports how far the estimates the system showed were from the waits
that followed: the live accuracy SOP 3 asks for.

**Download without database access.** The same panel has a "Download service
records (CSV)" button (`GET /api/queue/service_records.csv`, staff only). It
gives every served ticket's real wait and service time in seconds, the people
ahead, the counters open, the serving window, and the estimate shown. It has no
names, school IDs or student links, so it can go to whoever retrains the model.
`cashier_window` comes from `database_sql/2026_11_service_window_migration.sql`,
which the backend also applies at startup.

**On the dashboard.** The Analytics tab's "Are the Wait Estimates Right?"
panel (`GET /api/queue/prediction_accuracy`, staff only) does the same scoring
live over the last 30 days. For served tickets it shows how many were checked,
the average miss, and the share within 2 minutes. It scores a fixed 2-minute
guess on the same tickets, so the panel says plainly whether the engine is
beating the constant. It also shows whether estimates tend to run short or long,
and the latest tickets as "app said X, called after Y". It reads
`fetch_wait_observations` and caches the result for 30 seconds.

**Observation at the cashier.** `ML/wait_data_collection/` holds a recording
sheet and instructions: one row per ticket, with a head count of the line at the
moment the ticket is taken. A few days of this re-fits the slope on real counts
with the cashier's real service times.

**Deciding whether to replace the model.** Train to a separate file, then compare:

```powershell
.venv/Scripts/python.exe ML/train_wait_model.py --csv ML/data/wait_time/system_wait_log.csv --out ML/data/wait_time/candidate_model.json
```

On new data the script also scores the model in use, unchanged. Copy the
candidate over `Model/wait_time_model.json` only if the re-fitted line beats both
that model and the constant on unseen days. The script refuses fewer than three
days of data.

## 7. What was tested and left out

- **Service-time scaling.** Multiplying the per-person term by recent service
  time made it worse on unseen days (MAE 3.08 against 3.00), so it was left out of the regression.
  The real-time engine (`WAIT_TIME_MODEL.md`) uses the live average service time directly,
  because it plays the line forward one service at a time instead of fitting a
  slope.
- **Machine-learning models with queue inputs** (people ahead, time, arrival
  rate, recent service). Ridge, Random Forest and Gradient Boosting reached MAE
  2.93–3.01, also tying the constant. Ridge had the best R² (+0.24), but it needs
  an arrival rate measured from ticket numbers, which the live system does not
  have, so its accuracy would not carry over.
- **Department.** QuEx does not collect it, and adding it improved Ridge by only
  0.002 minutes in the aiprediction report.

These comparisons were one-off runs and are not part of `ML/train_wait_model.py`.

## 8. Retraining

```powershell
# October log: copy the cleaned CSV to ML/data/wait_time/, or pass --csv
.venv/Scripts/python.exe ML/train_wait_model.py --test-from 2026-10-17
```

This needs only numpy and scipy. Without `--out` it rewrites
`Model/wait_time_model.json`, which must be committed. Restart both the backend
and the detector to load it.

## 9. Questions the panel may ask

**Is this machine learning?**
It is a trained statistical model: its coefficients are learned from data and
validated on unseen days. Complex models were also tested and did not do better
on this data, so the simplest one that performs as well was selected.

**Why not use a more complex model?**
On 528 transactions, Ridge, Random Forest and Gradient Boosting all tied a
constant on MAE. A more complex model would add memory and risk without adding
accuracy.

**Your R² is negative. Isn't the model bad?**
It means the wait is hard to predict from the available data: on unseen days the
model does about as well as always predicting the average, not better. Most
waits are short and a few are very long, and the log does not record what caused
the long ones. We report this rather than hide it.

**Why isn't it better than the constant, and why use it?**
It ties the constant on average error. But a constant tells the 30th person the
same as the first. This model gives the same accuracy while responding to the
line and to staff opening counters.

**Why does it say 6 minutes for 30 people?**
This is a known limitation. The training data overcounts the people ahead, which
flattens the slope, so long lines are under-estimated. The system now logs the
real count and the real wait for every ticket, so the slope can be re-fitted on
exact data.

**How accurate is it in the running system?**
Open the dashboard's "Are the Wait Estimates Right?" panel and read it out:
tickets checked, average miss, the share within 2 minutes, and the fixed
2-minute guess on the same tickets. If it has too few tickets, say so, and say
that the system scores every served ticket on its own.

**Why not train it on the October data?**
That data has no real count of the people ahead (section 0). Every method scored
on it ties or loses to a constant: machine learning, the regression, and this
engine alike. A better model cannot recover a count that was never recorded, so
the system records it now.
