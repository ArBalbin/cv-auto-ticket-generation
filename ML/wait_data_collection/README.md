# Recording waits at the cashier

The wait-time model was trained on a log that never recorded how many people
were in line, so it had to guess that number from ticket numbers. This sheet
records the real count. A few days of it can re-fit the model on the input the
live system actually uses (see `WAIT_TIME_MODEL.md`, section 5).

Use `recording_sheet.csv`: one row per ticket you follow from start to finish.

## Columns

| Column | What to write | Example |
|---|---|---|
| `date` | The day, as year-month-day | `2026-11-12` |
| `ticket_number` | The number on the ticket | `57` |
| `window` | The window that served it | `Window 4` |
| `ticket_time` | When the ticket was taken, 24-hour clock | `09:14` |
| `people_ahead` | People **waiting** for Windows 3–5 at that moment. Count heads in the line, not the ones already at a window. | `6` |
| `windows_open` | Windows serving this queue at that moment | `3` |
| `called_time` | When the person reached the window | `09:19` |
| `finished_time` | When they left the window | `09:21` |

Seconds are allowed (`09:14:30`) but not needed.

## How to record

- Two people work best: one takes the ticket time and counts the line at once,
  the other watches the windows for the called and finished times. Set both
  phone clocks to the same time first.
- Follow a ticket only if you can see all of it, from taking the ticket to
  leaving the window. A ticket you lose sight of is left out, not guessed.
- Record at least **3 days**. Include a busy day if one comes: busy days are
  where the model is weakest.
- **Do not write names, student numbers, or anything else that identifies a
  person.** The sheet needs times and counts only.
- Ask the cashier office's permission before recording, as was done for the
  October log.

## Training on it

Save the sheet as CSV (UTF-8) in `ML/data/wait_time/`, which git ignores. Then
write a candidate model to a separate file, so the deployed one stays in place:

```powershell
.venv/Scripts/python.exe ML/train_wait_model.py --csv ML/data/wait_time/recording_sheet.csv --out ML/data/wait_time/candidate_model.json
```

The script also scores the model in use, unchanged, on the same days. Copy the
candidate over `Model/wait_time_model.json` only if the re-fitted line beats
both that model and the constant on unseen days.
