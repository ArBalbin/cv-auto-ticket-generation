# A third person getting a number while others are in view

Simulated 25 Nov 2026 with `load_testing/simulate_third_person.py`, before and
after the fix of the same day. The simulation runs the real backend code
(`queue_service`, `queue_tracker`) and the detector's real face schedule
(`face_schedule`) on a simulated clock. No camera, database or deployed
service is used.

## Why

In testing, when two people already had numbers, a third person who came to
get one had a hard time being detected and given a number. The live log
agreed: on 24 Nov one student stood in view for 58 seconds after being
confirmed before getting a number, with only 2 face reads in 52 camera
updates, and another took 78 seconds just to be confirmed.

## What was simulated

- One update reaches the backend each second. The detector pushes every frame,
  but each push waits for the backend's answer, which took about 0.8 s from
  the office (measured 25 Nov), and the 24 Nov log shows about one a second.
- The face worker reads one face in 0.26 s, choosing whom to read the way the
  detector does. A read finds the person's face 65% of the time when they
  stand clear, 45% when they overlap someone.
- When two people overlap, 10% of reads catch the neighbour's face instead.
  On someone standing right in front of a hidden person, 3% catch that
  person's face peeking past.
- Each student has a look for the day as similar to their registration photo
  as real reads were in the face-match log (0.45 to 0.75), and each read is
  close to that look.

## Three people (200 runs per row)

Students 1 and 2 already have numbers. Student 3 stands in a free spot
(beside), half in front of student 1 with both visible (overlapping), right
in front of student 1 after walking in from the side (in front), or comes
into view already in front of student 1 (appears in front). "Tapped" is
whether student 3 pressed Get a Queue Number before arriving or after
standing in view for 5 seconds; times count from the later of the two.

Before the fix:

| Student 3 | Tapped | Got a number | Median | 90th pct | Within 5 s | Within 10 s | Own number, on own box | 1 and 2 keep theirs, on own box |
|---|---|---|---|---|---|---|---|---|
| beside | before arriving | 200 | 3 s | 5 s | 186 | 200 | 200 | 200 |
| beside | after 5 s in view | 200 | 2 s | 4 s | 193 | 200 | 200 | 200 |
| overlapping | before arriving | 183 | 4 s | 7 s | 153 | 179 | 183 | 200 |
| overlapping | after 5 s in view | 169 | 3 s | 7 s | 139 | 165 | 169 | 200 |
| in front | before arriving | 200 | 3 s | 10 s | 156 | 192 | 200 | 200 |
| in front | after 5 s in view | 196 | 2 s | 6 s | 176 | 194 | 196 | 196 |
| appears in front | before arriving | 200 | 5 s | 10 s | 132 | 182 | 200 | 200 |
| appears in front | after 5 s in view | 195 | 2 s | 6 s | 169 | 190 | 195 | 195 |

After the fix:

| Student 3 | Tapped | Got a number | Median | 90th pct | Within 5 s | Within 10 s | Own number, on own box | 1 and 2 keep theirs, on own box |
|---|---|---|---|---|---|---|---|---|
| beside | before arriving | 200 | 1 s | 1 s | 200 | 200 | 200 | 200 |
| beside | after 5 s in view | 200 | 1 s | 2 s | 200 | 200 | 200 | 200 |
| overlapping | before arriving | 200 | 1 s | 2 s | 199 | 200 | 200 | 200 |
| overlapping | after 5 s in view | 200 | 2 s | 3 s | 197 | 200 | 200 | 200 |
| in front | before arriving | 200 | 1 s | 1 s | 200 | 200 | 200 | 200 |
| in front | after 5 s in view | 200 | 1 s | 2 s | 200 | 200 | 200 | 200 |
| appears in front | before arriving | 200 | 1 s | 1 s | 200 | 200 | 200 | 200 |
| appears in front | after 5 s in view | 200 | 1 s | 2 s | 200 | 200 | 200 | 200 |

"On own box" checks that the number sits on that student's own body in the
picture, not only that the record names the right student. Before the fix,
in the "in front" rows, student 1's number ended up on student 3's body in
up to 5 of 200 runs.

## A busy line (24 students, 30 runs)

Students arrive every 4 to 16 seconds. Half take a free spot, a quarter stand
half in front of someone, a quarter right in front of someone (who is hidden
until a spot frees up). 60% tapped before arriving, the rest 3 to 8 seconds
after. Staff serve the lowest number every 50 to 110 seconds and that student
leaves. The detector renumbers someone now and then (0.5% of people each
second). A student who has no number 60 seconds after asking and being in
view gives up and leaves. Counted: students who asked and whom the camera
could see.

| | Students | Got a number | Median | 90th pct | Within 5 s | Within 10 s | Never | Students whose number sat on someone else | Duplicate tickets |
|---|---|---|---|---|---|---|---|---|---|
| Before | 664 | 594 | 3 s | 28 s | 368 | 454 | 70 | 246 (7,005 student-seconds) | 0 |
| After | 601 | 600 | 1 s | 3 s | 595 | 599 | 1 | 17 (32 student-seconds) | 0 |

There are more students before the fix because those who gave up left and
freed their spot sooner, so more people got into view.

The one student without a number after the fix was in view for 5 seconds
before asking, then stood hidden behind someone for the rest of the minute.
The 17 mix-ups after the fix last about 2 seconds on average (32
student-seconds in all). Most are a newcomer's box briefly mapped onto the
person beside or behind them before the newcomer's first face read, which
then separates them.

## What changed

Backend (`app/services/queue_service.py`, `queue_tracker.py`, new
`track_sightings.py`, `routers/detector_api.py`):

1. A registered student who asked for a number is confirmed and issued it on
   the first update that shows their face, instead of after a second and
   three updates.
2. A new person overlapping someone already in line is no longer thrown out
   on every update without a fresh face read; a face already captured for
   them counts.
3. Two detector numbers seen in the same update are never taken for one
   person. Without a face, a new box is mapped onto a missing person only in
   the same place and of the same size; with a face, the face must match and
   the size be similar.
4. A number moves to a new box only when three face reads on it agree, and
   never onto a box that appeared the moment that person vanished and is
   clearly bigger or smaller.
5. A neighbour's face caught in a numbered student's box is not stored as
   theirs; a student recognised before asking is marked as such only after
   two reads agree, so one stray read cannot lock them out.
6. The backend tells the detector when a student is waiting for a number
   (`faces_wanted`), and tapping Get a Queue Number while already recognised
   issues the number on the next update from the face already captured.

Detector (`app/detector.py`, `app/services/face_schedule.py`): while a
student is waiting for a number, unidentified people are read at the quick
pace instead of once a second after six tries.

## Limits

These are simulations with synthetic face signatures and box positions, not
measurements at the cashier. The face read rates and misread rates are
assumptions stated above. The backend's own speed is not simulated beyond
one update a second. The camera test on campus is the real check.

Run: `.venv/Scripts/python.exe load_testing/simulate_third_person.py --trials 200 --crowds 30`
