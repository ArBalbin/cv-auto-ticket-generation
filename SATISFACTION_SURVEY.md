# QueuEx — Researcher-Made Likert Questionnaire (SOP #4)

Version 3.0 — September 24, 2026

This is the same instrument as the questionnaire appendix of
`Balbin_Archie_W3.tex` (*QueuEx Evaluation Questionnaire*). If the two ever
differ, the manuscript is the one the panel reads — change both together.

> **SOP #4:** *What is the level of user's satisfaction in terms of
> (a) system's usability; (b) system's reliability; and (c) system clarity?*

---

## Shortened evaluation protocol

The manuscript originally called for two weeks of regular use, a separate
pilot of 10–15 respondents, and a Slovin-sized student sample. None of these
fit the time left before the final defense, so the protocol was shortened and
**the manuscript says so explicitly** (Data Gathering Procedure,
*Shortened evaluation protocol*). What the evaluation now does:

| Original design | Shortened protocol (what the manuscript now states) |
|---|---|
| Two weeks of regular operation first | One **supervised trial session**; the survey is answered immediately afterwards |
| Separate pilot, 10–15 respondents | **Cronbach's alpha on the actual responses**, per dimension, per group |
| Content validation by three experts | **Kept** — one review round, before the survey is used |
| Students sized by Slovin's formula | **Convenience sample** of the students who volunteer and complete the trial |
| Staff: total enumeration | Unchanged — every staff member who operated the dashboard in the session |

Consequence stated in the manuscript: SOP #4 results describe users' *first
experience under supervision*, not satisfaction after sustained use.

---

## Design

- Each respondent answers **15 items, 5 per dimension**.
- **Two items per dimension are common** to both forms (U1–U2, R1–R2, C1–C2),
  so students and staff can be compared on identical statements.
- **Three per dimension are role-specific** — a student cannot judge the staff
  dashboard, and staff cannot judge the mobile app.
- Every item is worded so that agreement is favourable. **No reverse-scoring.**
- Codes: `U` usability, `R` reliability, `C` clarity; `-s` student, `-f` staff
  and administrators. **Keep the hyphen** — `ML/score_survey.py` only
  recognises codes written as `U3-s`, and silently skips `U3s`.

---

## SCREENING — before anything else

> **Are you 18 years old or above?**  ☐ Yes  ☐ No

If **No**, thank them and end the form. It must come *before* consent.

## INFORMED CONSENT

> **About this study.** You are invited to give feedback on QueuEx, a queue
> management system being evaluated at Naga College Foundation as part of an
> undergraduate thesis.
>
> **What you will do.** Answer a short questionnaire about the trial session,
> taking about five minutes.
>
> **Anonymity.** Your name, student number and email are not recorded with
> your answers. Results are reported only as group totals.
>
> **Face recognition.** If you enrolled, the system stored a set of numbers
> computed from your face — **not a photograph**. No image of your face is
> saved, and the numbers cannot be turned back into your face. You may ask for
> your enrolment to be deleted at any time.
>
> **Voluntary.** You may skip any question or stop at any time, with no effect
> on the service you receive.
>
> ☐ I have read and understood the above, and I agree to take part.

---

## SECTION I — Respondent profile

1. Respondent code (assigned by researchers): ________
2. **Respondent group:** ☐ Service Staff ☐ Administrator / IT Personnel ☐ Student
3. *(Students)* Year level: ☐ 1st ☐ 2nd ☐ 3rd ☐ 4th ☐ 5th or above
4. *(Students)* How did you receive your queue number during the session?
   ☐ Automatically, after the camera recognised me
   ☐ From staff, because the system could not recognise me
5. *(Staff/Admin)* Your role during the session:
   ☐ Operated the staff dashboard ☐ Observed or supervised

**Rating scale:** 5 = Strongly Agree · 4 = Agree · 3 = Neutral ·
2 = Disagree · 1 = Strongly Disagree

---

## SECTION II — Student form (15 items)

| Code | Statement |
|---|---|
| | ***A. System's usability*** |
| U1 | The system was easy to use without needing someone to guide me. |
| U2 | I was able to complete what I needed to do without difficulty. |
| U3-s | Registering my face in the mobile app was easy. |
| U4-s | Asking to be queued from the mobile app was convenient. |
| U5-s | I received my queue number without having to do anything else at the office. |
| | ***B. System's reliability*** |
| R1 | The system worked consistently throughout the session. |
| R2 | I did not encounter errors that stopped me from continuing. |
| R3-s | The system recognised me correctly when I entered the queue area. |
| R4-s | I trust that the queue number I received was mine and not someone else's. |
| R5-s | My position in the queue was updated correctly as the line moved. |
| | ***C. System clarity*** |
| C1 | The information shown on screen was easy to understand. |
| C2 | I always knew what the system was doing or waiting for. |
| C3-s | My position in the queue was clearly shown. |
| C4-s | The estimated waiting time was presented in a way I could understand. |
| C5-s | I understood what information the system keeps about me. |

## SECTION III — Staff and administrator form (15 items)

| Code | Statement |
|---|---|
| | ***A. System's usability*** |
| U1 | The system was easy to use without needing someone to guide me. |
| U2 | I was able to complete what I needed to do without difficulty. |
| U3-f | Navigating the staff dashboard was easy. |
| U4-f | Marking an entry as done or as a no-show was quick to do. |
| U5-f | When the system could not identify someone, resolving it from the dashboard was straightforward. |
| | ***B. System's reliability*** |
| R1 | The system worked consistently throughout the session. |
| R2 | I did not encounter errors that stopped me from continuing. |
| R3-f | The queue numbers the system issued matched the students actually present. |
| R4-f | When the system was unsure about someone, it referred them to staff instead of guessing. |
| R5-f | The queue list on the dashboard stayed accurate as students were served. |
| | ***C. System clarity*** |
| C1 | The information shown on screen was easy to understand. |
| C2 | I always knew what the system was doing or waiting for. |
| C3-f | The dashboard layout made it easy to find what I needed. |
| C4-f | The live camera view and the queue list were clear to read. |
| C5-f | The wait-time predictions and analytics were presented in a way I could act on. |

> **R4-f matters more than its single row suggests.** The system is designed to
> refuse rather than guess. If staff report that it guessed, that contradicts
> the measured open-set result and must be investigated, not averaged away.

## SECTION IV — Additional questions (not scored)

1. Did the system ever fail to recognise you, or give you a number you did not
   expect? ☐ No ☐ Yes, once ☐ Yes, more than once
2. If yes, what happened? ________
3. How comfortable are you with the system using your face to identify you in
   the queue? ☐ Very comfortable ☐ Comfortable ☐ Neutral ☐ Uncomfortable
   ☐ Very uncomfortable
4. What did you like most about QueuEx? ________
5. If you could change one thing about QueuEx, what would it be? ________

---

## Before the trial session

1. **Content validation — keep this step.** The thesis adviser, one faculty
   member in research methods or statistics, and one in computer science or
   information systems each rate every item 1–4 for relevance, in one round.
   I-CVI = raters giving 3 or 4 ÷ 3; each item needs **≥ 0.78** (all three
   must agree), and S-CVI/Average needs **≥ 0.90**. Revise any failing item
   *before* the session. Score with
   `python ML/score_survey.py --cvi cvi_ratings.csv`. Keep the signed forms.
2. **Enrol the student volunteers** in the mobile app before the session.

## Running it in Google Forms

1. Put the **age screening first** and end the form on "No"; consent second.
2. Make **Respondent group** a required question — the scorer reads any column
   whose header contains "group".
3. Branch on the group: students see only Section II, staff and administrators
   only Section III.
4. Make each dimension a *Multiple choice grid*, and **start every row's text
   with its code**, e.g. `U3-s. Registering my face in the mobile app was easy.`
   Keep the hyphen.
5. Turn **off** "Collect email addresses" — the consent promises anonymity.
6. Have respondents answer **immediately after the session**.
7. Include people whose recognition failed. They are the most informative
   respondents, and leaving them out would bias the result.

## Scoring

```bash
python ML/score_survey.py --responses responses.csv --csv-out results.csv
```

This computes, per group: the weighted mean of each item, each dimension, and
the group; the overall weighted mean; and **Cronbach's alpha per dimension
within each group** — which, under the shortened protocol, is the reliability
figure the manuscript reports. With fewer than 10 respondents in a group the
script marks alpha as too small to rely on; report it with that caveat, as was
done for the staff side of SOP #1.

Interpretation follows the manuscript's Likert table:

| WM | Interpretation |
|---|---|
| 4.21 – 5.00 | Strongly Agree / Excellent |
| 3.41 – 4.20 | Agree / Very Satisfactory |
| 2.61 – 3.40 | Neutral / Satisfactory |
| 1.81 – 2.60 | Disagree / Poor |
| 1.00 – 1.80 | Strongly Disagree / Very Poor |

Compare students and staff on the **common items only** (U1–U2, R1–R2,
C1–C2); the role-specific items are different questions and cannot be
compared across groups.
