# QueuEx — Researcher-Made Likert Questionnaire (SOP #4)

Version 2.0 — September 12, 2026

**Built to match the manuscript exactly.** Every structural choice below is
taken from `Balbin_Archie_W3.tex`, not invented:

| Element | Source in the manuscript |
|---|---|
| Three dimensions: usability, reliability, clarity | §Research Instrument, and SOP #4 §1.7 |
| 5-point scale and its verbal interpretation | Table `tab:likert` |
| Three respondent groups | §Sample and Sampling Technique |
| Weighted mean per item / dimension / group, then OWM | §Statistical Treatment, eq. `weighted_mean`, `overall_weighted_mean` |
| I-CVI ≥ 0.78, S-CVI/Ave ≥ 0.90, 3 expert raters | §Research Instrument, Instrument validation |
| Pilot 10–15 respondents, Cronbach's α ≥ 0.70 | §Research Instrument, Reliability testing |
| Under-18 screening before consent | §Sample, Exclusion criterion |
| Semi-structured interview guide | §Research Instrument |

> **SOP #4 as written:** *What is the level of user's satisfaction in terms of
> (a) system's usability; (b) system's reliability; and (c) system clarity?*

---

## How the three groups are handled

The manuscript specifies one questionnaire with three dimensions, administered
to **service staff**, **administrators**, and **students**. But its own
description of those dimensions spans interfaces the three groups do not
share — "mobile client" is a student's view, while "counter controls",
"analytics cards" and "live snapshot" exist only on the staff dashboard.

Asking a student to rate counter controls produces a meaningless number, and
dropping the item for some respondents breaks the weighted mean.

So each dimension has **common items** (answerable by everyone, forming the
comparable core) and **group-specific items**. Weighted means are computed per
group as the manuscript requires; the common items are what make the
cross-group comparison in §Statistical Treatment legitimate.

Item codes: `U` = usability, `R` = reliability, `C` = clarity.
Suffix `-s` = student only, `-f` = staff/administrator only.

---

# SCREENING — ask before anything else

> **Are you 18 years old or above?**
> ☐ Yes  ☐ No

**If No: stop here.** Thank them for their interest and do not proceed to
consent. The manuscript's exclusion criterion requires this to come *before*
the consent form, not after.

---

# INFORMED CONSENT

> **About this study.** You are invited to give feedback on QueuEx, a queue
> management system being evaluated at Naga College Foundation as part of an
> undergraduate thesis.
>
> **What you will do.** Answer a short questionnaire about your experience,
> taking about five minutes.
>
> **Anonymity.** Your name, student number and email are not recorded with
> your answers. Results are reported only as group totals.
>
> **Face recognition.** If you enrolled, the system stored a set of numbers
> computed from your face — **not a photograph**. No image of your face is
> saved or transmitted, and the stored numbers cannot be turned back into
> your face. You may ask for your enrolment to be deleted at any time and
> will still be served as a walk-in.
>
> **Voluntary.** You may skip any question or stop at any time, with no
> effect on the service you receive.
>
> ☐ I have read and understood the above, and I agree to take part.

---

# THE QUESTIONNAIRE

**Scale:** 5 = Strongly Agree · 4 = Agree · 3 = Neutral · 2 = Disagree ·
1 = Strongly Disagree

---

## Dimension A — System's Usability

*Manuscript scope: the dashboard, mobile client, and ticket workflow.*

**Common items — all groups**

| Code | Statement | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|
| U1 | The system was easy to use without needing someone to guide me. | ☐ | ☐ | ☐ | ☐ | ☐ |
| U2 | I was able to complete what I needed to do without difficulty. | ☐ | ☐ | ☐ | ☐ | ☐ |
| U3 | The steps I had to follow were few and straightforward. | ☐ | ☐ | ☐ | ☐ | ☐ |
| U4 | I learned to use the system quickly. | ☐ | ☐ | ☐ | ☐ | ☐ |

**Students only**

| Code | Statement | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|
| U5-s | Registering my face in the mobile app was easy to do. | ☐ | ☐ | ☐ | ☐ | ☐ |
| U6-s | Joining the queue from the mobile app was convenient. | ☐ | ☐ | ☐ | ☐ | ☐ |
| U7-s | Getting my queue number required no extra effort from me. | ☐ | ☐ | ☐ | ☐ | ☐ |
| U8-s | Checking my queue status on my phone was easy. | ☐ | ☐ | ☐ | ☐ | ☐ |

**Staff and administrators only**

| Code | Statement | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|
| U5-f | Navigating the staff dashboard was easy. | ☐ | ☐ | ☐ | ☐ | ☐ |
| U6-f | Marking a queue number as served was quick to do. | ☐ | ☐ | ☐ | ☐ | ☐ |
| U7-f | Entering a walk-in queue number manually was straightforward. | ☐ | ☐ | ☐ | ☐ | ☐ |
| U8-f | The system fit into how I normally do my work. | ☐ | ☐ | ☐ | ☐ | ☐ |

---

## Dimension B — System's Reliability

*Manuscript scope: queue assignment accuracy, recognition of returning users,
and no-show detection — **as perceived by users**. These items ask what people
experienced, not what the logs recorded; the measured accuracy figures come
from the operational log, not from here.*

**Common items — all groups**

| Code | Statement | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|
| R1 | The system gave the correct queue number every time. | ☐ | ☐ | ☐ | ☐ | ☐ |
| R2 | The system worked consistently throughout my use of it. | ☐ | ☐ | ☐ | ☐ | ☐ |
| R3 | I did not encounter errors that stopped me from continuing. | ☐ | ☐ | ☐ | ☐ | ☐ |
| R4 | I can depend on this system for day-to-day use. | ☐ | ☐ | ☐ | ☐ | ☐ |

**Students only**

| Code | Statement | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|
| R5-s | The system recognised me correctly when I entered the queue area. | ☐ | ☐ | ☐ | ☐ | ☐ |
| R6-s | I trust that the number I received was mine and not someone else's. | ☐ | ☐ | ☐ | ☐ | ☐ |
| R7-s | When I returned to the queue area, the system still recognised me correctly. | ☐ | ☐ | ☐ | ☐ | ☐ |

**Staff and administrators only**

| Code | Statement | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|
| R5-f | The queue numbers the system issued matched the people actually present. | ☐ | ☐ | ☐ | ☐ | ☐ |
| R6-f | When the system was unsure about someone, it asked me instead of guessing. | ☐ | ☐ | ☐ | ☐ | ☐ |
| R7-f | The no-show handling reflected what actually happened in the queue. | ☐ | ☐ | ☐ | ☐ | ☐ |

> **R6-f matters more than its single row suggests.** The system is designed to
> refuse rather than guess when a match is ambiguous. If staff report that it
> guessed instead, that contradicts the measured open-set result and must be
> investigated, not averaged away.

---

## Dimension C — System Clarity

*Manuscript scope: dashboard layout, live snapshot, analytics cards, counter
controls, and the presentation of wait-time predictions.*

**Common items — all groups**

| Code | Statement | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|
| C1 | The information shown on screen was easy to understand. | ☐ | ☐ | ☐ | ☐ | ☐ |
| C2 | I always knew what the system was doing or waiting for. | ☐ | ☐ | ☐ | ☐ | ☐ |
| C3 | The labels and wording used were clear to me. | ☐ | ☐ | ☐ | ☐ | ☐ |
| C4 | When something went wrong, the message told me what to do next. | ☐ | ☐ | ☐ | ☐ | ☐ |

**Students only**

| Code | Statement | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|
| C5-s | My position in the queue was clearly shown. | ☐ | ☐ | ☐ | ☐ | ☐ |
| C6-s | The estimated waiting time was presented in a way I could understand. | ☐ | ☐ | ☐ | ☐ | ☐ |
| C7-s | The printed ticket showed the information I needed. | ☐ | ☐ | ☐ | ☐ | ☐ |
| C8-s | I understood what information the system keeps about me. | ☐ | ☐ | ☐ | ☐ | ☐ |

**Staff and administrators only**

| Code | Statement | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|
| C5-f | The dashboard layout made it easy to find what I needed. | ☐ | ☐ | ☐ | ☐ | ☐ |
| C6-f | The live camera view and queue list were clear to read. | ☐ | ☐ | ☐ | ☐ | ☐ |
| C7-f | The analytics cards presented useful information clearly. | ☐ | ☐ | ☐ | ☐ | ☐ |
| C8-f | The counter controls were clearly labelled and easy to understand. | ☐ | ☐ | ☐ | ☐ | ☐ |
| C9-f | The wait-time predictions were presented in a way I could act on. | ☐ | ☐ | ☐ | ☐ | ☐ |

---

## Respondent profile

*Used for the percentage distribution in §Statistical Treatment. Kept
non-identifying.*

- **P1.** Group: ☐ Student ☐ Service staff ☐ Administrator
- **P2.** *(Students)* Year level: ☐ 1st ☐ 2nd ☐ 3rd ☐ 4th ☐ Other
- **P3.** *(Staff/Admin)* Length of involvement with the system: ☐ 2–3 weeks ☐ 1 month ☐ More than 1 month
- **P4.** *(Students)* Did you register your face? ☐ Yes ☐ No, I was served as a walk-in
- **P5.** How many times did you use the system during the evaluation period? ☐ Once ☐ 2–3 times ☐ 4 or more

---

## Supplementary questions

> **Not part of any weighted mean.** These are reported descriptively. They are
> kept outside the three dimensions deliberately, so the dimension structure
> the manuscript defines is not altered.

**S1.** Did the system ever fail to recognise you, or give you an unexpected
number? ☐ No ☐ Yes, once ☐ Yes, more than once

**S2.** If yes, what happened?
_______________________________________________________________

**S3.** How comfortable are you with the system using your face to identify
you for the queue?
☐ Very comfortable ☐ Comfortable ☐ Neutral ☐ Uncomfortable ☐ Very uncomfortable

**S4.** What did you like most about the system?
_______________________________________________________________

**S5.** What would you change or improve?
_______________________________________________________________

---

# SEMI-STRUCTURED INTERVIEW GUIDE

*The manuscript specifies this as a fourth instrument, used flexibly. Conduct
with staff and administrators, and with a small number of students who
reported problems in S1.*

**Opening**
1. Walk me through what happened the first time you used QueuEx.

**Usability**
2. Was there any point where you were not sure what to do next?
3. What did you have to learn before you could use it comfortably?

**Reliability**
4. Did the system ever get something wrong? What did you do about it?
5. *(Staff)* In what situations would you override what the system suggested?

**Clarity**
6. Was there anything on screen you found confusing or unnecessary?
7. *(Staff)* Were the wait-time predictions useful to you? Did you act on them?

**Acceptance**
8. How do you feel about being identified by face rather than pressing a button?
9. Would you want this system kept in daily use? Why or why not?

**Closing**
10. What one change would make the biggest difference?

---

# SCORING

Follow §Statistical Treatment exactly.

## Weighted mean per item

$$\text{WM} = \frac{\sum(f \times w)}{N}$$

where *f* is the frequency of each response, *w* is its weight (1–5), and *N*
is the number of respondents in that group.

## Then aggregate

1. **Per item**, per group.
2. **Per dimension**: mean of the item WMs within that dimension.
3. **Per group**: mean across the three dimensions.
4. **Overall weighted mean (OWM)**: $\text{OWM} = \frac{\sum \text{WM}}{n}$

Report per-group weighted means so the stakeholder perspectives can be
compared, as the manuscript requires.

## Interpretation — Table `tab:likert`

| WM range | Interpretation |
|---|---|
| 4.21 – 5.00 | Strongly Agree / Excellent |
| 3.41 – 4.20 | Agree / Very Satisfactory |
| 2.61 – 3.40 | Neutral / Satisfactory |
| 1.81 – 2.60 | Disagree / Poor |
| 1.00 – 1.80 | Strongly Disagree / Very Poor |

## Percentage distribution

$$P = \frac{f}{N} \times 100$$

Report for respondent composition and for the response spread on each item.

> **No item here is reverse-worded.** Every statement is phrased so that
> agreement is positive, which means responses can be summed directly with no
> recoding step. This is deliberate: a reverse-worded item that someone forgets
> to recode silently corrupts the weighted mean, and there is no way to detect
> it afterwards from the totals alone.

---

# BEFORE YOU DISTRIBUTE IT — required by the manuscript

The manuscript commits to two validation steps. Skipping either leaves a gap
a panel can point to, and neither can be done retroactively.

## 1. Content validation — three expert raters

The manuscript names them: the **thesis adviser**, one faculty member with
expertise in **research methods or statistics**, and one with expertise in
**computer science or information systems**.

Each rates every item independently on 1–4 relevance (1 = not relevant,
4 = highly relevant).

**I-CVI** (per item) = (raters scoring 3 or 4) ÷ (total raters).
With 3 raters: all three must rate 3–4 for I-CVI = 1.00; two of three gives
0.67, which is **below the 0.78 threshold** and the item must be revised or
removed.

**S-CVI/Average** = mean of all I-CVI values. Must reach **0.90**.

Keep the signed forms — the manuscript says they go in the appendices.

### Content validation form

| Item | Statement | 1 | 2 | 3 | 4 | Comments |
|---|---|---|---|---|---|---|
| U1 | *(copy each item)* | ☐ | ☐ | ☐ | ☐ | |
| … | | | | | | |

Rater name: ________________  Expertise: ________________
Signature: ________________  Date: __________

## 2. Pilot test — Cronbach's alpha

Pilot with **10–15 respondents** from the same population but **excluded from
the final sample**. Compute Cronbach's α per dimension; **α ≥ 0.70** is the
threshold. If it falls short, revise the items with low item-total correlation
and repeat.

Score both with:

```bash
python ML/score_survey.py --responses responses.csv
python ML/score_survey.py --cvi cvi_ratings.csv
```

---

# RUNNING IT

1. **Build in Google Forms.** Each table becomes one "Multiple choice grid".
   Use **section branching on P1** so students never see `-f` items and staff
   never see `-s` items. Turn **off** "Collect email addresses" — the consent
   notice promises anonymity.
2. **Put the age screening first**, before consent, with branching that ends
   the form on "No".
3. **Administer immediately after the transaction.** Satisfaction recalled a
   week later measures memory, not experience.
4. **Include the people who had problems.** A respondent whose recognition
   failed is the most informative one you will get; excluding them is how a
   survey becomes indefensible.
5. **Export to CSV** and score with `ML/score_survey.py`.

## Sample size

Per §Sample and Sampling Technique:

- **Staff and administrators:** total enumeration — invite everyone eligible.
- **Students:** Slovin's formula at 5 % margin, $n = N / (1 + Ne^2)$, where *N*
  is the number of students who transact with the office during the evaluation
  period. Then stratify purposively across year levels.

Record *N* during the beta — the operational log gives it directly, and
without it the Slovin computation cannot be shown.
