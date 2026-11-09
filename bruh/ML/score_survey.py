"""
Score the SOP #4 satisfaction questionnaire.

Implements the statistical treatment exactly as the manuscript specifies it
(Balbin_Archie_W3.tex, section "Statistical Treatment of Data"):

    WM  = sum(f * w) / N          per item, per respondent group
    OWM = sum(WM) / n             aggregated across items, then dimensions
    P   = f / N * 100             percentage distribution

plus the two validation measures the manuscript commits to before the
instrument may be used at all:

    Cronbach's alpha   per dimension, threshold 0.70
    I-CVI / S-CVI/Ave  per item and overall, thresholds 0.78 / 0.90

Why a script rather than a spreadsheet: the weighted mean has to be computed
at three levels for three groups, and every level is an opportunity to
average the wrong thing. Averaging raw 1-5 responses across dimensions
without going item -> dimension -> group first gives a different number that
looks equally plausible, and nothing in a spreadsheet catches it.

INPUT FORMAT
------------
A CSV exported from Google Forms. Column headers must begin with the item
code so they can be matched, e.g.

    "U1. The system was easy to use without needing someone to guide me."
    "R5-s. The system recognised me correctly when I entered the queue area."

One column must identify the respondent group; by default any column whose
header contains "Group" is used. Values are matched loosely against
student / staff / admin.

Blank cells are treated as not-answered and excluded from that item's N,
which is what the WM formula requires - N is the number of respondents to
that item, not the number of rows in the file.

Run:
    python ML/score_survey.py --responses responses.csv
    python ML/score_survey.py --responses responses.csv --csv-out results.csv
    python ML/score_survey.py --cvi cvi_ratings.csv
"""

import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Item codes at the start of a column header.
#
# The manuscript's Appendix B numbers its items A1..A6, B1.., C1.. — one
# letter per dimension, in the order the Statement of the Problem lists them
# (usability, reliability, clarity). That is the authoritative instrument, so
# A/B/C is what this script must read. U/R/C is accepted as well, because an
# earlier draft of the questionnaire used those initials and any responses
# already collected under them should still score.
ITEM_RE = re.compile(r"^\s*([ABCURC]\d+(?:-[sf])?)\b", re.IGNORECASE)

DIMENSIONS = {
    "A": "Usability",    # Appendix B
    "B": "Reliability",
    "C": "Clarity",      # shared by both schemes
    "U": "Usability",    # earlier draft
    "R": "Reliability",
}

# Table tab:likert in the manuscript. Ranges are inclusive of both bounds as
# printed there; a value is matched against the first range it falls into.
LIKERT_BANDS = [
    (4.21, 5.00, "Strongly Agree / Excellent"),
    (3.41, 4.20, "Agree / Very Satisfactory"),
    (2.61, 3.40, "Neutral / Satisfactory"),
    (1.81, 2.60, "Disagree / Poor"),
    (1.00, 1.80, "Strongly Disagree / Very Poor"),
]

GROUP_PATTERNS = [
    ("Student", ("student",)),
    ("Service staff", ("staff", "service")),
    ("Administrator", ("admin",)),
]


def interpret(wm: float) -> str:
    for low, high, label in LIKERT_BANDS:
        if low <= wm <= high:
            return label
    # Only reachable for a value outside 1.00-5.00, which means bad input.
    return f"OUT OF RANGE ({wm:.2f}) - check the response coding"


def normalise_group(raw: str) -> str | None:
    low = (raw or "").strip().lower()
    if not low:
        return None
    for label, needles in GROUP_PATTERNS:
        if any(n in low for n in needles):
            return label
    return raw.strip()


def load_responses(path: Path, group_header: str | None):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{path} has no data rows.")

    headers = list(rows[0].keys())

    item_cols = {}
    for h in headers:
        m = ITEM_RE.match(h)
        if m:
            item_cols[m.group(1).upper()] = h
    if not item_cols:
        sys.exit(
            "No item columns found. Column headers must start with the item "
            "code, e.g. 'U1. The system was easy to use...'.\n"
            f"Headers seen: {headers[:6]}"
        )

    if group_header is None:
        candidates = [h for h in headers if "group" in h.lower()]
        group_header = candidates[0] if candidates else None

    return rows, item_cols, group_header


def score_responses(args) -> None:
    path = Path(args.responses)
    if not path.exists():
        sys.exit(f"No such file: {path}")

    rows, item_cols, group_header = load_responses(path, args.group_column)

    # responses[group][item] = [1..5, ...]
    responses: dict = defaultdict(lambda: defaultdict(list))
    group_counts: dict = defaultdict(int)
    skipped = 0

    for row in rows:
        group = normalise_group(row.get(group_header, "")) if group_header else "All respondents"
        group = group or "Unspecified"
        group_counts[group] += 1
        for code, col in item_cols.items():
            raw = (row.get(col) or "").strip()
            if not raw:
                continue
            try:
                val = int(float(raw))
            except ValueError:
                # Google Forms can export the label instead of the number.
                low = raw.lower()
                if low.startswith("strongly agree"):
                    val = 5
                elif low.startswith("agree"):
                    val = 4
                elif low.startswith("neutral"):
                    val = 3
                elif low.startswith("strongly disagree"):
                    val = 1
                elif low.startswith("disagree"):
                    val = 2
                else:
                    skipped += 1
                    continue
            if not 1 <= val <= 5:
                skipped += 1
                continue
            responses[group][code].append(val)

    print("=" * 72)
    print("SOP #4 - Level of User Satisfaction")
    print("=" * 72)
    print(f"\nFile          : {path.name}")
    print(f"Rows          : {len(rows)}")
    print(f"Items detected: {len(item_cols)}  ({', '.join(sorted(item_cols))})")
    if skipped:
        print(f"Unreadable answers skipped: {skipped}")
    if not group_header:
        print("\n  NOTE: no group column found - all rows pooled together.")
        print("  Per-group weighted means are required by the manuscript;")
        print("  add a 'Group' column or pass --group-column.")

    all_group_owms = []

    for group in sorted(responses):
        print("\n" + "-" * 72)
        print(f"GROUP: {group}   (n = {group_counts[group]})")
        print("-" * 72)

        dim_means = {}
        for prefix, dim_name in DIMENSIONS.items():
            codes = sorted(c for c in responses[group] if c.upper().startswith(prefix))
            if not codes:
                continue

            print(f"\n  Dimension {prefix} - {dim_name}")
            print(f"    {'item':<8} {'N':>4} {'WM':>6}   distribution 5/4/3/2/1")
            item_wms = []
            for code in codes:
                vals = responses[group][code]
                n = len(vals)
                wm = sum(vals) / n
                item_wms.append(wm)
                dist = "/".join(str(vals.count(k)) for k in (5, 4, 3, 2, 1))
                print(f"    {code:<8} {n:>4} {wm:>6.2f}   {dist}")

            # Dimension WM = mean of the item WMs (OWM at the item level).
            dim_wm = sum(item_wms) / len(item_wms)
            dim_means[prefix] = dim_wm
            print(f"    {'':<8} {'':>4} {'-' * 6}")
            print(f"    Dimension WM: {dim_wm:.2f}   {interpret(dim_wm)}")

        if dim_means:
            group_owm = sum(dim_means.values()) / len(dim_means)
            all_group_owms.append(group_owm)
            print(f"\n  GROUP OWM: {group_owm:.2f}   {interpret(group_owm)}")

    if all_group_owms:
        overall = sum(all_group_owms) / len(all_group_owms)
        print("\n" + "=" * 72)
        print(f"OVERALL WEIGHTED MEAN (across groups): {overall:.2f}")
        print(f"Interpretation: {interpret(overall)}")
        print("=" * 72)

    # Cronbach's alpha per dimension, computed WITHIN each group.
    #
    # Not pooled across groups: the instrument deliberately has group-specific
    # items (-s for students, -f for staff), so a student never answers a -f
    # item and vice versa. Pooling leaves every dimension ragged, no total
    # score per respondent exists, and alpha is never computable at all.
    # Within a group the item set is complete, which is also the statistically
    # correct unit - alpha measures internal consistency over one coherent set
    # of items answered by one coherent set of respondents.
    print("\nCRONBACH'S ALPHA (internal consistency, threshold 0.70)")
    print("  Computed within each group; the manuscript requires it from the")
    print("  pilot, and small groups will not produce a stable value.")

    for group in sorted(responses):
        n_group = group_counts[group]
        for prefix, dim_name in DIMENSIONS.items():
            items = {c: v for c, v in responses[group].items()
                     if c.upper().startswith(prefix)}
            if len(items) < 2:
                continue
            alpha = cronbach_alpha(items)
            label = f"  {group} / {dim_name}"
            if alpha is None:
                print(f"{label:<34} not computable "
                      f"(unequal answers per item, or n < 2)")
                continue
            if n_group < 10:
                note = f"alpha = {alpha:.3f}   (n = {n_group}, too small to rely on)"
            elif alpha >= 0.70:
                note = f"alpha = {alpha:.3f}   acceptable"
            else:
                note = f"alpha = {alpha:.3f}   BELOW THRESHOLD - revise items"
            print(f"{label:<34} {note}")

    if args.csv_out:
        write_csv(Path(args.csv_out), responses, group_counts)
        print(f"\nPer-item table written to {args.csv_out}")
    print()


def cronbach_alpha(items: dict) -> float | None:
    """
    alpha = k/(k-1) * (1 - sum(item variances) / variance of total scores)

    Needs every respondent to have answered every item in the dimension, so
    that a total score exists per respondent. Returns None when the item
    lists are ragged rather than silently truncating, because truncating
    would quietly compute alpha over a different sample than reported.
    """
    lengths = {len(v) for v in items.values()}
    if len(lengths) != 1:
        return None
    n = lengths.pop()
    k = len(items)
    if n < 2 or k < 2:
        return None

    columns = [items[c] for c in sorted(items)]
    totals = [sum(col[i] for col in columns) for i in range(n)]
    var_total = statistics.variance(totals)
    if var_total == 0:
        return None
    var_items = sum(statistics.variance(col) for col in columns)
    return (k / (k - 1)) * (1 - var_items / var_total)


def write_csv(path: Path, responses: dict, group_counts: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "n_group", "item", "dimension", "n_item",
                    "weighted_mean", "interpretation",
                    "count_5", "count_4", "count_3", "count_2", "count_1"])
        for group in sorted(responses):
            for code in sorted(responses[group]):
                vals = responses[group][code]
                wm = sum(vals) / len(vals)
                w.writerow([
                    group, group_counts[group], code,
                    DIMENSIONS.get(code[0].upper(), "?"), len(vals),
                    f"{wm:.4f}", interpret(wm),
                    *(vals.count(k) for k in (5, 4, 3, 2, 1)),
                ])


def score_cvi(args) -> None:
    """
    Content Validity Index from expert-rater relevance scores.

    Expected CSV: one row per item, one column per rater, ratings 1-4.
    The first column is the item code.

        item,Rater1,Rater2,Rater3
        U1,4,4,3
        U2,4,3,2
    """
    path = Path(args.cvi)
    if not path.exists():
        sys.exit(f"No such file: {path}")

    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        sys.exit("CVI file needs a header row and at least one item row.")

    header, data = rows[0], rows[1:]
    n_raters = len(header) - 1

    print("=" * 60)
    print("CONTENT VALIDITY INDEX")
    print("=" * 60)
    print(f"\nRaters: {n_raters}  ({', '.join(header[1:])})")
    if n_raters < 3:
        print("  WARNING: the manuscript specifies three expert raters.")
    print(f"\n  {'item':<10} {'relevant':>9} {'I-CVI':>7}   verdict")

    icvis = []
    failed = []
    for row in data:
        if not row or not row[0].strip():
            continue
        code = row[0].strip()
        try:
            scores = [int(float(x)) for x in row[1:1 + n_raters] if str(x).strip()]
        except ValueError:
            print(f"  {code:<10} unreadable ratings - skipped")
            continue
        if not scores:
            continue
        # "Relevant" = rated 3 or 4 on the 1-4 relevance scale.
        relevant = sum(1 for s in scores if s >= 3)
        icvi = relevant / len(scores)
        icvis.append(icvi)
        ok = icvi >= 0.78
        if not ok:
            failed.append(code)
        print(f"  {code:<10} {relevant:>4}/{len(scores):<4} {icvi:>7.2f}   "
              f"{'ok' if ok else 'BELOW 0.78 - revise or remove'}")

    if not icvis:
        sys.exit("\nNo usable item rows.")

    scvi = sum(icvis) / len(icvis)
    print(f"\n  S-CVI/Average: {scvi:.3f}   "
          f"{'acceptable' if scvi >= 0.90 else 'BELOW 0.90 - instrument needs revision'}")

    if failed:
        print(f"\n  Items to revise or remove: {', '.join(failed)}")
        print("  Keep the experts' written comments - the manuscript says the")
        print("  signed forms go in the appendices.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--responses", help="CSV of questionnaire responses.")
    parser.add_argument("--cvi", help="CSV of expert relevance ratings (1-4).")
    parser.add_argument("--group-column",
                        help="Header of the respondent-group column "
                             "(default: first header containing 'Group').")
    parser.add_argument("--csv-out", help="Write the per-item table to this CSV.")
    args = parser.parse_args()

    if not args.responses and not args.cvi:
        parser.error("give --responses and/or --cvi")
    if args.cvi:
        score_cvi(args)
    if args.responses:
        score_responses(args)


if __name__ == "__main__":
    main()
