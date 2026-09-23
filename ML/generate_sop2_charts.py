"""
Figures answering Statement of the Problem No. 2:

    What is the level of accuracy of the face-recognition system in
    identifying enrolled students and assigning queue numbers without
    requiring manual input?

Unlike ML/generate_accuracy_charts.py, which plots a controlled photo-set
calibration, everything here is read live from the deployed system's own
records. Nothing is synthetic and nothing is pasted in as a constant:

  student_profiles     the enrolled roster and its stored embeddings, from
                       which the impostor distribution is recomputed.
  face_match_events    every evaluation the matcher has performed, accepted
                       or refused, with its score and its margin over the
                       runner-up.
  recognition_metrics  each queue number actually issued, with how it was
                       linked and how long it took from first sight.

Camera evaluations (track_id set) are what SOP 2 asks about. Face logins
from the mobile client (track_id NULL) use the same matcher but authenticate
account access rather than issue a number, so they are counted separately
and excluded from the operating-point figures.

Output: ML/figures/sop2_*.png
Run:    python ML/generate_sop2_charts.py
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from core.config import FACE_MATCH_THRESHOLD, FACE_MARGIN_THRESHOLD  # noqa: E402
from database.database_handler import get_db_pool  # noqa: E402
from services import face_service  # noqa: E402

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACCEPT = "#1b7f4d"
REFUSE = "#b3261e"
GRID = "#d8d8d8"
INK = "#222222"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def fetch():
    conn = get_db_pool().get_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT id, embedding FROM student_profiles WHERE is_active=TRUE")
    enrolled = cur.fetchall()

    cur.execute(
        "SELECT student_id, track_id, matched_score, margin, accepted, created_at "
        "FROM face_match_events ORDER BY created_at"
    )
    events = cur.fetchall()

    cur.execute(
        "SELECT student_id, queue_number, linked_via, seconds_to_confirm, "
        "seconds_to_link, embed_attempts, embed_successes, match_score, "
        "match_margin FROM recognition_metrics ORDER BY id"
    )
    issued = cur.fetchall()

    cur.close()
    conn.close()
    return enrolled, events, issued


def impostor_similarities(enrolled):
    """Cosine similarity between every pair of different enrolled students."""
    vectors = {}
    for row in enrolled:
        # The column holds the embedding as a JSON string, which is how
        # database_handler.get_all_active_embeddings() reads it too.
        try:
            raw = json.loads(row["embedding"])
        except (TypeError, ValueError):
            continue
        v = np.asarray(face_service.embedding_from_json(raw), dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm > 1e-9:
            vectors[row["id"]] = v / norm
    ids = sorted(vectors)
    return np.array([float(np.dot(vectors[a], vectors[b]))
                     for a, b in combinations(ids, 2)], dtype=np.float64), len(ids)


def queue_zone_events(events):
    """Camera evaluations where the margin is a real comparison.

    When only one student was enrolled there was no runner-up, and the
    matcher defines the second score as -1, so the margin comes out above 1.
    Those early events say nothing about how well two students are told
    apart, which is what the margin measures, so they are reported
    separately rather than plotted.
    """
    cam = [e for e in events
           if e["track_id"] is not None
           and e["matched_score"] is not None and e["margin"] is not None]
    usable = [e for e in cam if e["margin"] < 1.0]
    return cam, usable


def fig_decision_rule(events):
    """Every camera evaluation placed against the two-part accept rule."""
    cam, usable = queue_zone_events(events)
    acc = [(e["matched_score"], e["margin"]) for e in usable if e["accepted"]]
    rej = [(e["matched_score"], e["margin"]) for e in usable if not e["accepted"]]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    ax.axhspan(FACE_MARGIN_THRESHOLD, 2.2, xmin=0, xmax=1,
               color=ACCEPT, alpha=0.05, zorder=0)
    ax.add_patch(plt.Rectangle(
        (FACE_MATCH_THRESHOLD, FACE_MARGIN_THRESHOLD), 1.2, 2.2,
        facecolor=ACCEPT, alpha=0.08, edgecolor="none", zorder=0))

    ax.axvline(FACE_MATCH_THRESHOLD, color=INK, lw=1.1, ls="--")
    ax.axhline(FACE_MARGIN_THRESHOLD, color=INK, lw=1.1, ls="--")

    if rej:
        ax.scatter(*zip(*rej), s=46, marker="x", c=REFUSE, lw=1.6,
                   label=f"refused (n={len(rej)})", zorder=3)
    if acc:
        ax.scatter(*zip(*acc), s=42, marker="o", facecolors="none",
                   edgecolors=ACCEPT, lw=1.4,
                   label=f"accepted (n={len(acc)})", zorder=3)

    ax.text(FACE_MATCH_THRESHOLD + 0.012, 0.02,
            f"score $\\geq$ {FACE_MATCH_THRESHOLD:.2f}",
            fontsize=8.5, color=INK, rotation=90, va="bottom")
    ax.text(0.012, FACE_MARGIN_THRESHOLD + 0.015,
            f"margin $\\geq$ {FACE_MARGIN_THRESHOLD:.2f}",
            fontsize=8.5, color=INK)

    both = acc + rej
    if both:
        ax.set_xlim(-0.02, max(s for s, _ in both) * 1.12)
        ax.set_ylim(-0.02, max(m for _, m in both) * 1.22)

    # The empty band between the two clusters is the result worth seeing.
    if acc and rej:
        lo = max(s for s, _ in rej)
        hi = min(s for s, _ in acc)
        ax.annotate(
            f"no evaluation landed\nbetween {lo:.3f} and {hi:.3f}",
            xy=((lo + hi) / 2, ax.get_ylim()[1] * 0.60),
            ha="center", fontsize=8.5, color="#555555")

    ax.set_xlabel("Similarity to the best-matching enrolled student")
    ax.set_ylabel("Margin over the runner-up")
    ax.set_title("Every queue-zone evaluation against the accept rule",
                 fontsize=11.5, pad=10)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")

    fig.savefig(OUT_DIR / "sop2_01_decision_rule.png")
    plt.close(fig)
    return len(acc), len(rej), len(cam) - len(usable)


def fig_separation(events):
    """Score and margin distributions, accepted against refused."""
    _, usable = queue_zone_events(events)
    a_s = [e["matched_score"] for e in usable if e["accepted"]]
    r_s = [e["matched_score"] for e in usable if not e["accepted"]]
    a_m = [e["margin"] for e in usable if e["accepted"]]
    r_m = [e["margin"] for e in usable if not e["accepted"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for ax, acc_vals, rej_vals, thr, label in (
        (axes[0], a_s, r_s, FACE_MATCH_THRESHOLD, "Similarity score"),
        (axes[1], a_m, r_m, FACE_MARGIN_THRESHOLD, "Margin over runner-up"),
    ):
        bins = np.linspace(0, max(acc_vals + rej_vals + [thr]) * 1.05, 26)
        ax.hist(rej_vals, bins=bins, color=REFUSE, alpha=0.75, label="refused")
        ax.hist(acc_vals, bins=bins, color=ACCEPT, alpha=0.75, label="accepted")
        ax.axvline(thr, color=INK, lw=1.2, ls="--")
        ax.annotate(f"threshold {thr:.2f}", xy=(thr, ax.get_ylim()[1] * 0.92),
                    xytext=(6, 0), textcoords="offset points", fontsize=8.5)
        ax.set_xlabel(label)
        ax.set_ylabel("Evaluations")
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)

        if acc_vals and rej_vals:
            gap = min(acc_vals) - max(rej_vals)
            ax.set_title(f"separation: {gap:+.3f}", fontsize=10)

    fig.suptitle("Accepted and refused evaluations do not overlap",
                 fontsize=11.5, y=1.01)
    fig.savefig(OUT_DIR / "sop2_02_separation.png")
    plt.close(fig)


def fig_impostors(sims, n_students):
    """How close the enrolled students are to one another."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    ax.hist(sims, bins=18, color="#5a7fa6", alpha=0.85,
            label=f"{len(sims)} cross-person pairs")
    ax.axvline(FACE_MATCH_THRESHOLD, color=INK, lw=1.2, ls="--")
    ax.annotate(f"accept threshold {FACE_MATCH_THRESHOLD:.2f}",
                xy=(FACE_MATCH_THRESHOLD, ax.get_ylim()[1] * 0.88),
                xytext=(-8, 0), textcoords="offset points",
                fontsize=8.5, ha="right")

    worst = sims.max()
    ax.axvline(worst, color=REFUSE, lw=1.4)
    ax.annotate(f"closest pair {worst:.4f}",
                xy=(worst, ax.get_ylim()[1] * 0.55),
                xytext=(8, 0), textcoords="offset points",
                fontsize=8.5, color=REFUSE)

    ax.set_xlabel("Cosine similarity between two different enrolled students")
    ax.set_ylabel("Pairs")
    ax.set_title(
        f"Why the margin is required: {n_students} enrolled, "
        "one pair above the score threshold",
        fontsize=11.5, pad=10)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)

    fig.savefig(OUT_DIR / "sop2_03_impostor_separation.png")
    plt.close(fig)
    return worst


def fig_issuance(issued):
    """The two phases between arriving and holding a queue number.

    The phases are consecutive, not nested. seconds_to_confirm runs from the
    first frame a person appears in to the moment their presence is
    confirmed. seconds_to_link then runs from the moment the system was first
    able to act — the later of that confirmation and the student's join
    intent — to the number being issued. Measuring the second phase from
    first sighting instead would charge the system for however long a student
    stood in frame before opening the app.
    """
    rows = [r for r in issued
            if r["seconds_to_link"] is not None
            and r["seconds_to_confirm"] is not None]
    if not rows:
        return None

    confirm = [r["seconds_to_confirm"] for r in rows]
    link = [r["seconds_to_link"] for r in rows]
    manual = sum(1 for r in issued if r["linked_via"] != "face_only")

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    idx = np.arange(1, len(rows) + 1)

    ax.bar(idx, confirm, color="#b9c6d6",
           label="arrival $\\rightarrow$ presence confirmed")
    ax.bar(idx, link, bottom=confirm, color=ACCEPT, alpha=0.9,
           label="ready to issue $\\rightarrow$ number issued")

    mean_link = float(np.mean(link))
    total = [c + l for c, l in zip(confirm, link)]
    for x, (c, l) in zip(idx, zip(confirm, link)):
        ax.annotate(f"{l:.1f}s", xy=(x, c + l), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8)

    ax.set_xticks(idx)
    ax.set_ylim(0, max(total) * 1.18)
    ax.set_xlabel("Queue number issued (in order)")
    ax.set_ylabel("Seconds")
    ax.set_title(
        f"{len(issued)} numbers issued, {len(issued) - manual} without manual input\n"
        f"mean {mean_link:.2f} s from ready to issued",
        fontsize=11.5, pad=10)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.savefig(OUT_DIR / "sop2_04_issuance_time.png")
    plt.close(fig)

    attempts = sum(r["embed_attempts"] or 0 for r in issued)
    successes = sum(r["embed_successes"] or 0 for r in issued)
    return mean_link, manual, attempts, successes


def main():
    enrolled, events, issued = fetch()
    sims, n_students = impostor_similarities(enrolled)

    n_acc, n_rej, n_noruner = fig_decision_rule(events)
    fig_separation(events)
    worst = fig_impostors(sims, n_students)
    issuance = fig_issuance(issued)

    cam = [e for e in events if e["track_id"] is not None]
    mob = [e for e in events if e["track_id"] is None]

    print()
    print("SOP 2 — measured from the deployed system")
    print("-" * 58)
    print(f"  enrolled students              : {n_students}")
    print(f"  evaluations logged             : {len(events)}"
          f"  ({len(cam)} queue-zone, {len(mob)} mobile face login)")
    print(f"  queue-zone accepted / refused  : {n_acc} / {n_rej}")
    if n_noruner:
        print(f"  excluded, no runner-up existed : {n_noruner}"
              "  (only one student enrolled at the time)")
    print(f"  operating point                : score >= {FACE_MATCH_THRESHOLD}, "
          f"margin >= {FACE_MARGIN_THRESHOLD}")
    print()
    print(f"  impostor pairs                 : {len(sims)}")
    print(f"    mean                         : {sims.mean():.4f}")
    print(f"    std                          : {sims.std(ddof=1):.4f}")
    print(f"    max (closest two students)   : {worst:.4f}"
          f"   {'ABOVE' if worst > FACE_MATCH_THRESHOLD else 'below'} the threshold")
    if issuance:
        mean_s, manual, attempts, successes = issuance
        rate = (successes / attempts * 100) if attempts else 0.0
        print()
        print(f"  numbers issued                 : {len(issued)}")
        print(f"    requiring manual input       : {manual}")
        print(f"    mean ready-to-issued         : {mean_s:.2f} s")
        print(f"    face extraction at issuance  : {successes}/{attempts}"
              f"  ({rate:.1f}%)")
    print()
    print(f"  figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
