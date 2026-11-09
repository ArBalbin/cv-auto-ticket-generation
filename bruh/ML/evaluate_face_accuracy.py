"""
Biometric accuracy evaluation for the face-recognition layer (SOP #2).

This is the evaluation a panel expects to see for a biometric system. It is
deliberately separate from ML/calibrate_face_recognition.py, which answers a
narrower question ("which thresholds should we deploy?"). This one answers
"how accurate is the deployed system, in the standard vocabulary of the
field?" and reports three things that calibration does not:

  1. FAR / FRR / EER / ROC-AUC over the FULL impostor set.
     Calibration keeps only the worst impostor per trial (43 numbers). That
     is the right input for choosing a safe threshold, but it throws away
     93% of the comparisons and cannot produce a real False Accept Rate.
     Here every probe is scored against every gallery identity, so FAR is
     computed over all ~N*(N-1) impostor pairs, which is what the metric
     actually means.

  2. Open-set rejection (leave-one-person-out).
     The single most important safety question for QueuEx is NOT "can it
     recognise an enrolled student?" but "does an UNENROLLED person walking
     past the camera get handed someone else's queue number?" Closed-set
     accuracy cannot answer that, because it never presents an unknown face.
     This test removes each person from the gallery entirely and re-probes
     with their photos: every acceptance is a false identification of a
     stranger.

  3. Rank-1 identification accuracy, reported separately from verification.
     QueuEx performs 1:N identification (search the roster), not 1:1
     verification (confirm a claim). They are different tasks with different
     error rates and should not be quoted interchangeably.

Data layout is the same as calibrate_face_recognition.py:
    calibration_data/faces/<person>/enroll_*.jpg   (gallery)
    calibration_data/faces/<person>/verify_*.jpg   (probes)
(or an automatic 70/30 split when the prefixes are absent).

LIMITATION, and it must be stated in the thesis: these are enrollment-style
photos, captured close-up and cooperatively. Production probes come from a
YOLO person-crop at queue-zone distance, which is a harder input. Treat these
numbers as an upper bound on live performance, and report the live figures
from ML/report_recognition_metrics.py alongside them once beta data exists.

Run:
    python ML/evaluate_face_accuracy.py
    python ML/evaluate_face_accuracy.py --no-cache      # recompute embeddings
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from services import face_service  # noqa: E402

from calibrate_face_recognition import load_person_photos  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT_DIR = Path(__file__).parent / "figures"
REPORT_PATH = Path(__file__).parent / "face_accuracy_report.json"
CACHE_PATH = ROOT / "calibration_data" / "embedding_cache.npz"

# The operating point actually deployed (app/core/config.py).
MATCH_THRESHOLD = 0.30
MARGIN_THRESHOLD = 0.15


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

def build_embeddings(data_dir: Path, use_cache: bool = True) -> dict:
    """
    Returns {person: {"gallery": canonical_emb, "probes": [emb, ...]}}.

    Embedding extraction costs ~260ms per photo, so a 120-photo dataset takes
    ~30s. That is fine once, but this script is meant to be re-run while
    tuning the report, so results are cached by file path + mtime. Any edit
    to the photo set invalidates its own entry only.
    """
    cache = {}
    if use_cache and CACHE_PATH.exists():
        try:
            with np.load(CACHE_PATH, allow_pickle=True) as data:
                cache = {k: data[k] for k in data.files}
        except Exception as exc:
            print(f"  (cache unreadable, recomputing: {exc})")

    fresh_cache = {}
    people = {}

    for person_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        photos = load_person_photos(person_dir)

        def embed(paths: list[Path]) -> list[np.ndarray]:
            out = []
            for path in paths:
                key = f"{path.relative_to(data_dir)}|{int(path.stat().st_mtime)}"
                if key in cache:
                    emb = cache[key]
                else:
                    emb = face_service.compute_embedding_from_photo(path.read_bytes())
                    if emb is None:
                        print(f"  [warn] no face detected in {path.name} ({person_dir.name})")
                        continue
                    emb = np.asarray(emb, dtype=np.float32)
                fresh_cache[key] = emb
                out.append(emb)
            return out

        gallery_embs = embed(photos["enroll"])
        probe_embs = embed(photos["verify"])

        if not gallery_embs:
            print(f"  [skip] {person_dir.name}: no usable gallery photos")
            continue
        if not probe_embs:
            print(f"  [skip] {person_dir.name}: no held-out probe photos")
            continue

        people[person_dir.name] = {
            "gallery": face_service.build_enrollment_embedding(gallery_embs),
            "probes": probe_embs,
        }

    if use_cache:
        try:
            np.savez_compressed(CACHE_PATH, **fresh_cache)
        except Exception as exc:
            print(f"  (could not write cache: {exc})")

    return people


# --------------------------------------------------------------------------
# Score collection
# --------------------------------------------------------------------------

def collect_scores(people: dict) -> tuple[list[float], list[float]]:
    """
    Every probe scored against every gallery identity, split into genuine
    (probe vs. own identity) and impostor (probe vs. everyone else).

    This is the full comparison matrix, which is what FAR and FRR are defined
    over. With P probes and N identities it yields P genuine and P*(N-1)
    impostor scores.
    """
    genuine, impostor = [], []
    for person, data in people.items():
        for probe in data["probes"]:
            for other, other_data in people.items():
                score = face_service.cosine_similarity(probe, other_data["gallery"])
                (genuine if other == person else impostor).append(score)
    return genuine, impostor


# --------------------------------------------------------------------------
# Verification metrics (1:1)
# --------------------------------------------------------------------------

def far_frr_curve(genuine: list[float], impostor: list[float],
                  thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    FAR(t) = fraction of impostor pairs scoring >= t   (wrongly accepted)
    FRR(t) = fraction of genuine  pairs scoring <  t   (wrongly rejected)
    """
    g = np.asarray(genuine)
    i = np.asarray(impostor)
    far = np.array([(i >= t).mean() for t in thresholds])
    frr = np.array([(g < t).mean() for t in thresholds])
    return far, frr


def equal_error_rate(far: np.ndarray, frr: np.ndarray,
                     thresholds: np.ndarray) -> tuple[float, float]:
    """
    EER is where FAR and FRR cross — the standard single-number summary of a
    biometric system's separability, independent of the operating point the
    deployment happens to choose. Linear interpolation between the two grid
    points that bracket the crossing.
    """
    diff = far - frr
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        idx = int(np.argmin(np.abs(diff)))
        return float((far[idx] + frr[idx]) / 2), float(thresholds[idx])

    k = int(sign_change[0])
    d0, d1 = diff[k], diff[k + 1]
    frac = 0.0 if d1 == d0 else d0 / (d0 - d1)
    eer_threshold = thresholds[k] + frac * (thresholds[k + 1] - thresholds[k])
    eer = far[k] + frac * (far[k + 1] - far[k])
    return float(eer), float(eer_threshold)


def roc_auc(genuine: list[float], impostor: list[float]) -> float:
    """
    AUC via the Mann-Whitney U identity: the probability that a randomly
    chosen genuine pair outscores a randomly chosen impostor pair. Computed
    by rank rather than by integrating the curve, so it is exact and needs no
    threshold grid. Ties count as half, per the standard definition.
    """
    g = np.asarray(genuine)
    i = np.asarray(impostor)
    combined = np.concatenate([g, i])
    order = combined.argsort()
    ranks = np.empty(len(combined), dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1)

    # Average ranks within ties so tied scores contribute 0.5 each.
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]

    n_g, n_i = len(g), len(i)
    rank_sum_genuine = ranks[:n_g].sum()
    return float((rank_sum_genuine - n_g * (n_g + 1) / 2) / (n_g * n_i))


def d_prime(genuine: list[float], impostor: list[float]) -> float:
    """
    Separation in standard-deviation units. Unlike the raw score gap, d' is
    scale-free, so it can be compared against published face-recognition
    results. d' > 3 is generally considered strong separation.
    """
    mg, mi = statistics.mean(genuine), statistics.mean(impostor)
    vg = statistics.variance(genuine) if len(genuine) > 1 else 0.0
    vi = statistics.variance(impostor) if len(impostor) > 1 else 0.0
    pooled = ((vg + vi) / 2) ** 0.5
    return float((mg - mi) / pooled) if pooled > 1e-9 else float("inf")


# --------------------------------------------------------------------------
# Identification metrics (1:N) — what QueuEx actually does
# --------------------------------------------------------------------------

def closed_set_identification(people: dict) -> dict:
    """
    Every probe searched against the full gallery, using the deployed
    decision rule verbatim (best score >= threshold AND margin over
    runner-up >= margin threshold).

    Three outcomes, and they are not equally bad:
      correct  — linked to the right student
      wrong    — linked to the WRONG student (the unsafe failure)
      refused  — no confident match, escalated to staff (the safe failure)
    """
    gallery = [(person, data["gallery"]) for person, data in people.items()]
    correct = wrong = refused = 0
    rank1_hits = 0
    total = 0
    refused_detail = []

    for person, data in people.items():
        for idx, probe in enumerate(data["probes"]):
            total += 1
            scored = sorted(
                ((face_service.cosine_similarity(probe, emb), pid) for pid, emb in gallery),
                key=lambda t: t[0], reverse=True,
            )
            best_score, best_pid = scored[0]
            second = scored[1][0] if len(scored) > 1 else -1.0
            margin = best_score - second

            if best_pid == person:
                rank1_hits += 1

            accepted = best_score >= MATCH_THRESHOLD and margin >= MARGIN_THRESHOLD
            if accepted and best_pid == person:
                correct += 1
            elif accepted:
                wrong += 1
            else:
                refused += 1
                refused_detail.append({
                    "person": person, "probe": idx,
                    "score": round(best_score, 3), "margin": round(margin, 3),
                    "reason": "below score" if best_score < MATCH_THRESHOLD else "below margin",
                })

    return {
        "total": total,
        "rank1_accuracy": rank1_hits / total if total else 0.0,
        "correct": correct, "wrong": wrong, "refused": refused,
        "refused_detail": refused_detail,
    }


def open_set_rejection(people: dict) -> dict:
    """
    THE SAFETY TEST. Leave-one-person-out: remove a person from the gallery
    entirely, then probe with their photos. They are now a stranger — someone
    who never enrolled, walking through the queue zone.

    The system MUST refuse. Any acceptance means an unenrolled person was
    handed an enrolled student's identity, which in QueuEx means someone
    else's queue number. This is the failure mode that would invalidate the
    whole approach, and closed-set accuracy is blind to it.
    """
    names = list(people)
    accepted_as = []
    saved_by_margin = []
    refused = 0
    total = 0
    top_scores = []

    for held_out in names:
        gallery = [(p, people[p]["gallery"]) for p in names if p != held_out]
        if len(gallery) < 2:
            continue
        for probe in people[held_out]["probes"]:
            total += 1
            scored = sorted(
                ((face_service.cosine_similarity(probe, emb), pid) for pid, emb in gallery),
                key=lambda t: t[0], reverse=True,
            )
            best_score, best_pid = scored[0]
            second = scored[1][0] if len(scored) > 1 else -1.0
            margin = best_score - second
            top_scores.append(best_score)

            passes_score = best_score >= MATCH_THRESHOLD
            passes_margin = margin >= MARGIN_THRESHOLD

            if passes_score and passes_margin:
                accepted_as.append({
                    "stranger": held_out, "mistaken_for": best_pid,
                    "score": round(best_score, 3), "margin": round(margin, 3),
                })
            else:
                refused += 1
                # The case that justifies the margin rule's existence: a
                # stranger whose similarity to some enrolled student cleared
                # the score threshold outright. The score rule ALONE would
                # have handed them that student's queue number; only the
                # margin check stopped it. Counted separately because it is
                # the difference between "safe by design" and "safe by luck".
                if passes_score and not passes_margin:
                    saved_by_margin.append({
                        "stranger": held_out, "would_be": best_pid,
                        "score": round(best_score, 3), "margin": round(margin, 3),
                    })

    return {
        "total": total,
        "correctly_refused": refused,
        "false_accepts": len(accepted_as),
        "false_accept_rate": len(accepted_as) / total if total else 0.0,
        "detail": accepted_as,
        "max_stranger_score": max(top_scores) if top_scores else 0.0,
        "saved_by_margin": saved_by_margin,
        "score_rule_alone_false_accepts": len(accepted_as) + len(saved_by_margin),
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def make_figures(genuine, impostor, thresholds, far, frr, eer, eer_t, auc) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 10, "axes.grid": True,
        "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
    })

    # --- FAR / FRR crossing -------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, far * 100, color="#d1495b", linewidth=2,
            label="FAR — stranger accepted")
    ax.plot(thresholds, frr * 100, color="#2a9d8f", linewidth=2,
            label="FRR — real student refused")
    ax.axvline(MATCH_THRESHOLD, color="#1d3557", linestyle="--", linewidth=2,
               label=f"Deployed = {MATCH_THRESHOLD}")
    ax.plot([eer_t], [eer * 100], "o", color="#e9c46a", markersize=9,
            markeredgecolor="#1d3557", zorder=5,
            label=f"EER = {eer * 100:.2f}% @ {eer_t:.3f}")
    ax.set_xlabel("Match threshold (cosine similarity)")
    ax.set_ylabel("Error rate (%)")
    ax.set_title("False accept vs. false reject")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_far_frr.png")
    plt.close(fig)

    # --- ROC ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    tar = 1 - frr
    ax.plot(far, tar, color="#457b9d", linewidth=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="#adb5bd", linewidth=1, label="Chance")
    ax.set_xlabel("False accept rate")
    ax.set_ylabel("True accept rate")
    ax.set_title("ROC curve")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_roc_curve.png")
    plt.close(fig)

    # --- Full score distributions -------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-0.2, 1.0, 60)
    ax.hist(impostor, bins=bins, alpha=0.75, color="#d1495b", density=True,
            label=f"Impostor pairs (n={len(impostor)})")
    ax.hist(genuine, bins=bins, alpha=0.75, color="#2a9d8f", density=True,
            label=f"Genuine pairs (n={len(genuine)})")
    ax.axvline(MATCH_THRESHOLD, color="#1d3557", linestyle="--", linewidth=2,
               label=f"Deployed = {MATCH_THRESHOLD}")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title("Full comparison matrix — every probe vs. every identity")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_full_score_distribution.png")
    plt.close(fig)


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="calibration_data/faces")
    parser.add_argument("--no-cache", action="store_true",
                        help="Recompute every embedding from the photos.")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.is_dir():
        sys.exit(f"No such directory: {data_dir}")

    OUT_DIR.mkdir(exist_ok=True)

    print(f"Building embeddings from {data_dir} ...")
    people = build_embeddings(data_dir, use_cache=not args.no_cache)
    if len(people) < 3:
        sys.exit(f"Need at least 3 usable people; got {len(people)}.")

    n_probes = sum(len(d["probes"]) for d in people.values())
    print(f"  {len(people)} identities, {n_probes} probe photos\n")

    genuine, impostor = collect_scores(people)
    thresholds = np.linspace(-0.1, 1.0, 1101)
    far, frr = far_frr_curve(genuine, impostor, thresholds)
    eer, eer_t = equal_error_rate(far, frr, thresholds)
    auc = roc_auc(genuine, impostor)
    dp = d_prime(genuine, impostor)

    closed = closed_set_identification(people)
    openset = open_set_rejection(people)

    # Error rates at the point actually deployed.
    far_at = float((np.asarray(impostor) >= MATCH_THRESHOLD).mean())
    frr_at = float((np.asarray(genuine) < MATCH_THRESHOLD).mean())

    print("=" * 66)
    print("QueuEx face-recognition accuracy report")
    print("=" * 66)

    print(f"\nDATASET")
    print(f"  identities            : {len(people)}")
    print(f"  probe photos          : {n_probes}")
    print(f"  genuine comparisons   : {len(genuine)}")
    print(f"  impostor comparisons  : {len(impostor)}")

    print(f"\n1. VERIFICATION (1:1) — threshold-independent quality")
    # 6 decimals deliberately: at 4 dp this prints 1.0000, which reads as
    # perfect separation and is an overclaim — there IS one inverted pair.
    print(f"  ROC AUC               : {auc:.6f}")
    print(f"  Equal Error Rate      : {eer * 100:.2f}%  (at threshold {eer_t:.3f})")
    print(f"  d-prime (separation)  : {dp:.2f}")
    print(f"  genuine  mean +/- sd  : {statistics.mean(genuine):.3f} +/- "
          f"{statistics.stdev(genuine):.3f}")
    print(f"  impostor mean +/- sd  : {statistics.mean(impostor):.3f} +/- "
          f"{statistics.stdev(impostor):.3f}")

    print(f"\n2. AT THE DEPLOYED OPERATING POINT "
          f"(threshold {MATCH_THRESHOLD}, margin {MARGIN_THRESHOLD})")
    print(f"  FAR, score rule only  : {far_at * 100:.2f}%   "
          f"({int(far_at * len(impostor))}/{len(impostor)} impostor pairs)")
    print(f"  FRR, score rule only  : {frr_at * 100:.2f}%")
    print( "  NOTE: these two are the SCORE rule in isolation. The deployed")
    print( "  system also requires the margin check, so they are an upper")
    print( "  bound on its error, not its error. Sections 3 and 4 measure the")
    print( "  full rule; quote those as the system's accuracy.")

    print(f"\n3. IDENTIFICATION (1:N) — what the system actually performs")
    t = closed["total"]
    print(f"  Rank-1 accuracy       : {closed['rank1_accuracy'] * 100:.1f}%  "
          f"({int(closed['rank1_accuracy'] * t)}/{t} ranked correctly)")
    print(f"  Correctly linked      : {closed['correct']}/{t} "
          f"({closed['correct'] / t * 100:.1f}%)")
    print(f"  WRONG identity        : {closed['wrong']}/{t} "
          f"({closed['wrong'] / t * 100:.1f}%)   <- unsafe failure")
    print(f"  Refused to staff      : {closed['refused']}/{t} "
          f"({closed['refused'] / t * 100:.1f}%)   <- safe failure")
    for r in closed["refused_detail"]:
        print(f"      refused: {r['person']} probe {r['probe']} "
              f"(score {r['score']}, margin {r['margin']} — {r['reason']})")

    print(f"\n4. OPEN-SET REJECTION — unenrolled stranger at the camera")
    ot = openset["total"]
    print(f"  stranger probes       : {ot}")
    print(f"  correctly refused     : {openset['correctly_refused']}/{ot} "
          f"({openset['correctly_refused'] / ot * 100:.1f}%)")
    print(f"  FALSE IDENTIFICATIONS : {openset['false_accepts']}/{ot} "
          f"({openset['false_accept_rate'] * 100:.2f}%)")
    print(f"  highest stranger score: {openset['max_stranger_score']:.3f} "
          f"(threshold is {MATCH_THRESHOLD})")
    for d in openset["detail"]:
        print(f"      {d['stranger']} accepted as {d['mistaken_for']} "
              f"(score {d['score']}, margin {d['margin']})")

    saved = openset["saved_by_margin"]
    if saved:
        print(f"\n  MARGIN RULE CONTRIBUTION — read this before claiming the")
        print(f"  score threshold is safe on its own:")
        print(f"    stranger probes that CLEARED the {MATCH_THRESHOLD} score "
              f"threshold : {len(saved)}")
        print(f"    caught by the margin rule instead                    : "
              f"{len(saved)} (all of them)")
        for d in saved:
            print(f"      {d['stranger']} scored {d['score']} against "
                  f"{d['would_be']} (margin {d['margin']} < {MARGIN_THRESHOLD})")
        print(f"    Score rule alone would have produced "
              f"{openset['score_rule_alone_false_accepts']} false identification(s).")
        print(f"    The two-part rule produces {openset['false_accepts']}.")

    make_figures(genuine, impostor, thresholds, far, frr, eer, eer_t, auc)

    report = {
        "dataset": {
            "identities": len(people), "probe_photos": n_probes,
            "genuine_comparisons": len(genuine),
            "impostor_comparisons": len(impostor),
            "people": sorted(people),
        },
        "verification": {
            "roc_auc": auc, "eer": eer, "eer_threshold": eer_t, "d_prime": dp,
            "genuine_mean": statistics.mean(genuine),
            "genuine_sd": statistics.stdev(genuine),
            "impostor_mean": statistics.mean(impostor),
            "impostor_sd": statistics.stdev(impostor),
        },
        "operating_point": {
            "match_threshold": MATCH_THRESHOLD,
            "margin_threshold": MARGIN_THRESHOLD,
            "far": far_at, "frr": frr_at,
        },
        "identification_closed_set": closed,
        "identification_open_set": openset,
        "limitation": (
            "Probes are cooperative enrollment-style photos, not YOLO crops at "
            "queue-zone distance. These figures are an upper bound on live "
            "performance; pair them with ML/report_recognition_metrics.py."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"\nFigures : {OUT_DIR}  (06_far_frr, 07_roc_curve, 08_full_score_distribution)")
    print(f"Report  : {REPORT_PATH}")
    print()


if __name__ == "__main__":
    main()
