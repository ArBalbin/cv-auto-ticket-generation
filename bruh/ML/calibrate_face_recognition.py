"""
Calibrates FACE_MATCH_THRESHOLD / FACE_MARGIN_THRESHOLD (Algorithm 4) against
real captured face photos, replacing the current placeholder defaults (0.45 /
0.10) with values backed by measured accept/reject behavior.

This mirrors production matching exactly: for each held-out verification
photo of a person, their canonical enrollment embedding (built the same way
POST /api/students/me/face builds it — averaging enrollment photos) competes
against every OTHER enrolled person's canonical embedding, using the same
cosine-similarity + best-vs-second-best comparison face_service.match_student()
uses at runtime.

Expected data layout — one subfolder per distinct person, at least 2 people:
    calibration_data/faces/<person_id>/enroll_1.jpg
    calibration_data/faces/<person_id>/enroll_2.jpg   (2+ recommended)
    calibration_data/faces/<person_id>/verify_1.jpg   (1+ held-out photos)
    calibration_data/faces/<person_id>/verify_2.jpg
    ...
(Files are split by an "enroll"/"verify" filename prefix; if neither prefix
is used, the first 70% of a person's photos become enrollment and the rest
become verification.)

Photos should vary lighting/angle the same way a real enrollment + later
queue-zone capture would, ideally with the same camera/lighting the
deployment will actually use.

Run: python ML/calibrate_face_recognition.py --data calibration_data/faces
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

import numpy as np
from services import face_service

# .jfif and .webp matter in practice: Windows saves JPEGs from a browser as
# .jfif, and phone/messaging exports are often .webp. Omitting them silently
# dropped an entire enrolled person from an earlier evaluation run — the
# photos were fine, the filter was not.
IMAGE_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".webp"}


def load_person_photos(person_dir: Path) -> dict:
    files = sorted(p for p in person_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    enroll = [p for p in files if p.stem.lower().startswith("enroll")]
    verify = [p for p in files if p.stem.lower().startswith("verify")]
    if not enroll and not verify and files:
        split = max(1, int(len(files) * 0.7))
        enroll, verify = files[:split], files[split:]
    return {"enroll": enroll, "verify": verify}


def build_canonical_embeddings(data_dir: Path) -> dict:
    """Returns {person_id: (canonical_embedding, [verification_embeddings])}."""
    canonical = {}
    for person_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        photos = load_person_photos(person_dir)

        enroll_embeddings = []
        for photo_path in photos["enroll"]:
            emb = face_service.compute_embedding_from_photo(photo_path.read_bytes())
            if emb is not None:
                enroll_embeddings.append(emb)
            else:
                print(f"  [warn] no face detected in {photo_path}")
        if not enroll_embeddings:
            print(f"  [skip] {person_dir.name}: no usable enrollment photos")
            continue

        verify_embeddings = []
        for photo_path in photos["verify"]:
            emb = face_service.compute_embedding_from_photo(photo_path.read_bytes())
            if emb is not None:
                verify_embeddings.append(emb)
            else:
                print(f"  [warn] no face detected in {photo_path}")

        canonical_emb = face_service.build_enrollment_embedding(enroll_embeddings)
        canonical[person_dir.name] = (canonical_emb, verify_embeddings)
    return canonical


def run_trials_from_embeddings(canonical: dict) -> list:
    """Pure logic, no file I/O — takes {person_id: (canonical_emb, [verify_embs])}
    and runs the same best-vs-second-best comparison match_student() does,
    scored against every candidate so thresholds can be re-swept afterward.
    Also records the genuine score (vs. the verification photo's TRUE
    person) and the worst-case impostor score (vs. every OTHER person) for
    each trial — the basis for a proper separation-based recommendation,
    not just "first grid point that happened to work.\""""
    all_candidates = [(pid, emb) for pid, (emb, _) in canonical.items()]
    trials = []
    for true_pid, (true_emb, verify_embeddings) in canonical.items():
        for emb in verify_embeddings:
            scored = sorted(
                ((face_service.cosine_similarity(emb, c_emb), pid) for pid, c_emb in all_candidates),
                key=lambda t: t[0], reverse=True,
            )
            best_score, best_pid = scored[0]
            second_score = scored[1][0] if len(scored) > 1 else -1.0
            genuine_score = float(face_service.cosine_similarity(emb, true_emb))
            impostor_scores = [s for s, pid in scored if pid != true_pid]
            max_impostor_score = max(impostor_scores) if impostor_scores else -1.0
            trials.append({
                "true_pid": true_pid,
                "best_pid": best_pid,
                "best_score": float(best_score),
                "margin": float(best_score - second_score),
                "correct_top_match": best_pid == true_pid,
                "genuine_score": genuine_score,
                "max_impostor_score": float(max_impostor_score),
            })
    return trials


def recommend_from_separation(trials: list) -> dict | None:
    """Statistically grounded recommendation: place the match threshold at
    the midpoint between the worst (highest) impostor score and the worst
    (lowest) genuine score seen in this data — the classic decision-boundary
    placement when the two distributions don't overlap, and one that leaves
    equal safety margin on both sides rather than sitting at the edge of
    whatever happened to work on this specific sample. Falls back to a
    warning if genuine and impostor scores actually overlap in this data —
    no single threshold can then guarantee zero wrong-accepts."""
    genuine_scores = [t["genuine_score"] for t in trials]
    impostor_scores = [t["max_impostor_score"] for t in trials]
    min_genuine, max_genuine = min(genuine_scores), max(genuine_scores)
    min_impostor, max_impostor = min(impostor_scores), max(impostor_scores)

    print(f"Genuine score range  (same person, enrollment vs. verification photo): "
          f"[{min_genuine:.3f}, {max_genuine:.3f}]")
    print(f"Impostor score range (worst different-person match per trial):        "
          f"[{min_impostor:.3f}, {max_impostor:.3f}]")

    if max_impostor >= min_genuine:
        print("\n*** Genuine and impostor score ranges OVERLAP in this data ***")
        print("No single threshold can guarantee zero wrong-accepts on this dataset alone —")
        print("collect more/better-varied photos (more people, more angles/lighting) before")
        print("trusting a threshold from this run.")
        return None

    match_threshold = round((max_impostor + min_genuine) / 2, 3)
    safety_margin = round((min_genuine - max_impostor) / 2, 3)
    print(f"\nDistributions are cleanly separated (gap = {min_genuine - max_impostor:.3f}).")
    print(f"Recommended FACE_MATCH_THRESHOLD = {match_threshold:.3f} "
          f"(midpoint, ±{safety_margin:.3f} safety margin on both sides)")

    margins = [t["margin"] for t in trials if t["correct_top_match"]]
    margin_threshold = round(max(0.0, min(margins) / 2), 3) if margins else 0.0
    print(f"Recommended FACE_MARGIN_THRESHOLD = {margin_threshold:.3f} "
          f"(half the smallest correct-match margin observed, {min(margins):.3f})" if margins else "")

    return {"match_threshold": match_threshold, "margin_threshold": margin_threshold,
            "min_genuine": min_genuine, "max_genuine": max_genuine,
            "min_impostor": min_impostor, "max_impostor": max_impostor}


def evaluate(trials: list, match_threshold: float, margin_threshold: float) -> dict:
    correct_accept = wrong_accept = rejected = 0
    for t in trials:
        accepted = t["best_score"] >= match_threshold and t["margin"] >= margin_threshold
        if accepted and t["correct_top_match"]:
            correct_accept += 1
        elif accepted and not t["correct_top_match"]:
            wrong_accept += 1  # dangerous: accepted, but the WRONG identity
        else:
            rejected += 1      # safe: stays pending, staff resolves manually

    total = len(trials)
    return {
        "total": total,
        "correct_accept": correct_accept,
        "wrong_accept": wrong_accept,
        "rejected": rejected,
        "accuracy": correct_accept / total if total else 0.0,
        "wrong_accept_rate": wrong_accept / total if total else 0.0,
        "reject_rate": rejected / total if total else 0.0,
    }


def sweep(trials: list) -> dict:
    print(f"\n{'match_thr':>10} {'margin_thr':>11} {'correct':>8} {'wrong-accept':>13} {'rejected':>9}")
    print("-" * 58)
    best = None
    grid = []
    for match_t in np.arange(0.20, 0.71, 0.05):
        for margin_t in (0.0, 0.05, 0.10, 0.15, 0.20):
            match_t = round(float(match_t), 2)
            result = evaluate(trials, match_t, margin_t)
            grid.append({"match_threshold": match_t, "margin_threshold": margin_t, **result})
            # Never recommend a threshold with ANY wrong-accept in this data —
            # matches the system's "prefer reject over wrong guess" design.
            if result["wrong_accept"] == 0:
                if best is None or result["accuracy"] > best[2]["accuracy"]:
                    best = (match_t, margin_t, result)
            print(f"{match_t:>10.2f} {margin_t:>11.2f} {result['correct_accept']:>8} "
                  f"{result['wrong_accept']:>13} {result['rejected']:>9}")

    print()
    grid_pick = None
    if best:
        match_t, margin_t, result = best
        print("Grid-sweep pick (secondary sanity check only — see the separation-based")
        print("recommendation above for the one to actually use):")
        print(f"  FACE_MATCH_THRESHOLD={match_t:.2f}  FACE_MARGIN_THRESHOLD={margin_t:.2f}")
        print(f"  -> {result['correct_accept']}/{result['total']} correctly linked, "
              f"{result['rejected']} escalated to staff, 0 wrong identities")
        grid_pick = {"match_threshold": match_t, "margin_threshold": margin_t, **result}
    else:
        print("No grid point avoided all wrong-accepts either — this data does not "
              "cleanly support any threshold yet.")

    return {"grid": grid, "grid_pick": grid_pick}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="calibration_data/faces")
    parser.add_argument("--out", default=None, help="optional path to dump full results as JSON")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.is_dir():
        print(f"No such directory: {data_dir}\n")
        print(__doc__)
        return

    print(f"Building canonical enrollment embeddings from {data_dir} ...")
    canonical = build_canonical_embeddings(data_dir)
    print(f"Enrolled {len(canonical)} people.")
    if len(canonical) < 2:
        print("Need at least 2 people with usable photos to calibrate a MARGIN threshold.")
        return

    print("Running verification trials (mirrors production match_student() exactly)...")
    trials = run_trials_from_embeddings(canonical)
    print(f"Ran {len(trials)} verification trials.\n")
    if not trials:
        print("No usable verification photos found (need files named verify_*.jpg, or "
              "enough photos per person for the automatic 70/30 split).")
        return

    print("=" * 58)
    print("PRIMARY RECOMMENDATION (separation-based)")
    print("=" * 58)
    recommendation = recommend_from_separation(trials)

    print()
    print("=" * 58)
    print("FULL GRID (for context / the thesis evaluation chapter)")
    print("=" * 58)
    grid_results = sweep(trials)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(
            {"trials": trials, "recommendation": recommendation, **grid_results},
            indent=2,
        ))
        print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
