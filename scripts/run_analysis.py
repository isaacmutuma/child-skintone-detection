"""Phase 1 analysis: per detector x dataset metrics + primary endpoint with Holm.

Reads per-image CSVs (PER_IMAGE_DIR, never committed) and writes aggregate JSON only:
  results/phase1/metrics_<detector>_<dataset>.json   all metrics, CIs, n per group
  results/phase1/primary_endpoint.json               White - Black TPR gaps, Holm per dataset

  python scripts/run_analysis.py                    # tag 'full' (D5 uses 'full_sub')
  python scripts/run_analysis.py --tag dev --out-dir <scratch>   # pipeline test only
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from configs import config as C  # noqa: E402
from src.analysis import metrics as M  # noqa: E402
from src.analysis.regression import confounder_logit  # noqa: E402
from src.analysis.thresholds import equalised_thresholds  # noqa: E402
from src.provenance import write_result  # noqa: E402

# Holm families (PREREGISTRATION.md): per dataset, each CLIP model its own test
FAMILIES = {
    "utkface": ["mivolo_face", "fairface_clf", "clip_vitb32", "clip_vitl14", "vlm"],
    "fairface_val": ["mivolo_face", "clip_vitb32", "clip_vitl14", "vlm"],
}
EXPECTED_N = {"utkface": 23705, "fairface_val": 10954}


def load(per_dir, det, ds, tag):
    """Per-image CSV for one run (D5 full runs use the 'full_sub' tag)."""
    t = f"{tag}_sub" if det == "vlm" and tag == "full" else tag
    f = Path(per_dir) / f"per_image__{det}__{ds}__{t}.csv"
    return (pd.read_csv(f), t) if f.exists() else (None, t)


def check_consistency(df, det, ds, tag):
    """Decision must equal score > 0; no-score rows must be unflagged; n must match."""
    issues = []
    sc, fl = df.score.astype(float), df.is_child.astype(bool)
    bad = (sc.notna() & ((sc > 0) != fl)).sum() + (sc.isna() & fl).sum()
    if bad:
        issues.append(f"{bad} rows where is_child != (score > 0)")
    if tag == "full" and len(df) != EXPECTED_N[ds]:
        issues.append(f"n={len(df)} != expected {EXPECTED_N[ds]}")
    if df.id.duplicated().any():
        issues.append(f"{df.id.duplicated().sum()} duplicated ids")
    if issues:
        raise SystemExit(f"{det}/{ds}: " + "; ".join(issues))


def run_json(det, ds, tag):
    """Provenance of the detector run (commit, weights, runtime) to copy into the metrics."""
    p = C.RESULTS_DIR / "phase1" / "runs" / f"{det}__{ds}__{tag}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {"run_commit": d["git"]["commit"], "run_dirty": d["git"]["dirty"],
            "preregistration_commit": d.get("preregistration_commit"),
            "weights": d.get("weights"), "seconds": d.get("seconds")}


def pairwise_vs_best(df, ds, w, groups, B, rng):
    """Secondary: stratified TPR gap of every group vs the best group (bootstrap CI)."""
    df = df.copy()
    df["astratum"] = M.analysis_stratum(df, ds)
    fl = df.is_child.astype(bool).astype(float)
    est, reps = {}, {}
    for g in groups:
        m = df[(df.group == g) & (df.role == "minor")]
        est[g] = M.stratified_rate(fl.loc[m.index].values, m.astratum.values, w)
        reps[g] = M.boot_stratified(fl.loc[m.index].values, m.astratum.values, w, B, rng)
    best = max((g for g in est if not np.isnan(est[g])), key=lambda g: est[g])
    return {"best_group": best,
            "gaps": {g: {"gap": est[best] - est[g], "ci": M.ci(reps[best] - reps[g])}
                     for g in groups if g != best}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--per-image-dir", default=str(C.PER_IMAGE_DIR))
    ap.add_argument("--out-dir", default=str(C.RESULTS_DIR / "phase1"))
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    B = P = C.N_BOOTSTRAP
    lightest, darkest = C.PRIMARY_CONTRAST
    primary = {}

    for ds, family in FAMILIES.items():
        w = M.reference_weights(ds)
        tested = C.UTKFACE_TESTED_GROUPS if ds == "utkface" else C.FAIRFACE_RACES
        primary[ds] = {"family": family, "family_size": len(family), "reference_weights": w,
                       "detectors": {}}
        for det in family:
            df, tag = load(a.per_image_dir, det, ds, a.tag)
            if df is None:
                print(f"[{ds}] {det}: no per-image file for tag {a.tag} -> missing (counts as p=1 in Holm)")
                continue
            check_consistency(df, det, ds, tag)
            rng = np.random.default_rng(C.SEED)
            groups = sorted(df.group.unique())
            res = {
                "detector": det, "dataset": ds, "tag": tag, "n_images": int(len(df)),
                "reference_weights": w, "strata": M.strata_for(ds),
                "run": run_json(det, ds, tag),
                "groups": M.group_metrics(df, ds, w, B, rng),
                "primary_contrast": M.contrast(df, ds, w, lightest, darkest, B, P, rng),
                "pairwise_vs_best": pairwise_vs_best(df, ds, w, [g for g in groups if g in tested], B, rng),
                "fpr_equal_across_groups": M.fpr_independence(df[df.group.isin(tested)]),
                "confounder_logit": confounder_logit(df[df.group.isin(tested)]),
                "equalised_thresholds": equalised_thresholds(df[df.group.isin(tested)]),
            }
            write_result(out_dir / f"metrics_{det}_{ds}.json", res, C.as_dict(), C.REPO_DIR)
            pc = res["primary_contrast"]["tpr_gap"]
            primary[ds]["detectors"][det] = {"tag": tag, "tpr_gap": pc["gap"], "ci": pc["ci"],
                                             "p_perm": pc["p_perm"],
                                             "n_minor": {g: res["groups"][g]["n_minor"]
                                                         for g in (lightest, darkest)}}
            print(f"[{ds}] {det:13s} gap={pc['gap']:+.4f} CI={pc['ci']} p={pc['p_perm']:.4f}")

        dets = primary[ds]["detectors"]
        adj = M.holm({d: dets[d]["p_perm"] for d in dets}, family_size=len(family))
        n_support = 0
        for d in dets:
            dets[d]["p_holm"] = adj[d]
            lo = dets[d]["ci"][0]
            dets[d]["supports_h1"] = bool(dets[d]["tpr_gap"] > 0 and lo is not None and lo > 0
                                          and adj[d] < C.ALPHA)
            n_support += dets[d]["supports_h1"]
        primary[ds]["missing_detectors"] = [d for d in family if d not in dets]
        primary[ds]["h1_count"] = f"{n_support}/{len(family)}"
        primary[ds]["h1_supported"] = n_support >= len(family) / 2
        if primary[ds]["missing_detectors"]:
            primary[ds]["note"] = "incomplete family: missing detectors counted as p=1 (provisional)"

    write_result(out_dir / "primary_endpoint.json",
                 {"tag": a.tag, "contrast": f"{lightest} - {darkest}", "datasets": primary},
                 C.as_dict(), C.REPO_DIR)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
