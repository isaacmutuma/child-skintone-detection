"""Phase 1 figures (charts only, never images): dot + 95% CI per group, one series per
detector, small multiples by dataset. Reads results/phase1/metrics_*.json.

  figures/phase1_tpr_by_group.png         age-stratified TPR (H1)
  figures/phase1_fdr_by_group.png         face-detection failure among minors (H2a)
  figures/phase1_age_error_by_group.png   signed age error, detected minors (H2b)

Palette: validated categorical slots 1-5 (dataviz skill reference palette, light mode,
all checks pass; 3 slots < 3:1 vs surface -> marker shapes + legend + report table).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from configs import config as C  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
DETECTORS = [  # fixed order -> fixed colour + marker (never re-assigned by rank)
    ("mivolo_face", "D1 MiVOLO v2", "#2a78d6", "o"),
    ("fairface_clf", "D3 FairFace ViT", "#eb6834", "s"),
    ("clip_vitb32", "D4 CLIP B/32", "#1baf7a", "^"),
    ("clip_vitl14", "D4 CLIP L/14", "#eda100", "D"),
    ("vlm", "D5 Qwen2-VL-2B", "#e87ba4", "v"),
]
DATASETS = [("utkface", "UTKFace (strata 0–9 / 10–17)", C.UTKFACE_TESTED_GROUPS),
            ("fairface_val", "FairFace val (strata 0–2 / 3–9)", C.FAIRFACE_RACES)]
FIGS = {
    "tpr": ("tpr_strat", "tpr_strat_ci", "Age-stratified TPR (minors flagged as children)",
            "phase1_tpr_by_group.png", None),
    "fdr": ("face_fail_minor_strat", "face_fail_minor_strat_ci",
            "Face-detection failure rate among minors (age-stratified)",
            "phase1_fdr_by_group.png", None),
    "age": ("signed_age_err_strat", "signed_age_err_strat_ci",
            "Signed age error, detected minors (predicted − true, years)",
            "phase1_age_error_by_group.png", 0.0),
}


def load(metrics_dir):
    out = {}
    for det, *_ in DETECTORS:
        for ds, *_ in DATASETS:
            p = Path(metrics_dir) / f"metrics_{det}_{ds}.json"
            if p.exists():
                out[(det, ds)] = json.loads(p.read_text())
    return out


def plot(metrics, key, out_dir):
    est_k, ci_k, title, fname, ref = FIGS[key]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [4, 7]})
    drawn = set()
    for ax, (ds, ds_title, groups) in zip(axes, DATASETS):
        ax.set_facecolor(SURFACE)
        dets = [d for d in DETECTORS if (d[0], ds) in metrics and
                any(metrics[(d[0], ds)]["groups"].get(g, {}).get(est_k) is not None for g in groups)]
        k = len(dets)
        for j, (det, label, color, marker) in enumerate(dets):
            m = metrics[(det, ds)]["groups"]
            off = (j - (k - 1) / 2) * 0.14
            for i, g in enumerate(groups):
                r = m.get(g, {})
                if r.get(est_k) is None:
                    continue
                lo, hi = r.get(ci_k) or [None, None]
                y = i + off
                if lo is not None:
                    ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
                ax.scatter([r[est_k]], [y], s=46, color=color, marker=marker, zorder=3,
                           edgecolors=SURFACE, linewidths=1.5,
                           label=label if label not in drawn else None)
                drawn.add(label)
        if ref is not None:
            ax.axvline(ref, color=INK2, lw=1, zorder=1)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([f"{g}" for g in groups], color=INK, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(ds_title, color=INK, fontsize=10, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=9, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    handles, labels = [], []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h); labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=9, labelcolor=INK)
    ci_note = ("stratified Wilson score 95% CI" if key == "fdr"
               else "bootstrap 95% CI (2,000 resamples)")
    fig.text(0.01, 0.005, f"Dots: point estimate; lines: {ci_note}. "
             "Groups are dataset-provided perceived-race labels.", color=INK2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    out = Path(out_dir) / fname
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", default=str(C.RESULTS_DIR / "phase1"))
    ap.add_argument("--out-dir", default=str(C.FIGURES_DIR))
    a = ap.parse_args()
    metrics = load(a.metrics_dir)
    if not metrics:
        raise SystemExit(f"no metrics_*.json in {a.metrics_dir}")
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for key in FIGS:
        print("wrote", plot(metrics, key, a.out_dir))


if __name__ == "__main__":
    main()
