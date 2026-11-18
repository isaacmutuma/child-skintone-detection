"""Image-side age probe of zero-shot CLIP (plan: docs/IMAGE_AGE_PROBE_PLAN.md).

Exploratory, not preregistered. For each face image, CLIP's image embedding is scored
against 80 anchors "a photo of a {N}-year-old person" (N = 1..80, no race word); a softmax
with CLIP's own logit scale gives an age distribution, from which we take the expected age
(primary), the mode age and the entropy. The Black - White signed age error is then computed
with the audit's unchanged H2b code (src/analysis/metrics.py), as for MiVOLO and D3.

Embeddings: either a precomputed .npz (ids + raw image embeddings, sha256-checked; e.g. the
course project's UTKFace ViT-L/14 file) or extracted here from the images with D4's own
loader. Either way, D4's margin is recomputed from the embeddings and must reproduce the
Phase 1 per-image scores (max |diff| < 1e-3, identical decisions) before anything else runs.

  CHILD_AUDIT_ENV=colab python scripts/image_age_probe.py --arch ViT-L-14 --dataset utkface \\
      --embeddings /content/drive/MyDrive/robust_child_filtering_model/embeddings/utkface__clip_vitl14.npz \\
      --embeddings-sha256 4a6f66c1ec7b888f87ada759ab6bec3bde1407a077ff80ba28b12d83803bb3c3
  CHILD_AUDIT_ENV=colab python scripts/image_age_probe.py --arch ViT-B-32 --dataset utkface   # extracts

Per-image output (Drive only): PER_IMAGE_DIR/image_age_probe/<arch>__<dataset>.csv
Aggregate output (committable): results/image_age_probe/image_age_probe_<arch>_<dataset>.json
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs import config as C  # noqa: E402
from src.analysis import metrics as M  # noqa: E402
from src.provenance import write_result  # noqa: E402

AGES = np.arange(1, 81, dtype=float)
TEMPLATES = {  # T0 is the primary ruler (same as the text probe's M2); T1, T2 sensitivity only
    "T0": "a photo of a {n}-year-old person",
    "T1": "a close-up photo of the face of a {n}-year-old person",
    "T2": "a portrait of a {n}-year-old person",
}
CROSSCHECK_TOL = 1e-3
UTK_BANDS = [(0, 9), (10, 17), (18, 19), (20, 39), (40, 59), (60, 200)]


def sha256(path, bs=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


# ── pure functions (tested in tests/test_image_age_probe.py) ─────────────────
def implied_ages(F, A, scale):
    """Expected age, mode age and entropy (nats) per row.

    F: (n, d) L2-normalised image embeddings; A: (80, d) L2-normalised anchor embeddings
    for ages 1..80; scale: CLIP's logit scale (exp of the learned parameter).
    """
    logits = scale * (F @ A.T)
    logits = logits - logits.max(1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(1, keepdims=True)
    expected = p @ AGES
    mode = AGES[p.argmax(1)]
    entropy = -(p * np.log(np.clip(p, 1e-300, None))).sum(1)
    return expected, mode, entropy


def _diff_block(va, vb, B, rng):
    """Mean(b) - mean(a) with a within-group bootstrap CI (plain means)."""
    if len(va) == 0 or len(vb) == 0:
        return {"gap": None, "ci": [None, None], "n_a": int(len(va)), "n_b": int(len(vb))}
    reps = M.boot_mean(vb, B, rng) - M.boot_mean(va, B, rng)
    return {"gap": float(np.mean(vb) - np.mean(va)), "ci": M.ci(reps),
            "n_a": int(len(va)), "n_b": int(len(vb))}


def analyse(df, ds, B, P, seed=None):
    """All probe statistics for one dataset.

    df columns: group, role, age, stratum, margin, and for each template t:
    age_<t>_expected, age_<t>_mode, entropy_<t>.
    """
    rng = np.random.default_rng(C.SEED if seed is None else seed)
    w = M.reference_weights(ds)
    a, b = C.PRIMARY_CONTRAST                    # White, Black
    tested = C.UTKFACE_TESTED_GROUPS if ds == "utkface" else C.FAIRFACE_RACES

    def frame(col):
        d = df[["group", "role", "age", "stratum"]].copy()
        d["is_child"] = df["margin"] > 0         # D4's decision
        d["face_found"] = np.nan                 # no face stage: skips face-fail metrics
        d["age_est"] = df[col]
        return d

    out = {"reference_weights": w, "n_images": int(len(df))}
    # primary: T0 expected age, full H2b treatment (per group + Black - White contrast)
    d0 = frame("age_T0_expected")
    out["primary"] = {"estimate": "T0 expected age",
                      "groups": M.group_metrics(d0, ds, w, B, rng),
                      "contrast": M.contrast(d0, ds, w, a, b, B, P, rng)}
    # robustness: mode age and the other templates (contrast only)
    out["robustness"] = {}
    for t in TEMPLATES:
        for est in ("expected", "mode"):
            if (t, est) == ("T0", "expected"):
                continue
            c = M.contrast(frame(f"age_{t}_{est}"), ds, w, a, b, B, P, rng)
            out["robustness"][f"{t}_{est}"] = c.get("age_err_gap_b_minus_a")

    # uncertainty: stratified mean entropy of minors, Black - White
    dm = df[df.role == "minor"].copy()
    dm["astratum"] = M.analysis_stratum(dm, ds)
    ea, eb = dm[dm.group == a], dm[dm.group == b]
    reps = (M.boot_stratified(eb["entropy_T0"].values, eb.astratum.values, w, B, rng)
            - M.boot_stratified(ea["entropy_T0"].values, ea.astratum.values, w, B, rng))
    out["entropy_minor_b_minus_a"] = {
        "gap": float(M.stratified_rate(eb["entropy_T0"].values, eb.astratum.values, w)
                     - M.stratified_rate(ea["entropy_T0"].values, ea.astratum.values, w)),
        "ci": M.ci(reps)}

    # every tested group vs White (minors, stratified signed error, T0 expected)
    err = df["age_T0_expected"] - df["age"]
    vs_white = {}
    for g in tested:
        if g == a:
            continue
        mg = dm[dm.group == g]
        if len(mg) == 0:
            continue
        eg, ew = err.loc[mg.index].values, err.loc[ea.index].values
        reps = (M.boot_stratified(eg, mg.astratum.values, w, B, rng)
                - M.boot_stratified(ew, ea.astratum.values, w, B, rng))
        vs_white[g] = {"gap": float(M.stratified_rate(eg, mg.astratum.values, w)
                                    - M.stratified_rate(ew, ea.astratum.values, w)),
                       "ci": M.ci(reps)}
    out["minor_age_err_vs_white"] = vs_white

    # teenagers only (UTKFace) and adults 20-59: plain mean signed error, Black - White
    if ds == "utkface":
        teen = (df.age >= 10) & (df.age <= 17)
        out["teen_10_17_b_minus_a"] = _diff_block(err[teen & (df.group == a)].values,
                                                  err[teen & (df.group == b)].values, B, rng)
    adult = df.role == "adult_fpr"
    out["adult_20_59_b_minus_a"] = _diff_block(err[adult & (df.group == a)].values,
                                               err[adult & (df.group == b)].values, B, rng)

    # calibration (descriptive, pooled over groups)
    if ds == "utkface":
        cal = {}
        for lo, hi in UTK_BANDS:
            m = (df.age >= lo) & (df.age <= hi)
            name = f"{lo}+" if hi >= 200 else f"{lo}-{hi}"
            cal[name] = {"n": int(m.sum()), "mean_true_age": float(df.age[m].mean()),
                         "mean_implied_age_T0": float(df["age_T0_expected"][m].mean()),
                         "mean_mode_age_T0": float(df["age_T0_mode"][m].mean())}
    else:
        cal = {str(k): {"n": int(len(g)), "mean_implied_age_T0": float(g["age_T0_expected"].mean())}
               for k, g in df.groupby(df["age"])}
    out["calibration"] = cal
    out["pearson_r_implied_vs_true"] = float(np.corrcoef(df["age_T0_expected"], df["age"])[0, 1])

    # decision rule (docs/IMAGE_AGE_PROBE_PLAN.md, fixed before running)
    prim = out["primary"]["contrast"]["age_err_gap_b_minus_a"]
    mode = out["robustness"]["T0_mode"]
    c1 = bool(prim["gap"] > 0 and prim["ci"][0] is not None and prim["ci"][0] > 0)
    c2 = bool(mode is not None and mode["gap"] > 0)
    out["decision"] = {
        "c1_expected_gap_positive_ci_excludes_0": c1, "c2_mode_gap_same_sign": c2,
        "outcome": ("over_estimation" if c1 and c2 else
                    "possibly_uncertainty_driven" if c1 else "no_over_estimation_shown")}
    return out


# ── embeddings ───────────────────────────────────────────────────────────────
def extract(det, records, batch_size=128, workers=2):
    """Raw image embeddings with D4's own preprocessing and model (GPU)."""
    import torch
    from PIL import Image

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, i):
            return det.preprocess(Image.open(records[i].path).convert("RGB"))

    dl = torch.utils.data.DataLoader(DS(), batch_size=batch_size, num_workers=workers)
    out = []
    with torch.no_grad():
        for i, x in enumerate(dl):
            out.append(det.model.encode_image(x.to(det.device)).float().cpu().numpy())
            print(f"\r{min((i + 1) * batch_size, len(records))}/{len(records)}", end="", flush=True)
    print()
    return np.concatenate(out)


def main():
    import open_clip
    import torch
    from src.data import LOADERS
    from src.detectors.clip_zeroshot import ClipZeroShot

    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["ViT-B-32", "ViT-L-14"])
    ap.add_argument("--dataset", required=True, choices=["utkface", "fairface_val"])
    ap.add_argument("--embeddings", default=None, help="precomputed .npz (ids, emb)")
    ap.add_argument("--embeddings-sha256", default=None)
    a = ap.parse_args()
    t0 = time.time()
    det_name = f"clip_{a.arch.lower().replace('-', '')}"

    records, _ = LOADERS[a.dataset]()
    ids = [r.id for r in records]
    det = ClipZeroShot(a.arch)                   # pinned, sha256-checked, QuickGELU asserted
    winfo = det.weights_info()

    # 1. embeddings
    emb_dir = Path(C.DRIVE_ROOT) / "embeddings"
    if a.embeddings:
        got = sha256(a.embeddings)
        if a.embeddings_sha256 and got != a.embeddings_sha256:
            raise SystemExit(f"embeddings sha256 mismatch: {got}")
        z = np.load(a.embeddings)
        pos = {i: k for k, i in enumerate(z["ids"].tolist())}
        missing = [i for i in ids if i not in pos]
        if missing:
            raise SystemExit(f"{len(missing)} records missing from {a.embeddings}")
        E = z["emb"][[pos[i] for i in ids]].astype(np.float32)
        emb_src = {"source": "precomputed", "path": a.embeddings, "sha256": got}
    else:
        dst = emb_dir / f"{det_name}__{a.dataset}.npz"
        if dst.exists():
            z = np.load(dst)
            assert z["ids"].tolist() == ids, f"{dst} ids differ from the loader"
            E = z["emb"].astype(np.float32)
        else:
            E = extract(det, records)
            emb_dir.mkdir(parents=True, exist_ok=True)
            np.savez(dst, ids=np.array(ids), emb=E)
        emb_src = {"source": "extracted with D4's loader", "drive_path": str(dst),
                   "sha256": sha256(dst)}
    F = E / np.linalg.norm(E, axis=1, keepdims=True)

    # 2. cross-check: D4's margin from these embeddings must reproduce Phase 1
    T = det.text.cpu().numpy()
    sims = F @ T.T
    margin = sims[:, :det.n_child].max(1) - sims[:, det.n_child:].max(1)
    p1 = pd.read_csv(Path(C.PER_IMAGE_DIR) / f"per_image__{det_name}__{a.dataset}__full.csv",
                     usecols=["id", "score", "is_child"], dtype={"id": str}).set_index("id")
    if set(p1.index) != set(ids):
        raise SystemExit("Phase 1 per-image file covers different images")
    p1 = p1.loc[ids]
    diff = np.abs(margin - p1["score"].to_numpy(float))
    same = ((margin > 0) == p1["is_child"].astype(bool).to_numpy())
    cross = {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
             "decision_agreement": float(same.mean()), "tolerance": CROSSCHECK_TOL}
    cross["pass"] = bool(diff.max() < CROSSCHECK_TOL and same.all())
    print("cross-check vs Phase 1:", cross)
    if not cross["pass"]:
        raise SystemExit("STOP: embeddings do not reproduce Phase 1 D4 scores")

    # 3. implied ages from the anchors (text encoder of the same pinned model)
    tok = open_clip.get_tokenizer(a.arch)
    scale = float(det.model.logit_scale.exp())
    df = pd.DataFrame({"id": ids, "group": [r.group for r in records],
                       "role": [r.role for r in records], "age": [r.age for r in records],
                       "stratum": [r.stratum for r in records], "margin": margin})
    with torch.no_grad():
        for t, tpl in TEMPLATES.items():
            A = det.model.encode_text(tok([tpl.format(n=int(n)) for n in AGES]).to(det.device))
            A = A.float().cpu().numpy()
            A /= np.linalg.norm(A, axis=1, keepdims=True)
            ex, mo, en = implied_ages(F, A, scale)
            df[f"age_{t}_expected"], df[f"age_{t}_mode"], df[f"entropy_{t}"] = ex, mo, en

    per_dir = Path(C.PER_IMAGE_DIR) / "image_age_probe"
    per_dir.mkdir(parents=True, exist_ok=True)
    per_path = per_dir / f"{a.arch}__{a.dataset}.csv"
    df.drop(columns=["group", "role", "age", "stratum"]).to_csv(per_path, index=False)

    # 4. analysis with the audit's H2b code
    res = analyse(df, a.dataset, C.N_BOOTSTRAP, C.N_BOOTSTRAP)
    ref = json.loads((C.RESULTS_DIR / "phase1" / f"metrics_{det_name}_{a.dataset}.json").read_text())
    res["tpr_gap_check"] = {"probe": res["primary"]["contrast"]["tpr_gap"]["gap"],
                            "phase1": ref["primary_contrast"]["tpr_gap"]["gap"]}
    cfg = {"arch": a.arch, "dataset": a.dataset, "weights": winfo, "embeddings": emb_src,
           "templates": TEMPLATES, "ages": "1..80", "logit_scale": scale,
           "n_boot": C.N_BOOTSTRAP, "seed": C.SEED, "plan": "docs/IMAGE_AGE_PROBE_PLAN.md",
           "exploratory": True, "torch": torch.__version__}
    payload = {**res, "crosscheck_vs_phase1": cross,
               "per_image_file": {"drive_path": str(per_path), "sha256": sha256(per_path)},
               "seconds": round(time.time() - t0, 1)}
    out = write_result(ROOT / f"results/image_age_probe/image_age_probe_{a.arch}_{a.dataset}.json",
                       payload, {**cfg, "config": C.as_dict()}, ROOT)

    g = res["primary"]["contrast"]["age_err_gap_b_minus_a"]
    print(f"\n{a.arch} / {a.dataset}: Black - White minor age error (T0 expected) "
          f"{g['gap']:+.2f} yr [{g['ci'][0]:+.2f}, {g['ci'][1]:+.2f}], p_perm {g['p_perm']:.4f}")
    for k, v in res["robustness"].items():
        if v:
            print(f"  {k:12s} {v['gap']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]")
    e = res["entropy_minor_b_minus_a"]
    print(f"  entropy B-W {e['gap']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]")
    for k in ("teen_10_17_b_minus_a", "adult_20_59_b_minus_a"):
        if k in res and res[k]["gap"] is not None:
            print(f"  {k:22s} {res[k]['gap']:+.2f} [{res[k]['ci'][0]:+.2f}, {res[k]['ci'][1]:+.2f}]")
    print("  decision:", res["decision"])
    print("  TPR gap check (must match Phase 1):", res["tpr_gap_check"])
    print("->", per_path, "\n->", out)


if __name__ == "__main__":
    main()
