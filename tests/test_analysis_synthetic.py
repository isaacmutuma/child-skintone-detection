"""Synthetic recovery test for the Phase 1 analysis (no real images or outputs).

Builds fake per-image CSVs with the real Phase 0 group x stratum counts and two planted
scenarios, runs scripts/run_analysis.py on them, and checks the known answers:
  S1: age effect only, no group effect -> stratified White-Black gap CI covers 0
  S2: Black TPR -0.08 per stratum, +3 pts no-face, +2 yr age error
      -> gap ~ +0.10 with CI > 0, Holm p < 0.05, face-fail and age-error gaps detected
Run:  .venv/bin/python tests/test_analysis_synthetic.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def make(out, scen):
    rng = np.random.default_rng(1)
    rng = np.random.default_rng(1)
    d = json.load(open(ROOT / "results/phase0_counts.json"))["datasets"]
    def tpr(group, stratum):
        base = {"0-9": 0.95, "10-17": 0.70, "0-2": 0.97, "3-9": 0.92}[stratum]
        return base - (0.08 if scen == "S2" and group == "Black" else 0.0)
    for ds in ["utkface", "fairface_val"]:
        rows = []
        v = d[ds]
        table = v["minors_by_group_stratum_fallback"] if ds == "utkface" else v["minors_by_group_stratum"]
        i = 0
        for g, row in table.items():
            for s, n in row.items():
                for _ in range(n):
                    age = (rng.integers(0, 10) if s == "0-9" else rng.integers(10, 18)) if ds == "utkface" else {"0-2": 1.0, "3-9": 6.0}[s]
                    flag = rng.random() < tpr(g, s)
                    face = rng.random() > (0.05 if g == "Black" and scen == "S2" else 0.02)
                    flag = flag and face
                    rows.append(dict(id=f"m{i}", group=g, role="minor", stratum=(None if ds=="utkface" else s), age=float(age),
                                     is_child=flag, face_found=face, score=(1.0 if flag else -1.0) if face else np.nan,
                                     age_est=(age + (3 if g == "Black" and scen == "S2" else 1)) if face else np.nan,
                                     face_area_frac=rng.uniform(.3,.6) if face else np.nan,
                                     brightness=rng.uniform(.3,.7), contrast=rng.uniform(.1,.3), blur_lapvar=rng.uniform(.001,.01)))
                    i += 1
            na = v["by_group_role"][g]["adult_fpr"]
            for _ in range(na):
                f = rng.random() < 0.03
                rows.append(dict(id=f"a{i}", group=g, role="adult_fpr", stratum=None, age=30.0, is_child=f, face_found=True,
                                 score=1.0 if f else -1.0, age_est=30.0, face_area_frac=rng.uniform(.3,.6),
                                 brightness=rng.uniform(.3,.7), contrast=.2, blur_lapvar=.005)); i += 1
        df = pd.DataFrame(rows)
        df.to_csv(f"{out}/per_image__mivolo_face__{ds}__{scen}.csv", index=False)
        df.assign(face_found=np.nan, age_est=np.nan, face_area_frac=np.nan,
                  score=np.where(df.is_child, 1.0, -1.0)).to_csv(f"{out}/per_image__clip_vitb32__{ds}__{scen}.csv", index=False)



def run(scen, tmp):
    make(tmp, scen)
    out = Path(tmp) / f"out_{scen}"
    subprocess.run([sys.executable, str(ROOT / "scripts/run_analysis.py"), "--tag", scen,
                    "--per-image-dir", tmp, "--out-dir", str(out)], check=True, capture_output=True)
    return out


def main():
    with tempfile.TemporaryDirectory() as tmp:
        o1, o2 = run("S1", tmp), run("S2", tmp)
        for det in ["mivolo_face", "clip_vitb32"]:
            for ds in ["utkface", "fairface_val"]:
                g1 = json.load(open(o1 / f"metrics_{det}_{ds}.json"))["primary_contrast"]["tpr_gap"]
                g2 = json.load(open(o2 / f"metrics_{det}_{ds}.json"))["primary_contrast"]
                assert g1["ci"][0] < 0 < g1["ci"][1], (det, ds, "S1 CI should cover 0", g1)
                assert 0.07 < g2["tpr_gap"]["gap"] < 0.13 and g2["tpr_gap"]["ci"][0] > 0, (det, ds, g2["tpr_gap"])
                if det == "mivolo_face":
                    assert g2["face_fail_gap_b_minus_a"]["ci"][0] > 0, (ds, "face-fail gap not found")
                    assert abs(g2["age_err_gap_b_minus_a"]["gap"] - 2.0) < 1e-6, (ds, "age-err gap")
        pe = json.load(open(o2 / "primary_endpoint.json"))["datasets"]["utkface"]["detectors"]
        assert all(v["p_holm"] < 0.05 for v in pe.values()), pe
    print("synthetic recovery test PASSED")


if __name__ == "__main__":
    main()
