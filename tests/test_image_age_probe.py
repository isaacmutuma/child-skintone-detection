"""Synthetic tests for scripts/image_age_probe.py (no images, no real embeddings or ids).

  .venv/bin/python tests/test_image_age_probe.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import image_age_probe as P  # noqa: E402
from src.data.records import role_for_exact_age, stratum_for_age  # noqa: E402
from configs import config as C  # noqa: E402


def test_implied_ages():
    rng = np.random.default_rng(0)
    A = np.linalg.qr(rng.normal(size=(128, 80)))[0].T          # 80 orthonormal anchors
    F = A[[4, 16, 39]]                                         # images = anchors for ages 5, 17, 40
    ex, mo, en = P.implied_ages(F, A, scale=100.0)
    assert list(mo) == [5.0, 17.0, 40.0]
    assert np.allclose(ex, [5, 17, 40], atol=1e-6) and np.all(en < 1e-6)
    ex2, mo2, en2 = P.implied_ages(F, A, scale=1.0)            # low scale: flat, pulled to middle
    assert np.all(en2 > 4.0) and abs(ex2.mean() - 40.5) < 1.0 and list(mo2) == [5.0, 17.0, 40.0]


def synthetic(shift_black_minors, n=12000, seed=1):
    rng = np.random.default_rng(seed)
    groups = rng.choice(["White", "Black", "Asian", "Indian", "Others"], n, p=[.42, .19, .15, .17, .07])
    age = rng.integers(0, 80, n).astype(float)
    role = [role_for_exact_age(x) for x in age]
    stratum = [stratum_for_age(x, C.AGE_STRATA_FALLBACK) if r == "minor" else None
               for x, r in zip(age, role)]
    df = pd.DataFrame({"group": groups, "role": role, "age": age, "stratum": stratum})
    bm = (df.group == "Black") & (df.age <= 17)
    for t in P.TEMPLATES:
        ex = age + 3 + rng.normal(0, 3, n)                       # shared offset: CLIP reads high
        ex[bm] += shift_black_minors
        df[f"age_{t}_expected"] = ex
        df[f"age_{t}_mode"] = np.round(ex - 1 + rng.normal(0, 1, n))
        df[f"entropy_{t}"] = 2.0 + rng.normal(0, 0.1, n)
    df["margin"] = np.where(df[f"age_T0_expected"] < 18, 0.01, -0.01)
    return df


def test_analyse_planted_and_null():
    planted = P.analyse(synthetic(2.0), "utkface", B=500, P=500)
    g = planted["primary"]["contrast"]["age_err_gap_b_minus_a"]
    print(f"  planted +2.0 yr: gap {g['gap']:+.2f} {g['ci']}, mode {planted['robustness']['T0_mode']['gap']:+.2f},"
          f" outcome {planted['decision']['outcome']}")
    assert abs(g["gap"] - 2.0) < 0.6 and g["ci"][0] > 0
    assert planted["decision"]["outcome"] == "over_estimation"
    assert abs(planted["adult_20_59_b_minus_a"]["gap"]) < 0.5      # shift only in minors
    assert set(planted["minor_age_err_vs_white"]) == {"Black", "Asian", "Indian"}

    null = P.analyse(synthetic(0.0, seed=2), "utkface", B=500, P=500)
    g0 = null["primary"]["contrast"]["age_err_gap_b_minus_a"]
    print(f"  null: gap {g0['gap']:+.2f} {g0['ci']}, outcome {null['decision']['outcome']}")
    assert g0["ci"][0] <= 0 <= g0["ci"][1]
    assert null["decision"]["outcome"] == "no_over_estimation_shown"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
