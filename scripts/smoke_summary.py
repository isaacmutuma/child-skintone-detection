"""Aggregate smoke-test table from per-image CSVs (prints/stores counts and rates only).

  python scripts/smoke_summary.py --tag dev
Writes results/phase1/smoke_summary_<tag>.json. Rates on the dev subset are NOT results:
cells hold ~10-35 images and the subset is balanced, not representative.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from configs import config as C  # noqa: E402
from src.provenance import write_result  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dev")
    a = ap.parse_args()
    rows = []
    for f in sorted(Path(C.PER_IMAGE_DIR).glob(f"per_image__*__{a.tag}.csv")):
        _, det, ds, _ = f.stem.split("__")
        df = pd.read_csv(f)
        minors, adults = df[df.role == "minor"], df[df.role == "adult_fpr"]
        ff = df["face_found"].dropna()
        row = {"detector": det, "dataset": ds, "n": len(df),
               "n_minor": len(minors), "n_adult_fpr": len(adults),
               "flag_rate_minor": round(minors.is_child.mean(), 3) if len(minors) else None,
               "flag_rate_adult": round(adults.is_child.mean(), 3) if len(adults) else None,
               "face_found_rate": round(ff.astype(bool).mean(), 3) if len(ff) else None,
               "n_null_score": int(df.score.isna().sum()),
               "fail_reasons": df.fail_reason.dropna().value_counts().to_dict()}
        if df.age_est.notna().any() and len(minors):
            m = minors.dropna(subset=["age_est"])
            row["mean_signed_age_err_minor"] = round(float((m.age_est - m.age).mean()), 2) if len(m) else None
        rows.append(row)
    if not rows:
        raise SystemExit(f"no per-image files for tag {a.tag} in {C.PER_IMAGE_DIR}")
    print(pd.DataFrame(rows).drop(columns=["fail_reasons"]).to_string(index=False))
    for r in rows:
        if r["fail_reasons"]:
            print(r["detector"], r["dataset"], "fail reasons:", r["fail_reasons"])
    out = C.RESULTS_DIR / "phase1" / f"smoke_summary_{a.tag}.json"
    write_result(out, {"tag": a.tag, "warning": "dev subset; not a result", "runs": rows},
                 {"ENV": C.ENV}, C.REPO_DIR)
    print("wrote", out)


if __name__ == "__main__":
    main()
