"""D5 stratified subsample (<= 2,000 images per dataset), drawn with seed 0 BEFORE any D5 output.

Design (logged in PREREGISTRATION.md Deviations as a clarification):
  minors: per tested group x primary-analysis stratum, take min(n, MINOR_CAP) at random
          (UTKFace strata 0-9 / 10-17; FairFace 0-2 / 3-9)
  adults (20-59, FPR): per tested group, take min(n, ADULT_CAP) at random
  UTKFace tested groups only (Others excluded); FairFace all 7 groups.
TPR/FPR are estimated within group x stratum, and the analysis re-weights strata with the
pooled reference weights, so per-cell sampling doesn't bias the stratified estimate.

Writes id lists to PER_IMAGE_DIR/vlm_subsample__<dataset>.txt (Drive; never committed) and
aggregate cell counts to results/phase1/vlm_subsample.json.
"""

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import config as C  # noqa: E402
from src.data import LOADERS, MINOR, ADULT_FPR  # noqa: E402
from src.data.records import stratum_for_age  # noqa: E402
from src.provenance import write_result  # noqa: E402

DESIGN = {
    "utkface": {"groups": C.UTKFACE_TESTED_GROUPS, "minor_cap": 150, "adult_cap": 200,
                "strata": "fallback 0-9/10-17"},
    "fairface_val": {"groups": C.FAIRFACE_RACES, "minor_cap": 110, "adult_cap": 140,
                     "strata": "native 0-2/3-9"},
}
MAX_PER_DATASET = 2000


def stratum(r, ds):
    return stratum_for_age(r.age, C.AGE_STRATA_FALLBACK) if ds == "utkface" else r.stratum


def main():
    rng = random.Random(C.SEED)
    out = {}
    for ds, d in DESIGN.items():
        recs, _ = LOADERS[ds]()
        cells = defaultdict(list)
        for r in recs:
            if r.group not in d["groups"]:
                continue
            if r.role == MINOR:
                cells[("minor", r.group, stratum(r, ds))].append(r)
            elif r.role == ADULT_FPR:
                cells[("adult_fpr", r.group, None)].append(r)
        chosen, counts = [], {}
        for key in sorted(cells, key=str):
            pool = sorted(cells[key], key=lambda r: r.id)       # deterministic order
            cap = d["minor_cap"] if key[0] == "minor" else d["adult_cap"]
            take = rng.sample(pool, min(cap, len(pool)))
            chosen += take
            counts["|".join(str(k) for k in key)] = {"available": len(pool), "sampled": len(take)}
        assert len(chosen) <= MAX_PER_DATASET, (ds, len(chosen))
        ids_path = Path(C.PER_IMAGE_DIR) / f"vlm_subsample__{ds}.txt"
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.write_text("\n".join(sorted(r.id for r in chosen)) + "\n")
        out[ds] = {"design": d, "n_total": len(chosen), "cells": counts,
                   "by_role": dict(Counter(r.role for r in chosen))}
        print(f"{ds}: {len(chosen)} images {out[ds]['by_role']} -> {ids_path}")
    write_result(C.RESULTS_DIR / "phase1" / "vlm_subsample.json",
                 {"max_per_dataset": MAX_PER_DATASET, "seed": C.SEED, "datasets": out},
                 {"ENV": C.ENV, "SEED": C.SEED}, C.REPO_DIR)


if __name__ == "__main__":
    main()
