"""Phase 0: counts of minors per group x age stratum, adults for FPR, and a
power check. Writes results/phase0_counts.json (aggregate counts only).

Usage:
  python scripts/run_phase0_counts.py                    # both datasets, config paths
  python scripts/run_phase0_counts.py --datasets utkface
  python scripts/run_phase0_counts.py --out results/x.json --utkface-dir ... --fairface-dir ...
"""

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.stats import norm  # noqa: E402

from configs import config as C  # noqa: E402
from src.data import load_utkface, load_fairface_val, MINOR, ADULT_FPR, AMBIGUOUS  # noqa: E402
from src.data.records import stratum_for_age  # noqa: E402
from src.provenance import write_result  # noqa: E402

P_ASSUMED = 0.85   # assumed TPR for power check (EXPERIMENT_PLAN.md §3)
POWER = 0.80


def tested_groups(dataset):
    return C.UTKFACE_TESTED_GROUPS if dataset == "utkface" else C.FAIRFACE_RACES


def strata_labels(dataset, strata):
    if dataset == "fairface_val":
        return list(C.FAIRFACE_MINOR_BINS)
    return [f"{lo}-{hi}" for lo, hi in strata]


def count_cells(records, dataset, strata):
    groups = sorted({r.group for r in records})
    labels = strata_labels(dataset, strata)
    minors = [r for r in records if r.role == MINOR]
    if dataset == "utkface":
        key = lambda r: stratum_for_age(r.age, strata)  # noqa: E731
    else:
        key = lambda r: r.stratum  # noqa: E731
    cell = Counter((r.group, key(r)) for r in minors)
    table = {g: {s: cell.get((g, s), 0) for s in labels} for g in groups}
    small = [{"group": g, "stratum": s, "n": table[g][s]}
             for g in tested_groups(dataset) for s in labels
             if table.get(g, {}).get(s, 0) < C.MIN_CELL]
    return table, small


def power_check(table, dataset):
    """Approximate CI half-widths and minimum detectable gap at P_ASSUMED."""
    z, zb = norm.ppf(1 - C.ALPHA / 2), norm.ppf(POWER)
    pq = P_ASSUMED * (1 - P_ASSUMED)
    pooled = Counter()
    for g, row in table.items():
        pooled.update(row)
    total = sum(pooled.values())
    w = {s: pooled[s] / total for s in pooled} if total else {}

    def var_unstrat(g):
        n = sum(table[g].values())
        return pq / n if n else math.inf

    def var_strat(g):  # var of reference-weighted TPR
        if any(table[g][s] == 0 for s in w if w[s] > 0):
            return math.inf
        return sum(w[s] ** 2 * pq / table[g][s] for s in w if w[s] > 0)

    per_group = {g: {"n_minors": sum(table[g].values()),
                     "ci_halfwidth_unstratified": round(float(z) * math.sqrt(var_unstrat(g)), 4),
                     "ci_halfwidth_stratified": round(float(z) * math.sqrt(var_strat(g)), 4)}
                 for g in tested_groups(dataset) if g in table}
    a, b = C.PRIMARY_CONTRAST
    mde = {}
    if a in table and b in table:
        for name, f in [("unstratified", var_unstrat), ("stratified", var_strat)]:
            mde[name] = round(float((z + zb) * math.sqrt(f(a) + f(b))), 4)
    return {
        "assumptions": {"tpr": P_ASSUMED, "alpha_two_sided": C.ALPHA, "power": POWER,
                        "note": "normal approximation, equal TPR in both groups; "
                                "no Holm correction; stratified uses pooled-minor weights"},
        "reference_weights": {s: round(v, 4) for s, v in w.items()},
        "per_group": per_group,
        "primary_contrast": list(C.PRIMARY_CONTRAST),
        "min_detectable_gap_80pct_power": mde,
    }


def summarise(dataset, records, skipped):
    by_role = Counter((r.group, r.role) for r in records)
    groups = sorted({r.group for r in records})
    table, small = count_cells(records, dataset, C.AGE_STRATA)
    out = {
        "n_records": len(records),
        "n_skipped_unparseable_or_missing": len(skipped),
        "groups": groups,
        "by_group_role": {g: {role: by_role.get((g, role), 0)
                              for role in (MINOR, ADULT_FPR, AMBIGUOUS, "other")}
                          for g in groups},
        "minors_by_group_stratum": table,
        "cells_below_min": small,
        "fallback_strata_needed": dataset == "utkface" and bool(small),
    }
    if out["fallback_strata_needed"]:
        ftable, fsmall = count_cells(records, dataset, C.AGE_STRATA_FALLBACK)
        out["minors_by_group_stratum_fallback"] = ftable
        out["cells_below_min_fallback"] = fsmall
        table = ftable
    out["power"] = power_check(table, dataset)
    if dataset == "fairface_val":
        out["note"] = ("FairFace minor strata are native bins 0-2 / 3-9; 10-19 is "
                       "AMBIGUOUS (excluded from TPR). Cells < MIN_CELL are reported "
                       "but the 0-9/10-17 fallback cannot be formed from FairFace bins.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["utkface", "fairface_val"])
    ap.add_argument("--utkface-dir", default=None)
    ap.add_argument("--fairface-dir", default=None)
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "phase0_counts.json"))
    args = ap.parse_args()

    payload = {"datasets": {}}
    for ds in args.datasets:
        if ds == "utkface":
            recs, skipped = load_utkface(args.utkface_dir)
        elif ds == "fairface_val":
            recs, skipped = load_fairface_val(args.fairface_dir)
        else:
            raise SystemExit(f"unknown dataset {ds}")
        payload["datasets"][ds] = summarise(ds, recs, skipped)
        s = payload["datasets"][ds]
        print(f"[{ds}] records={s['n_records']} skipped={s['n_skipped_unparseable_or_missing']} "
              f"cells<{C.MIN_CELL}={len(s['cells_below_min'])} "
              f"MDE={s['power']['min_detectable_gap_80pct_power']}")
    p = write_result(args.out, payload, C.as_dict(), C.REPO_DIR)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
