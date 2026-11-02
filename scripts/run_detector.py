"""Run one detector on one dataset.

Per-image outputs (id, group, role, stratum, age, detector outputs, confounders) go to
PER_IMAGE_DIR (gitignored locally / Drive on Colab), never to results/.
results/phase1/runs/<detector>__<dataset>__<tag>.json gets aggregate run metadata only.

  python scripts/run_detector.py --detector clip_vitb32 --dataset utkface            # local = dev subset
  CHILD_AUDIT_ENV=colab python scripts/run_detector.py --detector clip_vitl14 --dataset fairface_val
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from configs import config as C  # noqa: E402
from src.data import LOADERS  # noqa: E402
from src.detectors import build, ALLOWED  # noqa: E402
from src.analysis.image_stats import image_stats  # noqa: E402
from src.provenance import write_result  # noqa: E402


def prereg_hash():
    txt = (C.REPO_DIR / "PREREGISTRATION.md").read_text()
    m = re.search(r"Commit hash after freezing: `([0-9a-f]{7,40})`", txt)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--dataset", required=True, choices=list(LOADERS))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N records (debugging)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dev", action="store_true", help="dev-subset smoke run (tag 'dev')")
    ap.add_argument("--ids-file", default=None,
                    help="text file of record ids to keep (D5 subsample); tag gets '_sub'")
    a = ap.parse_args()

    if a.dataset not in ALLOWED.get(a.detector, set()):
        raise SystemExit(f"{a.detector} is not allowed on {a.dataset} (see PREREGISTRATION.md)")
    tag = "dev" if (C.ENV == "local" or a.dev) else "full"
    ph = prereg_hash()
    if tag == "full" and not ph:
        raise SystemExit("PREREGISTRATION.md has no frozen commit hash; full runs are blocked.")

    records, skipped = LOADERS[a.dataset]()
    if tag == "dev" and len(records) > C.DEV_SUBSET_MAX:
        raise SystemExit(f"dev run on {len(records)} images > DEV_SUBSET_MAX; point "
                         "UTKFACE_DIR/FAIRFACE_DIR at the dev subset or drop --dev")
    if a.ids_file:
        keep = set(Path(a.ids_file).read_text().split())
        records = [r for r in records if r.id in keep]
        missing = len(keep) - len(records)
        if missing:
            raise SystemExit(f"{missing} ids from {a.ids_file} not found in {a.dataset}")
        tag += "_sub"
    if a.limit:
        records = records[: a.limit]

    det = build(a.detector, device=a.device)
    t0, rows = time.time(), []
    for i in range(0, len(records), a.batch_size):
        batch = records[i:i + a.batch_size]
        for r, o in zip(batch, det.predict(batch)):
            try:
                st = image_stats(r.path)
            except Exception:
                st = {}
            rows.append({"id": r.id, "dataset": r.dataset, "group": r.group, "role": r.role,
                         "stratum": r.stratum, "age": r.age, "age_is_exact": r.age_is_exact,
                         "age_bin": r.age_bin, "gender": r.gender, **o, **st})
        print(f"\r{min(i + a.batch_size, len(records))}/{len(records)}", end="", flush=True)
    secs = time.time() - t0
    print()

    df = pd.DataFrame(rows)
    out_dir = Path(C.PER_IMAGE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image = out_dir / f"per_image__{a.detector}__{a.dataset}__{tag}.csv"
    df.to_csv(per_image, index=False)

    # aggregate-only summary (no ids, no per-image values)
    summary = {
        "detector": a.detector, "dataset": a.dataset, "tag": tag,
        "preregistration_commit": ph, "n_images": len(df), "n_skipped_loader": len(skipped),
        "seconds": round(secs, 1), "device": str(getattr(det, "device", "")),
        "n_flagged_child": int(df["is_child"].sum()),
        "fail_reasons": dict(Counter(df["fail_reason"].dropna())),
        "n_by_role": dict(Counter(df["role"])),
        "weights": det.weights_info(),
    }
    res = C.RESULTS_DIR / "phase1" / "runs" / f"{a.detector}__{a.dataset}__{tag}.json"
    write_result(res, summary, C.as_dict(), C.REPO_DIR)

    env_path = C.RESULTS_DIR / "env.json"   # pin weights alongside package versions
    if env_path.exists():
        env = json.loads(env_path.read_text())
        env.setdefault("weights", {}).update(det.weights_info())
        env_path.write_text(json.dumps(env, indent=2) + "\n")
    print(f"per-image -> {per_image}\nsummary   -> {res}")


if __name__ == "__main__":
    main()
