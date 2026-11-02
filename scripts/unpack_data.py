"""Unpack the pinned archives from Drive (RAW_DIR) to DATA_DIR on Colab's local disk,
exactly as Phase 0 did (notebooks/phase0_data_prep.ipynb), verifying SHA256 first.

  CHILD_AUDIT_ENV=colab python scripts/unpack_data.py
UTKFace -> DATA_DIR/UTKFace/*.jpg ; FairFace val -> DATA_DIR/fairface/val/<n>.jpg + CSV.
Idempotent: skips a dataset whose output already has the expected file count.
"""

import csv
import hashlib
import sys
import tarfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import config as C  # noqa: E402

AGE = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
GENDER = ["Male", "Female"]
RACE = ["East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic",
        "Southeast Asian"]
N_UTK_FILES, N_FF_ROWS = 23708, 10954   # results/data_provenance.json


def sha256(p, bs=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def verified(name):
    s = C.SOURCES[name]
    p = Path(C.RAW_DIR) / Path(s["file"]).name
    got = sha256(p)
    if got != s["sha256"]:
        raise SystemExit(f"{name}: sha256 mismatch for {p}: {got}")
    return p


def unpack_utk():
    out = Path(C.DATA_DIR) / "UTKFace"
    if out.is_dir() and sum(1 for _ in out.iterdir()) == N_UTK_FILES:
        print("UTKFace already unpacked"); return
    tar = verified("utkface")
    Path(C.DATA_DIR).mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as t:
        members = [m for m in t.getmembers() if m.isfile() and not m.name.startswith("/")
                   and ".." not in PurePosixPath(m.name).parts]
        t.extractall(C.DATA_DIR, members=members)
    cands = [p for p in Path(C.DATA_DIR).iterdir()
             if p.is_dir() and p.name.lower().startswith("utkface") and p.name != "UTKFace"]
    if not out.exists() and cands:
        cands[0].rename(out)
    n = sum(1 for _ in out.iterdir())
    assert n == N_UTK_FILES, f"UTKFace: {n} files, expected {N_UTK_FILES}"
    print("UTKFace files:", n)


def unpack_ff():
    import pyarrow.parquet as pq
    ff = Path(C.DATA_DIR) / "fairface"
    if (ff / "fairface_label_val.csv").exists() and sum(1 for _ in (ff / "val").iterdir()) == N_FF_ROWS:
        print("FairFace already unpacked"); return
    rows = pq.read_table(verified("fairface_val")).to_pylist()
    (ff / "val").mkdir(parents=True, exist_ok=True)
    with open(ff / "fairface_label_val.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "age", "gender", "race", "service_test"])
        for i, r in enumerate(rows, start=1):   # parquet order, as in Phase 0
            rel = f"val/{i}.jpg"
            (ff / rel).write_bytes(r["image"]["bytes"])
            w.writerow([rel, AGE[r["age"]], GENDER[r["gender"]], RACE[r["race"]], r["service_test"]])
    assert len(rows) == N_FF_ROWS, f"FairFace: {len(rows)} rows, expected {N_FF_ROWS}"
    print("FairFace val rows:", len(rows))


if __name__ == "__main__":
    if C.ENV != "colab":
        raise SystemExit("unpack_data.py is for Colab (CHILD_AUDIT_ENV=colab)")
    unpack_utk()
    unpack_ff()
