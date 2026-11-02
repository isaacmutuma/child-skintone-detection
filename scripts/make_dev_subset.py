"""Build the <=500-image dev subset (stratified over group x role x stratum) as a
zip for the Mac. Run on Colab after data is unpacked. Output layout inside the zip
mirrors data/dev/: UTKFace/<files>, fairface/val/<files>, fairface/fairface_label_val.csv

  python scripts/make_dev_subset.py --out /content/drive/MyDrive/child-detection-audit/dev_subset.zip
"""

import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import config as C  # noqa: E402
from src.data import load_utkface, load_fairface_val, dev_subset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-utk", type=int, default=250)
    ap.add_argument("--n-ff", type=int, default=250)
    a = ap.parse_args()
    assert a.n_utk + a.n_ff <= C.DEV_SUBSET_MAX

    utk, _ = load_utkface()
    ff, _ = load_fairface_val()
    utk_s, ff_s = dev_subset(utk, a.n_utk), dev_subset(ff, a.n_ff)

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_STORED) as z:
        for r in utk_s:
            z.write(r.path, f"UTKFace/{Path(r.path).name}")
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["file", "age", "gender", "race", "service_test"])
        for r in ff_s:
            rel = Path(r.path).relative_to(C.FAIRFACE_DIR).as_posix()
            z.write(r.path, f"fairface/{rel}")
            w.writerow([rel, r.age_bin, r.gender.capitalize(), r.group, ""])
        z.writestr("fairface/fairface_label_val.csv", buf.getvalue())
    print(f"dev subset: {len(utk_s)} UTKFace + {len(ff_s)} FairFace -> {a.out}")


if __name__ == "__main__":
    main()
