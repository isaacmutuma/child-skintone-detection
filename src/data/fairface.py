"""FairFace validation-split loader (official CSV format).

Expects FAIRFACE_DIR/fairface_label_val.csv with columns
  file, age, gender, race[, service_test]
where `file` is relative to FAIRFACE_DIR (e.g. 'val/1.jpg').
If the data comes from the HF mirror, export it to this layout first.

Age is a bin; `age` holds the bin midpoint and age_is_exact=False.
Minor strata are FairFace's native bins (0-2, 3-9): the plan's 0-4/5-9 strata
cannot be formed from FairFace bins.
"""

from pathlib import Path

import pandas as pd

from configs import config as C
from .records import Record, MINOR, ADULT_FPR, AMBIGUOUS, OTHER

REQUIRED = {"file", "age", "gender", "race"}


def _role(age_bin):
    if age_bin in C.FAIRFACE_MINOR_BINS:
        return MINOR
    if age_bin in C.FAIRFACE_AMBIGUOUS_BINS:
        return AMBIGUOUS
    if age_bin in C.FAIRFACE_ADULT_FPR_BINS:
        return ADULT_FPR
    return OTHER


def load_fairface_val(root=None, labels_csv=None, check_files=True):
    """Return (records, skipped). Reads labels only, never pixels."""
    root = Path(root or C.FAIRFACE_DIR)
    labels_csv = Path(labels_csv or (root / "fairface_label_val.csv"))
    df = pd.read_csv(labels_csv, dtype=str)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{labels_csv} missing columns {missing}")
    records, skipped = [], []
    for row in df.itertuples(index=False):
        bad = (row.age not in C.FAIRFACE_BIN_MIDPOINTS or row.race not in C.FAIRFACE_RACES)
        path = root / row.file
        if bad or (check_files and not path.exists()):
            skipped.append(row.file)
            continue
        role = _role(row.age)
        records.append(Record(
            id=Path(row.file).stem, dataset="fairface_val",
            age=C.FAIRFACE_BIN_MIDPOINTS[row.age], age_is_exact=False,
            age_bin=row.age, gender=str(row.gender).lower(), group=row.race,
            split="val", path=str(path), role=role,
            stratum=row.age if role == MINOR else None,
        ))
    return records, skipped
