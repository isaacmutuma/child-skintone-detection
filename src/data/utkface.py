"""UTKFace (aligned & cropped) loader.

File names encode labels: [age]_[gender]_[race]_[datetime].jpg(.chip.jpg)
  gender: 0 male, 1 female; race: 0 White, 1 Black, 2 Asian, 3 Indian, 4 Others.
A few upstream files have malformed names (e.g. missing race); they are skipped
and counted in `skipped`, never guessed.
"""

import re
from pathlib import Path

from configs import config as C
from .records import Record, role_for_exact_age, stratum_for_age, MINOR

_NAME = re.compile(r"^(\d+)_([01])_([0-4])_(\d+)")
_GENDER = {0: "male", 1: "female"}


def load_utkface(root=None, strata=None):
    """Return (records, skipped_names). Reads file names only, never pixels."""
    root = Path(root or C.UTKFACE_DIR)
    if not root.is_dir():
        raise FileNotFoundError(f"UTKFace folder not found: {root}")
    records, skipped = [], []
    for p in sorted(root.iterdir()):
        if not p.name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        m = _NAME.match(p.name)
        if not m:
            skipped.append(p.name)
            continue
        age, g, r = int(m[1]), int(m[2]), int(m[3])
        role = role_for_exact_age(age)
        records.append(Record(
            id=p.name.split(".")[0], dataset="utkface",
            age=float(age), age_is_exact=True, age_bin=None,
            gender=_GENDER[g], group=C.UTKFACE_RACES[r], split="all",
            path=str(p), role=role,
            stratum=stratum_for_age(age, strata) if role == MINOR else None,
        ))
    return records, skipped
