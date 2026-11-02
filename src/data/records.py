"""Unified record format shared by all datasets (EXPERIMENT_PLAN.md §8)."""

from dataclasses import dataclass, asdict
from typing import Optional

from configs import config as C

# role of an image in the analysis
MINOR = "minor"            # counts toward TPR
ADULT_FPR = "adult_fpr"    # adult 20-59, counts toward FPR
AMBIGUOUS = "ambiguous"    # FairFace 10-19: reported separately, not in TPR
OTHER = "other"            # 18-19 or 60+: not used in primary metrics


@dataclass(frozen=True)
class Record:
    """One image with its dataset labels, in the format every loader returns."""

    id: str                     # dataset-unique id (file name stem); never committed
    dataset: str                # 'utkface' | 'fairface_val'
    age: Optional[float]        # exact age (UTKFace) or bin midpoint (FairFace)
    age_is_exact: bool
    age_bin: Optional[str]      # FairFace native bin, else None
    gender: Optional[str]
    group: str                  # dataset-provided perceived race label
    split: str
    path: str
    role: str                   # MINOR | ADULT_FPR | AMBIGUOUS | OTHER
    stratum: Optional[str]      # age stratum for minors, else None

    def to_dict(self):
        """Plain dict of all fields."""
        return asdict(self)


def stratum_for_age(age, strata=None):
    """Label (e.g. "0-4") of the age stratum containing `age`, or None if outside all."""
    strata = strata or C.AGE_STRATA
    for lo, hi in strata:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    return None


def role_for_exact_age(age):
    """Analysis role from an exact age: minor (< 18), adult_fpr (20-59) or other."""
    if age <= C.MINOR_MAX_AGE:
        return MINOR
    lo, hi = C.ADULT_FPR_RANGE
    if lo <= age <= hi:
        return ADULT_FPR
    return OTHER
