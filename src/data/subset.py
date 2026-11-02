"""Stratified dev subset (<= DEV_SUBSET_MAX images) for local smoke tests."""

import random
from collections import defaultdict

from configs import config as C


def dev_subset(records, n=None, seed=None):
    """Sample up to n records, balanced over (group, role, stratum) cells."""
    n = min(n or C.DEV_SUBSET_MAX, C.DEV_SUBSET_MAX)
    rng = random.Random(C.SEED if seed is None else seed)
    cells = defaultdict(list)
    for r in records:
        cells[(r.group, r.role, r.stratum)].append(r)
    for v in cells.values():
        rng.shuffle(v)
    out, keys = [], sorted(cells, key=str)
    while len(out) < n and any(cells[k] for k in keys):
        for k in keys:
            if cells[k] and len(out) < n:
                out.append(cells[k].pop())
    return out
