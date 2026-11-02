"""Mitigation analysis (RQ5 / H5): per-group thresholds that equalise TPR.

All detectors expose score with  score > 0  <=>  flagged child at the default threshold.
Global operating point = pooled-minor TPR at the default threshold (unstratified).
For each group, t_g = the score threshold at which that group's minor TPR equals the global
target; we report the resulting FPR on that group's adults (the cost). Images with no score
(no face found) can never be flagged, so they cap the achievable TPR.
"""

import numpy as np


def equalised_thresholds(df):
    """Per-group threshold that brings each group's minor TPR to the pooled TPR.

    Returns, per group, the default TPR/FPR, the new threshold, and the TPR/FPR at that
    threshold. The FPR increase is the cost of equalising detection that way.
    """
    minors, adults = df[df.role == "minor"], df[df.role == "adult_fpr"]
    target = float(minors.is_child.astype(bool).mean())
    out = {"target_tpr": target, "groups": {}}
    for g in sorted(df.group.unique()):
        sm = minors[minors.group == g].score.astype(float).values
        sa = adults[adults.group == g].score.astype(float).values
        if len(sm) == 0:
            continue
        valid = sm[~np.isnan(sm)]
        max_tpr = len(valid) / len(sm)
        r = {"n_minor": int(len(sm)), "n_adult_fpr": int(len(sa)),
             "tpr_default": float(np.mean(np.nan_to_num(sm, nan=-np.inf) > 0)),
             "fpr_default": float(np.mean(np.nan_to_num(sa, nan=-np.inf) > 0)) if len(sa) else None,
             "max_achievable_tpr": max_tpr}
        if target > max_tpr or len(valid) == 0:
            r["threshold"], r["note"] = None, "target TPR not reachable (no-face images)"
        else:
            k = int(np.ceil(target * len(sm)))               # need k minors above threshold
            t = np.sort(valid)[::-1][k - 1] if k > 0 else np.inf
            t = float(np.nextafter(t, -np.inf))              # flag score > t  -> includes the k-th
            r["threshold"] = t
            r["tpr_equalised"] = float(np.mean(np.nan_to_num(sm, nan=-np.inf) > t))
            r["fpr_equalised"] = float(np.mean(np.nan_to_num(sa, nan=-np.inf) > t)) if len(sa) else None
        out["groups"][g] = r
    return out
