"""Group metrics with age stratification, bootstrap CIs and permutation tests
(EXPERIMENT_PLAN.md §5, PREREGISTRATION.md).

Conventions
- Rates are computed per (group, stratum) and combined with FIXED reference weights
  (pooled minors of the full dataset, from results/phase0_counts.json), so a group with
  older children is not penalised. Unstratified rates are reported as secondary.
- Bootstrap: resample images with replacement within each group (2,000 draws, seed 0),
  implemented with multinomial counts; percentile 95% CI.
- Permutation test for a two-group gap: shuffle group labels within each stratum
  (stratified permutation), two-sided p = (#{|perm| >= |obs|} + 1) / (P + 1).
"""

import json

import numpy as np
import pandas as pd

from configs import config as C

UTK_STRATA = ["0-9", "10-17"]          # fallback strata (PREREGISTRATION.md)
FF_STRATA = ["0-2", "3-9"]             # native bins (PREREGISTRATION.md)


# ── setup ────────────────────────────────────────────────────────────────────
def reference_weights(dataset, counts_path=None):
    """Pooled-minor stratum weights from the frozen Phase 0 counts (all groups)."""
    d = json.loads((counts_path or C.RESULTS_DIR / "phase0_counts.json").read_text())["datasets"]
    if dataset == "utkface":
        table = d["utkface"]["minors_by_group_stratum_fallback"]
    else:
        table = d["fairface_val"]["minors_by_group_stratum"]
    pooled = {}
    for row in table.values():
        for s, n in row.items():
            pooled[s] = pooled.get(s, 0) + n
    tot = sum(pooled.values())
    return {s: n / tot for s, n in pooled.items()}


def analysis_stratum(df, dataset):
    """Primary-analysis stratum per row (minors only; None otherwise)."""
    if dataset == "utkface":
        return np.where(df.age <= 9, "0-9", "10-17")
    return df.stratum.values


def strata_for(dataset):
    """Stratum labels used for `dataset` (UTKFace 0-9/10-17, FairFace 0-2/3-9)."""
    return UTK_STRATA if dataset == "utkface" else FF_STRATA


# ── point estimates ──────────────────────────────────────────────────────────
def stratified_rate(values, strata, weights, mask=None):
    """sum_s w_s * mean(values | stratum s). NaN if any weighted stratum is empty."""
    values, strata = np.asarray(values, float), np.asarray(strata)
    if mask is not None:
        values, strata = values[mask], strata[mask]
    out = 0.0
    for s, w in weights.items():
        m = strata == s
        if w > 0 and not m.any():
            return float("nan")
        if m.any():
            out += w * values[m].mean()
    return out


# ── bootstrap ────────────────────────────────────────────────────────────────
def _boot_counts(n, B, rng):
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B) if n else np.zeros((B, 0))


def boot_stratified(values, strata, weights, B, rng):
    """B bootstrap replicates of a stratified mean (resampling within this group)."""
    values, strata = np.asarray(values, float), np.asarray(strata)
    n = len(values)
    if n == 0:
        return np.full(B, np.nan)
    W = _boot_counts(n, B, rng).astype(float)            # B x n resample counts
    est = np.zeros(B)
    for s, w in weights.items():
        m = (strata == s).astype(float)
        num, den = W @ (values * m), W @ m
        with np.errstate(invalid="ignore", divide="ignore"):
            est += w * np.where(den > 0, num / den, np.nan)
    return est


def boot_mean(values, B, rng):
    """B bootstrap replicates of a plain (unstratified) mean."""
    values = np.asarray(values, float)
    if len(values) == 0:
        return np.full(B, np.nan)
    W = _boot_counts(len(values), B, rng).astype(float)
    return (W @ values) / len(values)


def ci(reps, alpha=None):
    """Percentile (1 - alpha) interval of bootstrap replicates, ignoring NaNs."""
    a = (alpha or C.ALPHA) / 2
    reps = np.asarray(reps)[~np.isnan(reps)]
    if len(reps) == 0:
        return [None, None]
    return [float(np.quantile(reps, a)), float(np.quantile(reps, 1 - a))]


# ── score intervals (face-detection failure; Isaac, 2026-11-03) ──────────────
# The percentile bootstrap collapses to [0, 0] when a group has zero failures, so
# face-fail CIs use a stratified Wilson interval and the gap uses Newcombe's hybrid
# score interval (Newcombe 1998, method 10). Bootstrap CIs are kept as *_boot.
def effective_n(strata, weights):
    """Kish-type effective n of a stratified proportion: 1 / sum_s w_s^2 / n_s."""
    strata = np.asarray(strata)
    tot = 0.0
    for s, w in weights.items():
        n_s = int((strata == s).sum())
        if w > 0:
            if n_s == 0:
                return float("nan")
            tot += w * w / n_s
    return 1.0 / tot if tot > 0 else float("nan")


def wilson(p, n, alpha=None):
    """Wilson score interval for proportion p with (effective) sample size n."""
    from scipy.stats import norm
    if p is None or n is None or np.isnan(p) or np.isnan(n) or n <= 0:
        return [None, None]
    z = norm.ppf(1 - (alpha or C.ALPHA) / 2)
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = 0.0 if p <= 0 else max(0.0, centre - half)     # exact bounds at p = 0 / 1
    hi = 1.0 if p >= 1 else min(1.0, centre + half)
    return [float(lo), float(hi)]


def newcombe_gap(p1, ci1, p2, ci2):
    """Newcombe hybrid score CI for p1 - p2 from the two Wilson intervals."""
    if None in (*ci1, *ci2):
        return [None, None]
    (l1, u1), (l2, u2) = ci1, ci2
    d = p1 - p2
    return [float(d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)),
            float(d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))]


def stratified_wilson(values, strata, weights):
    """Stratified rate of 0/1 `values` and its Wilson CI at the effective n."""
    p = stratified_rate(values, strata, weights)
    return p, wilson(p, effective_n(strata, weights))


# ── permutation test ─────────────────────────────────────────────────────────
def perm_test_gap(vals_a, strata_a, vals_b, strata_b, weights, P, rng):
    """Two-sided stratified permutation p for stratified(a) - stratified(b)."""
    va, sa = np.asarray(vals_a, float), np.asarray(strata_a)
    vb, sb = np.asarray(vals_b, float), np.asarray(strata_b)
    obs = stratified_rate(va, sa, weights) - stratified_rate(vb, sb, weights)
    if np.isnan(obs):
        return float("nan"), float("nan")
    perm = np.zeros(P)
    for s, w in weights.items():
        a, b = va[sa == s], vb[sb == s]
        pool, na = np.concatenate([a, b]), len(a)
        if len(pool) == 0 or w == 0:
            continue
        order = rng.random((P, len(pool))).argsort(axis=1)     # P random permutations
        shuffled = pool[order]
        ma = shuffled[:, :na].mean(axis=1) if na else np.zeros(P)
        mb = shuffled[:, na:].mean(axis=1) if len(pool) - na else np.zeros(P)
        perm += w * (ma - mb)
    p = (np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (P + 1)
    return float(obs), float(p)


def holm(pvals, family_size=None):
    """Holm step-down adjusted p-values. Missing family members count as p = 1."""
    keys = list(pvals)
    m = family_size or len(keys)
    ps = [(pvals[k] if pvals[k] is not None and not np.isnan(pvals[k]) else 1.0) for k in keys]
    ps += [1.0] * (m - len(ps))
    order = np.argsort(ps)
    adj, running = np.zeros(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return {k: float(adj[i]) for i, k in enumerate(keys)}


# ── per detector x dataset ───────────────────────────────────────────────────
def group_metrics(df, dataset, weights, B, rng):
    """Per-group TPR / FPR / face-detection / signed age error with bootstrap CIs."""
    df = df.copy()
    df["astratum"] = analysis_stratum(df, dataset)
    flagged = df.is_child.astype(bool).astype(float)
    has_face_stage = df.face_found.notna().any()
    has_age = df.age_est.notna().any()
    out = {}
    for g, gdf in df.groupby("group"):
        minors, adults = gdf[gdf.role == "minor"], gdf[gdf.role == "adult_fpr"]
        fm = flagged.loc[minors.index].values
        r = {"n_minor": int(len(minors)), "n_adult_fpr": int(len(adults)),
             "n_minor_by_stratum": minors.astratum.value_counts().to_dict()}
        r["tpr_strat"] = stratified_rate(fm, minors.astratum, weights)
        r["tpr_strat_ci"] = ci(boot_stratified(fm, minors.astratum, weights, B, rng))
        r["tpr_unstrat"] = float(fm.mean()) if len(fm) else None
        r["tpr_unstrat_ci"] = ci(boot_mean(fm, B, rng))
        r["tpr_by_stratum"] = {s: float(fm[minors.astratum.values == s].mean())
                               for s in weights if (minors.astratum.values == s).any()}
        fa = flagged.loc[adults.index].values
        r["fpr"] = float(fa.mean()) if len(fa) else None
        r["fpr_ci"] = ci(boot_mean(fa, B, rng))
        if has_face_stage:
            fail = (~minors.face_found.astype(bool)).astype(float).values
            r["face_fail_minor_strat"], r["face_fail_minor_strat_ci"] = \
                stratified_wilson(fail, minors.astratum, weights)
            r["face_fail_minor_strat_ci_method"] = "wilson, stratified (Kish n_eff)"
            r["face_fail_minor_n_eff"] = effective_n(minors.astratum, weights)
            # kept so the RNG stream (and every other CI) is unchanged
            r["face_fail_minor_strat_ci_boot"] = ci(boot_stratified(fail, minors.astratum, weights, B, rng))
            r["face_fail_all"] = float((~gdf.face_found.astype(bool)).mean())
        if has_age:
            det = minors[minors.age_est.notna()]
            err = (det.age_est - det.age).values
            r["n_minor_age_est"] = int(len(det))
            r["signed_age_err_strat"] = stratified_rate(err, det.astratum, weights)
            r["signed_age_err_strat_ci"] = ci(boot_stratified(err, det.astratum, weights, B, rng))
            r["signed_age_err_by_stratum"] = {s: float(err[det.astratum.values == s].mean())
                                              for s in weights if (det.astratum.values == s).any()}
        out[g] = r
    return out


def contrast(df, dataset, weights, a, b, B, P, rng):
    """Primary contrast a - b: TPR gap (H1), face-fail gap (H2a), age-error gap (H2b)."""
    df = df.copy()
    df["astratum"] = analysis_stratum(df, dataset)
    flagged = df.is_child.astype(bool).astype(float)
    ma = df[(df.group == a) & (df.role == "minor")]
    mb = df[(df.group == b) & (df.role == "minor")]
    res = {"a": a, "b": b}

    def gap_block(va, sa, vb, sb):
        """Stratified gap a - b with permutation p and bootstrap CI."""
        obs, p = perm_test_gap(va, sa, vb, sb, weights, P, rng)
        reps = (boot_stratified(va, sa, weights, B, rng) - boot_stratified(vb, sb, weights, B, rng))
        return {"gap": obs, "ci": ci(reps), "p_perm": p}

    res["tpr_gap"] = gap_block(flagged.loc[ma.index].values, ma.astratum.values,
                               flagged.loc[mb.index].values, mb.astratum.values)
    ua, ub = flagged.loc[ma.index].values, flagged.loc[mb.index].values
    res["tpr_gap_unstrat"] = {"gap": float(ua.mean() - ub.mean()),
                              "ci": ci(boot_mean(ua, B, rng) - boot_mean(ub, B, rng))}
    if df.face_found.notna().any():
        fa = (~ma.face_found.astype(bool)).astype(float).values
        fb = (~mb.face_found.astype(bool)).astype(float).values
        # H2a direction: darker (b) fails more -> report b - a
        blk = gap_block(fb, mb.astratum.values, fa, ma.astratum.values)
        pb, cib = stratified_wilson(fb, mb.astratum.values, weights)
        pa, cia = stratified_wilson(fa, ma.astratum.values, weights)
        blk["ci_boot"] = blk["ci"]
        blk["ci"] = newcombe_gap(pb, cib, pa, cia)
        blk["ci_method"] = "newcombe hybrid score (stratified wilson)"
        res["face_fail_gap_b_minus_a"] = blk
    if df.age_est.notna().any():
        da, db = ma[ma.age_est.notna()], mb[mb.age_est.notna()]
        ea, eb = (da.age_est - da.age).values, (db.age_est - db.age).values
        # H2b direction: darker (b) over-estimated more -> report b - a
        res["age_err_gap_b_minus_a"] = gap_block(eb, db.astratum.values, ea, da.astratum.values)
    return res


def fpr_independence(df):
    """H3: chi-square test that FPR (adults 20-59) is equal across groups."""
    from scipy.stats import chi2_contingency
    ad = df[df.role == "adult_fpr"]
    tab = pd.crosstab(ad.group, ad.is_child.astype(bool))
    if tab.shape[1] < 2:
        return {"chi2": None, "p": None, "note": "no variation in flags among adults"}
    chi2, p, dof, _ = chi2_contingency(tab.values)
    return {"chi2": float(chi2), "dof": int(dof), "p": float(p)}
