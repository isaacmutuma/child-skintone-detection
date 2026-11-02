"""Confounder sensitivity analysis (EXPERIMENT_PLAN.md §5):
logistic regression on minors  flagged ~ group + age + log(face_size) + brightness + log(blur)
face_size (face_area_frac) only exists for detectors with a face stage (D1, D3); those models
are fit on face-found minors, because no-face images have no face size.
Reports odds ratios vs the reference group (White) with 95% CIs.
"""

import numpy as np
import pandas as pd


def confounder_logit(df, reference="White"):
    """Fit the confounder logit on the minors in `df` (one detector x dataset).

    Returns odds ratios with 95% CIs for every term. An odds ratio below 1 for a group
    means its minors are flagged less often than `reference` minors with the same age,
    brightness, blur (and face size). Returns {"error": ...} if the fit fails.
    """
    import statsmodels.formula.api as smf
    m = df[df.role == "minor"].copy()
    m["flagged"] = m.is_child.astype(bool).astype(int)
    m["log_blur"] = np.log(m.blur_lapvar.clip(lower=1e-6))
    terms = ["age", "brightness", "log_blur"]
    note = None
    if "face_area_frac" in m and m.face_area_frac.notna().any():
        m = m[m.face_area_frac.notna()]
        m["log_face_size"] = np.log(m.face_area_frac.clip(lower=1e-6))
        terms.append("log_face_size")
        note = "fit on minors with a detected face (face size undefined otherwise)"
    m = m.dropna(subset=["age", "brightness", "log_blur"])
    if m.flagged.nunique() < 2:
        return {"error": "outcome has no variation", "n": int(len(m))}
    formula = f"flagged ~ C(group, Treatment(reference='{reference}')) + " + " + ".join(terms)
    try:
        fit = smf.logit(formula, data=m).fit(disp=0, maxiter=200)
    except Exception as e:   # e.g. perfect separation
        return {"error": f"{type(e).__name__}: {e}", "n": int(len(m)), "formula": formula}
    conf = fit.conf_int()
    rows = {}
    for name in fit.params.index:
        rows[name] = {"or": float(np.exp(fit.params[name])),
                      "or_ci": [float(np.exp(conf.loc[name, 0])), float(np.exp(conf.loc[name, 1]))],
                      "p": float(fit.pvalues[name])}
    return {"formula": formula, "n": int(fit.nobs), "converged": bool(fit.mle_retvals.get("converged", True)),
            "pseudo_r2": float(fit.prsquared), "terms": rows, "note": note}
