# Preregistration (fill in, then commit BEFORE any full-dataset detector run)
Commit hash after freezing: `________`  ·  Date frozen: `________`  ·  Frozen by: Isaac Mutuma

## Hypotheses (from EXPERIMENT_PLAN.md §1, edit only if you disagree)
- H1: For at least half of detectors D1–D5, TPR for the darkest group < TPR for the lightest group (age-stratified), with the 95% bootstrap CI of the gap excluding 0.
- H2a: Face-detection failure rate is higher for the darkest group than the lightest.
- H2b: Among detected minors, mean signed age error (pred − true) is larger (more over-estimation) for the darkest group.
- H3: FPR on adults 20–59 differs across groups (two-sided).
- H4 (Phase 2): The H1 gap direction holds on ICCWD with MST bins.
- H5: Per-group thresholds can equalise TPR. We report the FPR cost; no directional claim.

## Primary endpoint
Age-stratified TPR gap (lightest − darkest group), per detector, Holm-corrected across detectors. Strata 0–4 / 5–9 / 10–14 / 15–17, weighted by pooled minor age distribution. Fallback if any cell < 50: strata 0–9 / 10–17.

## Fixed analysis choices
- Groups, Phase 1 UTKFace: White, Black, Asian, Indian (Others reported, not tested). Lightest/darkest for the primary endpoint: White vs Black.
- Groups, Phase 1 FairFace: all 7 reported. Primary: White vs Black.
- Groups, Phase 2: MST bins light 1–3, medium 4–7, dark 8–10.
- Child rule: min estimated age < 18; no face/person found = not child.
- CLIP prompts: (copy from configs/config.py once written) ________
- VLM prompt: "Is there a child (a person under 18 years old) in this image? Answer yes or no."
- Bootstrap: 2,000 resamples, seed 0. α = 0.05.

## What would count as a null result
All primary-endpoint CIs include 0 → report "no detectable gap at this sample size" with the minimum detectable effect. Null results get reported.

## Deviations (log any after freezing)
| Date | Deviation | Reason |
|---|---|---|
