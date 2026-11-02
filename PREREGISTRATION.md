# Preregistration (fill in, then commit BEFORE any full-dataset detector run)
Commit hash after freezing: `2e92b8bef6bfe1d146f46d56a052dbc70893c758`  ·  Date frozen: `2026-11-02`  ·  Frozen by: Isaac Mutuma

## Hypotheses (from EXPERIMENT_PLAN.md §1, edit only if you disagree)
- H1: For at least half of detectors D1–D5, TPR for the darkest group < TPR for the lightest group (age-stratified), with the 95% bootstrap CI of the gap excluding 0.
- H2a: Face-detection failure rate is higher for the darkest group than the lightest.
- H2b: Among detected minors, mean signed age error (pred − true) is larger (more over-estimation) for the darkest group.
- H3: FPR on adults 20–59 differs across groups (two-sided).
- H4 (Phase 2): The H1 gap direction holds on ICCWD with MST bins.
- H5: Per-group thresholds can equalise TPR. We report the FPR cost; no directional claim.

## Primary endpoint
Age-stratified TPR gap (lightest − darkest group = White − Black), per detector and dataset, with a 2,000-sample bootstrap 95% CI (resampling images within group) and a permutation test. Holm-corrected across detectors **within each dataset**: UTKFace family = D1, D3, D4-B/32, D4-L/14, D5 (5 tests); FairFace family = D1, D4-B/32, D4-L/14, D5 (4 tests; D3 is never run on FairFace because it was trained on it).

**Strata (fixed after the Phase 0 counts, before any detector output; see `results/phase0_counts.json`, `docs/DATA_CARD.md`):**
- UTKFace: the pre-specified fallback **0–9 / 10–17** applies. Primary cells below 50 were Black 10–14 = 43, Asian 10–14 = 43, Asian 15–17 = 23 and Indian 15–17 = 48; every fallback cell is ≥ 50. Weights = pooled minors: 0–9 = 0.7234, 10–17 = 0.2766.
- FairFace-val: native minor bins **0–2 / 3–9**. Every 0–2 cell is 10–49, but the < 50 merge rule is waived for FairFace, because merging the two bins would remove stratification altogether. Weights = pooled minors: 0–2 = 0.128, 3–9 = 0.872. The 10–19 bin is excluded from TPR and reported separately.
- The unstratified gap is reported as a secondary result for both datasets.

**Minimum detectable White − Black gap** (80% power, α = 0.05, TPR = 0.85, before Holm): UTKFace 0.0677, FairFace 0.0888.

## Fixed analysis choices
- Groups, Phase 1 UTKFace: White, Black, Asian, Indian (Others reported, not tested). Lightest/darkest for the primary endpoint: White vs Black.
- Groups, Phase 1 FairFace: all 7 reported. Primary: White vs Black.
- Groups, Phase 2: MST bins light 1–3, medium 4–7, dark 8–10.
- Child rule: min estimated age < 18; no face/person found = not child.
- CLIP prompts (`configs/config.py`): child = {"a photo of a child", "a photo of a kid", "a photo of a baby", "a photo of a teenager"}; adult = {"a photo of an adult", "a photo of a man", "a photo of a woman"}. Flagged as child if max child-prompt cosine similarity > max adult-prompt similarity; the raw margin is logged. Models: OpenCLIP ViT-B-32 and ViT-L-14, OpenAI weights.
- VLM prompt: "Is there a child (a person under 18 years old) in this image? Answer yes or no."
- Bootstrap: 2,000 resamples, seed 0. α = 0.05.
- Detectors, Phase 1: D1 MiVOLO (face) on UTKFace + FairFace-val; D3 FairFace classifier on UTKFace only; D4 CLIP (2 models) on both; D5 open VLM on a group × stratum-stratified subsample of ≤ 2,000 images per dataset (Colab). Model versions and weight hashes are pinned in `results/env.json` before the full runs.
- FPR population: adults 20–59 (UTKFace exact ages; FairFace bins 20–29 to 50–59).
- Signed age error: UTKFace exact ages; FairFace bin midpoints (approximate, reported as secondary).
- Data: UTKFace (`py97/UTKFace-Cropped` @ e80f21fb, sha256 2c065539…) 23,705 records; FairFace val (`mutuma00/FairFace` @ e8e670de, sha256 d0cfac28…) 10,954 rows. See `results/data_provenance.json`.

## What would count as a null result
All primary-endpoint CIs include 0 → report "no detectable gap at this sample size" with the minimum detectable effect (UTKFace 0.0677, FairFace 0.0888 before Holm). Null results get reported. A significant gap in the opposite direction (Black > White TPR) is also reported as a finding.

## Deviations (log any after freezing)
| Date | Deviation | Reason |
|---|---|---|
| 2026-11-02 | Clarification (before any D1/D3 output): D1 = `iitolstykh/mivolo_v2` @e8bdd8ae, face-only input (body crop empty), faces found by YOLOv8x person_face @b3d071ae (conf 0.4, IoU 0.7, the detector card's defaults); age_est = min over detected faces; child if < 18; no face = not child | Prereg named "MiVOLO" without a checkpoint. MiVOLO's UTKFace-trained checkpoints are excluded (train/test leakage); v2 is the current model and is trained on LAGENDA + other data |
| 2026-11-02 | Clarification (before any D3 output): D3 age estimate = Σ P(bin) × bin midpoint over FairFace's 9 age bins (70+ → 75); child if < 18; faces from dlib CNN detector + 5-point alignment, as in the FairFace repo; min over faces; no face = not child | Prereg rule "estimated age < 18" did not say how to handle a bin classifier whose 10–19 bin straddles 18 |
| 2026-11-02 | Deviation (before any D3 output): D3 model replaced by `dima806/fairface_age_image_detection` @4e02ab80 (ViT-B/16 fine-tuned on FairFace, same 9 age bins; model.safetensors sha256 1265d8e5…). dlib detection + 5-point chips (size 300, padding 0.25) and the expected-age < 18 rule are unchanged; still UTKFace only | The original dchen236/FairFace ResNet-34 weights folder returns 404 and no verified mirror exists. Isaac chose this substitute over dropping D3 |
