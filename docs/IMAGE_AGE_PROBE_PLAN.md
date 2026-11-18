# Image-side age probe of CLIP: experiment plan

**Status:** exploratory side experiment, not part of the preregistered study (H1–H5). Written on 2026-11-18, before any probe output existed. Changes after the first run are listed at the end.

## Question
When zero-shot CLIP reads a **face image**, does it judge Black minors as older than White minors of the same true age? In other words, does CLIP itself show the age over-estimation (H2b) that MiVOLO and the FairFace classifier showed in Phase 1?

## Why
- **The gap in the evidence.** Phase 1 found CLIP's largest detection gap (ViT-L/14: 10.0 points; teenagers 83% vs 55%), and the text probe placed the cause in the image encoder. But D4 only compares child and adult prompts, so the audit never measured CLIP's age error. H2b was tested only for the two age estimators (MiVOLO +1.52 years, FairFace classifier +0.72 on UTKFace).
- **Why the course project's number is not enough.** Model A, an age model trained on CLIP ViT-L/14 image embeddings, over-estimates Black minors by +1.90 years more than White minors. But model A learns from UTKFace's own age labels, which are known to be noisy, so part of that gap could come from the labels rather than from CLIP. This probe uses no training and no labels: any gap comes from CLIP's representation itself.
- **What it adds to a paper.** If CLIP shows the shift on its own, over-estimation appears in three independent model families (MiVOLO, a FairFace-trained ViT, CLIP), measured the same way. That answers "is this a CLIP quirk, or a quirk of model A's training labels?"

## What it does and does not do
- Reads stored CLIP image embeddings; **no images are generated, and none are shown.** Ethics rules unchanged.
- No model is trained or fine-tuned.
- Per-image outputs (one implied age per image) stay on Drive, never in git. Only aggregate numbers are committed.

## Data and model
- **Primary:** UTKFace, all 23,705 records, CLIP **ViT-L/14** with QuickGELU. The image embeddings already exist on Drive from the course project (`robust_child_filtering_model/embeddings/utkface__clip_vitl14.npz`, sha256 `4a6f66c1…`). They reproduced the audit's Phase 1 L/14 scores to within 3.1e-07 with identical decisions, so they are the audit's CLIP. The script checks the sha256 before use.
- **Secondary:** CLIP **ViT-B/32** on UTKFace. This needs new embeddings (about 10 minutes on Colab), extracted with the audit's pinned weights and the same QuickGELU check.
- **Optional replication:** FairFace validation set with ViT-L/14, using bin midpoints as MiVOLO did. It can only test children under ten. It needs new embeddings (about 10 minutes).

## Method
1. **Implied age (same ruler as the text probe's M2).** Score each image embedding against 80 anchors, "a photo of a {N}-year-old person" (N = 1…80), with no race word. Turn the scores into probabilities with CLIP's own logit scale (softmax) and take the expected age.
2. **Robustness versions, fixed now:**
   - **Mode age:** the single most likely N. Unlike the expected age, it is not pulled toward the middle of the 1–80 range when CLIP is uncertain.
   - **Uncertainty check:** the entropy of the age distribution, by group. If CLIP is simply less certain about Black faces, a spread-out distribution would raise the expected age even without a real shift. The mode version and this check catch that case.
   - **Template sensitivity:** repeat with two more anchor wordings, "a close-up photo of the face of a {N}-year-old person" and "a portrait of a {N}-year-old person". Reported only.
3. **Analysis: the audit's own H2b code, unchanged.** Feed the implied age into `src/analysis/metrics.py` as `age_est`, with D4's own decision as `is_child` and every face counted as found:
   - **Primary:** Black − White signed age error (implied − true) for minors, age-stratified 0–9 / 10–17 with the Phase 0 weights 0.7234 / 0.2766. 95% bootstrap CI (2,000 within-group resamples, seed 0) and stratified permutation p, exactly as for MiVOLO and the FairFace classifier.
   - **Secondary:** the same for teenagers 10–17 only; for every group (Asian, Indian against White); and for adults 20–59. The adult result shows whether CLIP reads Black faces as older at every age (a general shift) or mainly at the 18 boundary.
   - **Calibration (descriptive):** mean implied age by true-age band, pooled over groups, so readers can see how coarse CLIP's ruler is.

## Decision rule (fixed before running)
CLIP shows **image-side age over-estimation of Black minors** if both hold on UTKFace with ViT-L/14:
1. The primary Black − White stratified signed error (expected age) is above 0 and its 95% CI excludes 0.
2. The mode-age version has the same sign.

If (1) holds but (2) does not, the shift is reported as possibly driven by uncertainty (the entropy check), not as over-estimation. The size is reported next to MiVOLO's +1.52 and model A's +1.90 years. There is no multiple-testing family: this is one exploratory contrast, labelled as such.

**What follows:**
- **Shift found:** the "not a CLIP quirk" argument rests on three model families with the same measure. The report and the doc add a CLIP row to the H2b table, marked exploratory.
- **No shift:** CLIP's detection gap would come from something other than a uniform age shift, for example weaker child-versus-adult separation for Black teenagers (consistent with the threshold result: 27–58% of Black adults flagged when equalising). Model A's +1.90 would then be more likely to come from UTKFace's labels. Either answer is informative.

## Limits
- Zero-shot implied age is a coarse ruler. On text it read high on ViT-B/32 but was well calibrated on ViT-L/14. Group comparisons at the same true age cancel a shared offset, but not a group-specific one, which is why the mode and entropy checks are fixed in advance.
- Anchors stop at 80. This doesn't affect minors or adults 20–59.
- Perceived-race labels, cropped faces and UTKFace's noisy labels: same limits as Phase 1.

## Outputs
- `scripts/image_age_probe.py` (code). It reads the embeddings, checks their sha256, encodes the anchors with the pinned weights and QuickGELU, and writes per-image implied ages to Drive only.
- `results/image_age_probe/image_age_probe_<arch>_<dataset>.json` (aggregate numbers, config, git state, embedding sha256).
- Runs on Colab CPU for L/14 on UTKFace (no image is re-read); B/32 and FairFace need a T4 for about 10 minutes each.

## Changes after the first run
(none yet)
