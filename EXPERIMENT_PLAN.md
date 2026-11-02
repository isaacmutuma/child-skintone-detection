# Experiment Plan: Skin-Tone Disparities in Automated Child Detection
*Does child-safety filtering miss darker-skinned children?*
Version 1.0 · 2026-11-02 · Owner: Isaac Mutuma · Intended host: Ana-Maria Crețu (CISPA)

---

## 0. Background (read once)
- Text-to-image (T2I) developers filter images of children out of training data so models can't be misused to make child sexual abuse material (CSAM). Filtering relies on **automated child detectors**.
- Crețu et al., *Evaluating Concept Filtering Defenses against CSAM Generation by T2I Models*, IEEE S&P 2026 (arXiv 2512.05707; code github.com/spring-epfl/t2i-child-filtering):
  - benchmarked 20+ detectors;
  - the best removes ~94% of children (75.3% TPR on the harder ICCWD set);
  - **no breakdown by skin tone**;
  - their agenda warns that "unequal filtering rates for sub-categories of a concept introduce a bias in the distribution of remaining images."
- **Gap we test:** if detectors miss darker-skinned children more, then (a) filtering protects them less, and (b) filtered data and models are skewed.

## 1. Research questions and hypotheses
Freeze these in `PREREGISTRATION.md` (commit it) **before** looking at any detector output.

| ID | Question | Hypothesis | Test |
|---|---|---|---|
| RQ1 | Do child detectors have lower **TPR** (minors flagged as minors) for darker skin tones? | H1: TPR(darkest group) < TPR(lightest group) for ≥ half of detectors | Per-detector TPR gap, bootstrap 95% CI, age-stratified |
| RQ2 | Where does the gap come from? | H2a: face-detection failure rate is higher for darker skin. H2b: for detected minors, age is **over-estimated** more for darker skin ("adultification") | Face-detection rate per group; signed age error (pred − true) per group |
| RQ3 | Do detectors over-flag **adults** as children unequally (FPR)? | H3: FPR differs across groups (direction not pre-specified) | Per-group FPR on adults 20–59 |
| RQ4 | Does the gap survive in realistic, in-the-wild images and with captions? | H4: gap persists on ICCWD; caption-based detectors show their own, different gap | Phase 2 annotated ICCWD |
| RQ5 | Can it be mitigated? | H5: per-group thresholds or image+caption ensembles shrink the gap; report the FPR cost | Equal-TPR threshold analysis |
| RQ6 (stretch) | Does the skew carry into filtered datasets/models? | H6: residual child images after filtering are enriched for darker skin | Composition of what each filter leaves behind |

**Primary endpoint** (decided in advance): the TPR gap between the darkest and lightest skin-tone groups, averaged over age strata 0–4 / 5–9 / 10–14 / 15–17, for each detector. Everything else is secondary.

## 2. Group variable: skin tone, not race
- **Phase 1 (controlled pilot):** datasets ship **perceived race** labels (UTKFace: White/Black/Asian/Indian/Others; FairFace: 7 groups). Use them *as given* and report them as "dataset-provided perceived race labels". Do not re-infer race.
- **Phase 2 (main study):** **human-annotated perceived skin tone, Monk Skin Tone (MST) scale 1–10**, collapsed to 3 bins for analysis: light (1–3), medium (4–7), dark (8–10). Collapse rule is fixed now.
- Never use a model's skin-tone or race prediction as ground truth. Model-estimated skin tone (e.g. ITA from face pixels) may be reported only as a secondary sensitivity check.

## 3. Datasets (only these; nothing else)
| Phase | Dataset | What | Minors? | Labels | Access / licence (verify before download) |
|---|---|---|---|---|---|
| 1 | **UTKFace** (aligned & cropped) | ~23k faces | exact age 0–116 → minors = 0–17 | age, gender, race (5) | Non-commercial research. Official site or a verified mirror; record source + checksum |
| 1 | **FairFace** (validation split) | ~10.9k faces, race-balanced | age bins 0–2, 3–9 = minors; **10–19 is ambiguous → excluded from TPR**, reported separately | age bin, gender, race (7) | CC BY 4.0 (HF `HuggingFaceM4/FairFace` or official repo) |
| 2 | **ICCWD**: Image-Caption Children in the Wild (Kireev, Crețu et al. 2025, arXiv 2506.10117) | 10k image–caption pairs, child yes/no | in-the-wild, incl. partial bodies | child label (+ our MST annotation) | Check release terms; may need to request from authors, ideally via Crețu |
| 2 (opt.) | Crețu et al. CC3M-10k / LAION-Face-2k labelled subsets | as in the S&P paper | yes | child label | Only via the authors |

**Size check (do first, report in `results/phase0_counts.json`):** number of minors per group per age stratum. If any primary cell has < 50 images, merge strata (pre-specified fallback: 0–9 / 10–17) and note it.

**Power (approximate):** with ~300 minors per group and TPR ≈ 0.85, the 95% CI half-width per group is ≈ ±0.04, so gaps of ≳ 7–8 points are detectable. The agent should recompute this from the real counts.

## 4. Detectors
Each detector maps image (± caption) → {child, not child} (+ age estimate when available). Use the paper's decision rules where they exist: **child if min estimated age over detected people < 18**; "no face/person detected" = **not child** (that's what a filter would do). Log the failure reason.

| # | Detector | Type | Runs where | Notes |
|---|---|---|---|---|
| D1 | **MiVOLO face** (github.com/WildChlamydia/MiVOLO; YOLOv8 person/face detector + MiVOLO age) | face age estimator | Mac MPS or Colab | The paper's SoTA face-based estimator. Keep detector stage separate so face-detection failure can be logged |
| D2 | **MiVOLO face+body** | face+body age estimator | same | Matters in Phase 2 (partial bodies) |
| D3 | **FairFace classifier** (github.com/dchen236/FairFace) | face age-bin classifier | Mac/Colab | **Never evaluate on FairFace** (trained on it). UTKFace + ICCWD only |
| D4 | **CLIP zero-shot** ViT-B/32 and ViT-L/14 (OpenCLIP/OpenAI weights) | image–text similarity | Mac | Prompts fixed in advance, e.g. {"a photo of a child", "a photo of a kid", "a photo of a baby", "a photo of a teenager"} vs {"a photo of an adult", "a photo of a man", "a photo of a woman"}; child if max child-prompt sim > max adult-prompt sim. Also log raw margin |
| D5 | **Open VLM yes/no** (e.g. LLaVA-1.5-7B or Qwen2-VL-2B/7B) | VQA | **Colab GPU only** (too big for this disk) | Prompt fixed in advance: "Is there a child (a person under 18 years old) in this image? Answer yes or no." Use P(yes) for threshold analysis |
| D6 (Phase 2) | **Caption keyword filter** | text | Mac | Reuse the paper's keyword lists from their repo (SUBSTR/SUBWORD variants) |
| D7 (Phase 2, opt.) | **Caption LLM filter** | text | Colab or API | Same prompt template as the paper if available |
| (opt.) | Amazon Rekognition | commercial | API ($) | Only if Isaac approves cost |

Pin every model version and weight hash in `results/env.json`.

## 5. Metrics and statistics
- **TPR_g** = P(flagged child | minor, group g). **FPR_g** = P(flagged child | adult 20–59, group g).
- **Face-detection rate_g** (D1–D3). **Signed age error_g** = mean(pred − true) for detected minors (UTKFace exact ages; FairFace bin midpoints flagged as approximate).
- **Gap** = TPR_lightest − TPR_darkest (also every pairwise gap vs. the best group).
- **Age stratification:** compute within strata 0–4, 5–9, 10–14, 15–17. Report the stratum-weighted average using a common reference age distribution (the pooled minors), so groups with older children aren't penalised. Also report the unstratified gap.
- **Uncertainty:** 2,000-sample bootstrap 95% CIs (resample images within group). Permutation test for the gap. Holm correction across detectors for the primary endpoint.
- **Confounders to log per image:** resolution, brightness/contrast, face size, blur (Laplacian variance). Sensitivity analysis: logistic regression `flagged ~ group + age + log(face_size) + brightness + blur` per detector.
- **Mitigation (RQ5):** for score-based detectors (D4, D5, MiVOLO age), find the per-group threshold that equalises TPR at the global operating point. Report the resulting FPR per group, and the gain from OR-ensembling image + caption detectors.

## 6. Phases, outputs and stop points
Each phase ends with a short write-up in `docs/` and a **STOP: Isaac reviews** before the next phase.

### Phase 0: Setup (≤ ½ day)
- Env via `uv` (Python 3.11), torch with MPS. `configs/config.py` with `ENV in {'local','colab'}`, as in the PADS repo.
- Data lives on Google Drive / Colab, **not this Mac** (≈ 6 GB free). Locally, keep at most a 500-image dev subset.
- Download UTKFace + FairFace-val to Drive. Record source URL, licence, SHA256 in `docs/DATA_CARD.md`.
- Output: `results/phase0_counts.json` (minors per group × stratum), `results/env.json`.
- **STOP:** Isaac fills and commits `PREREGISTRATION.md`.

### Phase 1: Controlled pilot on faces (1–3 days)
- Run D1, D3 (UTKFace only), D4 on UTKFace + FairFace-val; D5 on a stratified subsample (≤ 2,000 images) on Colab.
- Outputs:
  - `results/phase1/metrics_<detector>_<dataset>.json` (all metrics + CIs + config + git commit)
  - `figures/phase1_tpr_by_group.png`, `phase1_age_error_by_group.png`, `phase1_fdr_by_group.png`
  - `docs/PHASE1_REPORT.md`, 1 page: table + 3 bullet findings + caveats
- **Caveat to always state:** cropped, aligned faces are easier than real filtering conditions. This is a pilot, not the claim.
- **STOP.** This table goes into the email to Crețu (only if the protocol was followed; null results are fine to report honestly).

### Phase 2: In-the-wild main study (needs data access; ideally as the internship)
- Get ICCWD. Annotate **MST** on all child-positive images (and a matched random sample of adult-only images for FPR).
  - **Two independent annotators** (Isaac + one other). Annotation tool: `src/annotation/` local HTML/Flask app showing one image at a time with the MST swatch strip, keyboard 1–0, "can't tell" option, no other metadata shown.
  - Report Cohen's weighted κ. Disagreements > 2 MST points get adjudicated; "can't tell" is excluded and counted.
- Run D1–D7 with the paper's thresholds. Same metrics; add caption-detector gaps and image+caption ensemble.
- Output: `docs/PHASE2_REPORT.md`.

### Phase 3 (stretch, only with host approval): downstream effect
- **3a (cheap):** skin-tone composition of the child images each filter *leaves behind* vs. the original child distribution.
- **3b (expensive, host lab only):** with the authors' filtered SD 1.x checkpoints/code, measure per-group difficulty (queries until success) for the benign proxy "child wearing glasses", e.g. prompts specifying skin tone. **Do not run 3b outside a supervised, ethics-approved setting.**

## 7. Ethics and safety rules (non-negotiable)
1. **No CSAM, ever.** No sexual or nudity-related prompts, data or classifiers. The only generation proxy is "child wearing glasses", and only in Phase 3b with the host.
2. **Only the datasets in §3.** No web scraping or new collection of images of children.
3. **No images, per-image labels or model outputs tied to image IDs leave this machine or Drive.** Git ignores them (see `.gitignore`). Only aggregate metrics are committed.
4. **No identification of individuals.** Skin tone is annotated as perceived appearance only. No names, faces in figures, or re-identification.
5. **Results are for improving protection.** Do not write "how to evade filters" guidance. Disclosure of significant gaps is coordinated with the host lab / dataset authors.
6. Phase 2 annotation of a third-party dataset needs the dataset owners' permission and, as a project, the host's ethics/IRB process. Confirm before starting.
7. Figures show **charts only**, never example images of children.

## 8. Repo layout
```
child-detection-skin-tone-audit/
├── CLAUDE.md               # rules + session log for the agent
├── EXPERIMENT_PLAN.md      # this file
├── PREREGISTRATION.md      # frozen hypotheses (commit before results)
├── configs/config.py       # ENV, paths, seeds, thresholds, prompts
├── src/data/               # dataset loaders → unified records {id, age, age_is_exact, group, split, path}
├── src/detectors/          # one module per detector, common interface predict(batch) -> {is_child, age_est, score, face_found}
├── src/analysis/           # metrics, bootstrap, stratification, regression, threshold analysis
├── src/annotation/         # MST annotation app (Phase 2)
├── scripts/                # run_phase0_counts.py, run_detector.py, run_analysis.py, make_figures.py
├── notebooks/              # Colab runner (Phase 1 D5, Phase 2)
├── results/                # JSON metrics only (each with config + git commit)
├── figures/                # charts only
└── docs/                   # DATA_CARD.md, PHASE*_REPORT.md
```

## 9. Definition of done (Phase 1)
- [ ] `PREREGISTRATION.md` committed before any detector output exists
- [ ] Counts table; every primary cell ≥ 50 or the fallback strata applied
- [ ] D1, D3, D4 on UTKFace; D1, D4 on FairFace-val; D5 on a stratified subsample
- [ ] TPR / FPR / face-detection / signed age error by group, age-stratified, with bootstrap CIs
- [ ] Confounder regression per detector
- [ ] `docs/PHASE1_REPORT.md` with honest caveats; no claims beyond the data
- [ ] Every number in the report traceable to a JSON in `results/` with its git commit
