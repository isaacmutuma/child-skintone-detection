# Phase 1 Report: Controlled pilot on face crops

**Status:** draft, 2026-11-03. Pilot on cropped, aligned faces, which are easier than real filtering conditions. This is a pilot, not the claim.
**Provenance:** analysis at commit `56c3fb5` (clean); detector runs at `74ae7b6` (D1, D3, D4) and `095dac0` (D5), all clean; preregistration frozen at `2e92b8b` before any full run. Sources: `results/phase1/primary_endpoint.json`, `results/phase1/metrics_<detector>_<dataset>.json`, `results/phase1/runs/*.json`. Groups are dataset-provided perceived-race labels.

## Primary endpoint: age-stratified TPR gap, White − Black (minors)

| Dataset | Detector | TPR White | TPR Black | Gap [95% CI] | Holm p | Unstrat. gap |
|---|---|---|---|---|---|---|
| UTKFace | D1 MiVOLO v2 | 0.940 | 0.856 | **0.084** [0.053, 0.119] | 0.0025 | 0.131 |
| UTKFace | D3 FairFace ViT | 0.864 | 0.824 | **0.040** [0.004, 0.078] | 0.016 | 0.070 |
| UTKFace | D4 CLIP B/32 | 0.975 | 0.877 | **0.098** [0.067, 0.132] | 0.0025 | 0.142 |
| UTKFace | D4 CLIP L/14 | 0.958 | 0.854 | **0.104** [0.072, 0.138] | 0.0025 | 0.158 |
| UTKFace | D5 Qwen2-VL-2B* | 0.921 | 0.860 | **0.061** [0.020, 0.103] | 0.014 | 0.093 |
| FairFace val | D1 MiVOLO v2 | 0.963 | 0.959 | 0.004 [−0.031, 0.035] | 1.0 | 0.005 |
| FairFace val | D4 CLIP B/32 | 0.973 | 0.976 | −0.003 [−0.031, 0.024] | 1.0 | −0.000 |
| FairFace val | D4 CLIP L/14 | 0.982 | 0.993 | −0.011 [−0.031, 0.007] | 1.0 | −0.010 |
| FairFace val | D5 Qwen2-VL-2B* | 0.952 | 0.979 | −0.027 [−0.071, 0.017] | 1.0 | −0.017 |

n minors White / Black: UTKFace 1,856 / 280; FairFace 240 / 281. *D5 on the preregistered stratified subsample (UTKFace 300 / 276, FairFace 154 / 135). Strata: UTKFace 0–9 / 10–17, FairFace 0–2 / 3–9 (FairFace has no minors above 9 that can be tested; 10–19 is excluded). Holm within each dataset; 0.0025 = 5 × the permutation floor (1/2,001).

**H1 (≥ half of detectors, CI excludes 0):** supported on UTKFace (5/5). Not supported on FairFace (0/4): no detectable gap. The FairFace gap CIs exclude effects larger than about 0.035 in the White > Black direction.

## Three findings
1. **On UTKFace, every detector misses Black minors more often than White minors**, by 4 to 10 points after age stratification. The gap sits mostly in the 10–17 stratum: TPR White vs Black is 0.807 vs 0.563 (D1), 0.925 vs 0.659 (CLIP B/32), 0.859 vs 0.556 (CLIP L/14), 0.800 vs 0.579 (D5). In the 0–9 stratum the differences are 0.000 to 0.034. After adjusting for age, brightness and blur (plus face size for D1), the Black vs White odds ratio for being flagged stays below 1 for D1 (0.32 [0.20, 0.51]), CLIP B/32 (0.19 [0.12, 0.32]), CLIP L/14 (0.19 [0.12, 0.30]) and D5 (0.53 [0.30, 0.94]). It does not for D3 (0.81 [0.52, 1.24]).
2. **The mechanism looks like age over-estimation, not missed faces (H2b yes, H2a no).** Among detected minors, MiVOLO over-estimates Black children's age more than White children's: by +1.52 years [0.95, 2.28] on UTKFace and by +1.23 [0.83, 1.63] on FairFace (bin midpoints, approximate). The same holds for D3 on UTKFace (+0.72 [0.14, 1.43]). All three have permutation p ≤ 0.009. Face-detection failure is not higher for Black minors. On UTKFace, MiVOLO found a face in all 280 Black minor images (White failure rate 0.006). The Black − White difference is −0.006 [−0.011, 0.009] on UTKFace and 0.005 [−0.031, 0.039] on FairFace (Newcombe score CIs).
3. **Fixing the gap with per-group thresholds is costly (H5).** Raising Black TPR to the global operating point on UTKFace raises the share of Black adults (20–59) wrongly flagged as children: D1 0.005 → 0.074, CLIP B/32 0.043 → 0.560, CLIP L/14 0.008 → 0.297, D5 0.025 → 0.220 (D3, not a preregistered H5 detector: 0.022 → 0.126). Adult FPR already differs by group without any threshold change (H3, χ² p < 0.05, uncorrected) in 6/9 runs. On FairFace, Black adults are flagged more often than White adults: D1 0.078 vs 0.018, CLIP B/32 0.102 vs 0.062, CLIP L/14 0.037 vs 0.018.

## Caveats
- **Pilot conditions:** cropped, aligned faces; perceived-race labels, not skin tone (Phase 2 uses MST).
- **The two datasets test different ages.** The UTKFace gap is concentrated in 10–17, an age range FairFace cannot test, because its minors are 0–9 only and the 10–19 bin is excluded. So the FairFace null result is not evidence against a teenage gap. This reading was not preregistered; treat it as interpretation.
- **Small cells:** only 280 Black minors in UTKFace, 126 of them aged 10–17. UTKFace ages and race labels are known to be noisy.
- **Model substitutions:** D3 is a FairFace-trained ViT substitute (the original weights are gone). Its expected-age rule over-estimates age in every group (+4.5 to +5.4 years), so its absolute TPR is low. MiVOLO v2's training data is partly proprietary, so overlap with UTKFace can't be ruled out. D5 ran on a subsample, which gives wider CIs.
- **Post-hoc change (logged in PREREGISTRATION.md Deviations):** face-fail CIs are stratified Wilson intervals, and the gap uses Newcombe's interval. The preregistered bootstrap collapsed to [0, 0] for groups with zero failures, which made the UTKFace gap CI wrongly exclude 0 (permutation p = 0.29 for D1, 0.66 for D3). The bootstrap CIs are still stored as `*_boot`; no other number changed.
- **Analysis limits:**
  - The D1 FairFace logit did not converge (quasi-separation; ORs up to 10⁹), so it can't be interpreted.
  - For D1 on FairFace, the target TPR is out of reach for White, Black and Latino_Hispanic, because no-face images can never be flagged.
