# Data Card: Phase 1 datasets

Sources: `results/data_provenance.json`. Counts: `results/phase0_counts.json`. Both were made on Colab at commit `40491cb` on 2026-11-02 (recorded from the `GIT_COMMIT` stamp in the Drive zip). Images and per-image labels are kept only on Google Drive and Colab. Neither file contains any image or per-image record.

## Sources

| | UTKFace (aligned & cropped) | FairFace (validation, 0.25 padding) |
|---|---|---|
| Source | HF `py97/UTKFace-Cropped` @ `e80f21fb…` (official link is dead; the mirror holds the original `UTKFace.tar.gz`) | HF `mutuma00/FairFace` @ `e8e670de…`, a byte-identical copy of `HuggingFaceM4/FairFace` |
| File | `UTKFace.tar.gz`, 106,634,631 B | `0.25/validation-…951dbd63c8724ee1.parquet`, 63,189,799 B |
| SHA256 (verified on download) | `2c0655397b498e81b82d685ee74bee8c2e4364a5897caf3afec6ea48929024a7` | `d0cfac2888356b3d3fa3d837bc30961d8c860934e4c5df862e8cb87812bfa137` |
| Licence | Non-commercial research only (original UTKFace terms; the mirror's MIT tag is wrong) | CC BY 4.0 |
| Images | 23,708 files; 3 have malformed names and are skipped → **23,705 records** | **10,954 rows** |
| Labels | Exact age, gender, race (5), read from file names | Age bin (9), gender, race (7) |
| Group variable | Dataset-provided perceived race labels, used as given | Same |

**UTKFace cross-check:** we compared file names with the independent mirror `nu-delta/utkface` (23,705 rows). 23,704 names match. 4 appear only in ours (this includes the 3 malformed names) and 1 appears only in the mirror. The labels are encoded in the file names, so matching names mean matching labels.

**FairFace file names:** the Hugging Face copy has no original file names, so rows are written as `val/1.jpg` … `val/10954.jpg` in parquet order. It is unverified whether these numbers match the official file names. No analysis depends on this.

## Analysis roles
- **Minor (TPR):** UTKFace age 0–17; FairFace bins 0–2 and 3–9.
- **Adult (FPR):** UTKFace age 20–59; FairFace bins 20–29 to 50–59.
- **Ambiguous:** the FairFace 10–19 bin, which straddles 18. It is reported separately and left out of TPR.
- **Other:** ages 18–19 and 60+. Not used in the primary metrics.

## Counts

### UTKFace minors by group × age stratum
| Group | 0–4 | 5–9 | 10–14 | 15–17 | **Minors** | Adults 20–59 |
|---|---|---|---|---|---|---|
| White | 603 | 503 | 390 | 360 | **1,856** | 6,218 |
| Black | 86 | 68 | **43** | 83 | **280** | 3,845 |
| Asian | 798 | 131 | **43** | **23** | **995** | 2,128 |
| Indian | 382 | 107 | 54 | **48** | **591** | 3,104 |
| Others (not tested) | 298 | 86 | 59 | 68 | 511 | 1,129 |

Bold cells are below 50. Under the pre-registered rule this triggers the **fallback strata 0–9 / 10–17**. Every fallback cell is at least 50:

| Group | 0–9 | 10–17 |
|---|---|---|
| White | 1,106 | 750 |
| Black | 154 | 126 |
| Asian | 929 | 66 |
| Indian | 489 | 102 |

Reference weights (pooled minors): 0–9 = 0.7234, 10–17 = 0.2766.

### FairFace-val minors by group × age bin
| Group | 0–2 | 3–9 | **Minors** | 10–19 (excl.) | Adults 20–59 |
|---|---|---|---|---|---|
| White | 44 | 196 | **240** | 167 | 1,581 |
| Black | 25 | 256 | **281** | 203 | 1,020 |
| Latino_Hispanic | 19 | 178 | **197** | 215 | 1,162 |
| East Asian | 49 | 220 | **269** | 145 | 1,092 |
| Southeast Asian | 34 | 207 | **241** | 180 | 940 |
| Indian | 18 | 183 | **201** | 186 | 1,060 |
| Middle Eastern | 10 | 116 | **126** | 85 | 924 |

**Every 0–2 cell is below 50 (10–49).** The UTKFace-style fallback can't be built from FairFace's bins. Merging the two minor bins (0–2 + 3–9) gives a single 0–9 stratum, which is just the unstratified estimate. Reference weights (pooled minors): 0–2 = 0.128, 3–9 = 0.872.

## Power check
Assumes TPR = 0.85 in both groups, two-sided α = 0.05, 80% power, and a normal approximation. **No Holm correction**, so the true minimum detectable effect is larger once we correct across detectors. Stratified results use the reference weights above.

| Dataset | Group | n minors | 95% CI half-width (stratified) |
|---|---|---|---|
| UTKFace | White | 1,856 | ±0.0168 |
| UTKFace | Black | 280 | ±0.0443 |
| UTKFace | Asian | 995 | ±0.0290 |
| UTKFace | Indian | 591 | ±0.0299 |
| FairFace | White | 240 | ±0.0456 |
| FairFace | Black | 281 | ±0.0421 |
| FairFace | Middle Eastern (smallest) | 126 | ±0.0633 |

**Minimum detectable White − Black TPR gap (primary contrast):**
- **UTKFace:** 0.0677 stratified (0.0641 unstratified)
- **FairFace:** 0.0888 stratified (0.0879 unstratified)

So gaps of about 7 points (UTKFace) and 9 points (FairFace) are detectable before Holm correction. This is close to the plan's estimate of 7–8 points. Black minors in UTKFace (n = 280) are the limiting cell.

## Caveats
- **Age mix differs a lot by group, which is why stratification matters.** In UTKFace, 798 of 995 Asian minors are aged 0–4, versus 86 of 280 Black minors and 603 of 1,856 White minors. An unstratified TPR would partly measure age, not group.
- These are cropped, aligned faces, which is easier than real filtering conditions. Phase 1 is a pilot.
- Perceived race labels are noisy annotator judgements, not skin tone. UTKFace ages are known to contain some label noise.
- FairFace ages are bins, so signed age error on FairFace uses bin midpoints and is approximate.
- D3 (the FairFace classifier) must never be evaluated on FairFace, because it was trained on it.
