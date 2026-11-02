# CLAUDE.md: Child-Detection Skin-Tone Audit

## RULES FOR EVERY AGENT (read first, every session)
1. Read this file, then `EXPERIMENT_PLAN.md`. The plan is the spec. Don't change hypotheses, endpoints, groups, thresholds or prompts without Isaac's explicit OK; log any approved change in the Decision Log.
2. **Ethics rules in EXPERIMENT_PLAN.md §7 are absolute.** Never generate images of children, never use sexual/nudity content or classifiers, never scrape the web for images, never commit or upload images or per-image labels, never show children's images in figures. If a task seems to need any of this, stop and ask.
3. **Disk:** this Mac has ~6 GB free. Datasets and large weights live on Google Drive / Colab (`ENV='colab'`). Locally, use at most a ~500-image dev subset. Check `df -h ~` before any download over 200 MB and ask first.
4. **Preregistration first:** do not run any detector on the full evaluation sets until `PREREGISTRATION.md` is filled by Isaac and committed. Dev-subset smoke tests are fine.
5. **Phase gates:** finish one phase, write its report in `docs/`, then STOP and wait for Isaac.
6. **No fabricated numbers.** Every number in a report must come from a JSON in `results/` that stores its config and git commit. Mark anything unverified `(unverified)`.
7. **Log as you work:** append dated rows to the Session Log below after each meaningful action (file created, run finished, bug, decision). Record results with numbers, failed approaches and why. Don't restate what the code shows.
8. Put scratch files in the session scratchpad, not in this repo. Commit only when Isaac asks (no remote is set up; the repo should stay private if one is added).
9. Isaac is an undergrad ML researcher (PyTorch, CLIP probing, Re-ID audits, bootstrap stats). Explain decisions briefly and plainly.

## Project in one paragraph
Child-safety filtering of text-to-image training data relies on automated child detectors (Crețu et al., IEEE S&P 2026, arXiv 2512.05707). They never measured whether detectors miss darker-skinned children more often. We measure TPR/FPR, face-detection failure and age-estimation bias by perceived race (Phase 1: UTKFace, FairFace) and by annotated Monk skin tone (Phase 2: ICCWD), age-stratified with bootstrap CIs, then test mitigations. Target outcome: a pilot table for an internship pitch to Ana-Maria Crețu (CISPA), then a workshop paper.

## Environment
- Mac M2, 16 GB RAM, MPS available. Python via `uv` (create `.venv` with Python 3.11). Base conda is `/opt/miniconda3` (Python 3.13, torch 2.12). Don't install into it.
- Colab (GPU) for the VLM detector (D5) and full-dataset runs; Drive holds data and results, as in `~/federated-subset-scanning-pads` (`scripts/config.py` ENV pattern).

## Current status
- 2026-11-02: Phase 0 steps 1–3 done (env, config, loaders, counts script; tested on a synthetic fixture only). Sources pinned in `configs/config.py` SOURCES; Colab notebook `notebooks/phase0_data_prep.ipynb` ready. Sources approved. **Waiting on Isaac:** initial commit decision, then run the notebook on Colab.

## Open questions for Isaac
- No git commit exists yet → results record `NO_COMMIT_YET`; rule 6 needs a commit before real runs. Ask Isaac to allow an initial commit.
- Access to ICCWD (ask Crețu / authors) for Phase 2.
- Second annotator for MST labels (Phase 2).
- Approve Colab GPU time for D5; approve any paid API (Rekognition), default NO.

## Decision Log
| Date | Decision | Why |
|---|---|---|
| 2026-11-02 | Group variable = dataset perceived-race labels (Phase 1), human MST skin tone (Phase 2); never model-inferred labels as ground truth | Race can't be read from images; skin tone is what detectors see |
| 2026-11-02 | "No face/person detected" counts as NOT child | Matches how a filter behaves; failure reason logged separately |
| 2026-11-02 | FairFace 10–19 bin excluded from TPR | Straddles the 18 boundary |
| 2026-11-02 | FairFace minor strata = native bins 0–2 / 3–9 (no 0–4/5–9, no fallback) | FairFace only ships these bins; Isaac approved |
| 2026-11-02 | Cell < 50 check covers all tested groups (UTKFace 4, FairFace 7), not just White/Black | Isaac approved |
| 2026-11-02 | UTKFace source = `py97/UTKFace-Cropped` @e80f21fb (sha256 2c065539…); FairFace = `mutuma00/FairFace` @e8e670de | Official UTKFace link dead; Isaac approved both |
| 2026-11-02 | `.gitignore` allows `figures/*.png` | Charts only must be committable; Isaac approved |

## Session Log
| Date | Who | Action / result |
|---|---|---|
| 2026-11-02 | Claude (planning session) | Created repo, EXPERIMENT_PLAN.md v1.0, PREREGISTRATION.md template, CLAUDE.md, AGENT_KICKOFF.md |
| 2026-11-02 | Claude (Phase 0) | `uv sync` py3.11 venv via `pyproject.toml` + `uv.lock` (torch 2.15.0, open_clip 3.3.0; .venv 879 MB, disk 5.9→5.2 GB free). `results/env.json` written; MPS smoke test passes |
| 2026-11-02 | Claude (Phase 0) | Wrote `configs/config.py` (ENV via `CHILD_AUDIT_ENV`, prompts copied verbatim from plan §4), `src/data/` loaders (UTKFace from file names, FairFace from official val CSV), `src/provenance.py`, `scripts/run_phase0_counts.py`, `scripts/write_env.py` |
| 2026-11-02 | Claude (Phase 0) | Smoke test on synthetic fixture (empty files with fake names, scratchpad only): malformed UTKFace name skipped and counted, fallback strata triggered, dev subset capped at 500. No real data touched |
| 2026-11-02 | Claude (Phase 0) | Issue found: FairFace minor bins (0–2, 3–9) can't form plan strata 0–4/5–9 or fallback 0–9/10–17 → loader uses native bins; needs Isaac's OK. Also `.gitignore` `*.png` would block `figures/*.png` charts |
| 2026-11-02 | Claude (Phase 0) | FairFace source = Isaac's mirror `mutuma00/FairFace` @e8e670de; all 3 `0.25/` parquet LFS sha256 match `HuggingFaceM4/FairFace` → byte-identical. Val parquet 63 MB. Mirror is public (OK under CC BY 4.0) |
| 2026-11-02 | Claude (Phase 0) | Isaac couldn't find UTKFace. Surveyed HF mirrors: `py97/UTKFace-Cropped` @e80f21fb has original `UTKFace.tar.gz` (106.6 MB, sha256 2c065539…; card says official link dead, MIT tag is wrong). Parquet mirrors (`nu-delta/utkface` 23,705 rows with file_name; `vtsouval`, `rixmape`) drop or re-encode names → `nu-delta` used only as label cross-check |
| 2026-11-02 | Claude (Phase 0) | Wrote `notebooks/phase0_data_prep.ipynb` (pinned download + sha256 check → Drive `raw/`, unpack to `/content/data`, cross-check, counts, dev subset), `scripts/make_dev_subset.py` (250 UTK + 250 FF), `scripts/pack_repo_for_colab.sh`; provenance reads `GIT_COMMIT` stamp on Colab. Round trip tested on fixture only |
| 2026-11-02 | Claude (Phase 0) | Isaac approved sources, FairFace native strata, all-groups cell check, figures in git. Logged in Decision Log; `.gitignore` updated |
| 2026-11-02 | Claude (Phase 0) | Bug: `.gitignore` `data/` also ignored `src/data/` loaders → anchored to `/data/`. Initial commit approved by Isaac |
