# How to run this project with a separate agent

## Start
In a new terminal:
```
cd ~/child-detection-skin-tone-audit
claude
```
Then paste the kickoff prompt below. The agent picks up its rules from `CLAUDE.md` automatically.

## Kickoff prompt (Phase 0)
> Read CLAUDE.md and EXPERIMENT_PLAN.md fully. Then do Phase 0 only:
> 1. Create a `uv` venv (Python 3.11) with torch (MPS), open_clip, pandas, numpy, scipy, scikit-learn, matplotlib, pillow. Write `results/env.json`.
> 2. Write `configs/config.py` with ENV local/colab, paths, seed, CLIP prompts and VLM prompt from the plan.
> 3. Write the dataset loaders in `src/data/` for UTKFace and FairFace-val (unified record format from §8), plus `scripts/run_phase0_counts.py`.
> 4. Before downloading anything, tell me the sources, licences and sizes you plan to use and wait for my OK. Disk is nearly full, so data goes to Google Drive.
> 5. After I approve and the data is in place, produce `results/phase0_counts.json` and a power check, write `docs/DATA_CARD.md`, update the Session Log, then STOP.

## Later phases (one prompt each, after reviewing the previous report)
- **Phase 1:** "Preregistration is committed at <hash>. Run Phase 1 per EXPERIMENT_PLAN.md §6: implement D1, D3, D4 locally on the dev subset first, show me the smoke-test output, then build the Colab notebook for full runs + D5. Then run the analysis and write docs/PHASE1_REPORT.md. STOP after the report."
- **Phase 2:** "We have ICCWD access via <source>. Build the MST annotation app (§6 Phase 2) and stop for annotation."

## What to check in each report
- Every number points to a `results/*.json` with a git commit
- CIs are reported, and per-group sample sizes are shown
- Caveats: cropped faces ≠ in-the-wild; perceived race labels are noisy
- No images of children anywhere in figures or docs

## How it connects
- Pitch target: Ana-Maria Crețu (CISPA). Email draft A2 in `~/Research-Applications/programs/cispa-csr/outreach.md`; brief in `project-brief-child-detection-bias.md`.
- Add the Phase 1 table to the email only if the protocol was followed.
