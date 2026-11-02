# Overnight runbook: Phase 1 completion (2026-11-02 → 11-03)

For a Claude Code session started with `claude --chrome` in this repo and run under `/loop`.
Read `CLAUDE.md` (rules) and this file first. Every rule in CLAUDE.md still applies.
Isaac reads the result in the morning: `docs/OVERNIGHT_LOG.md` + `docs/PHASE1_REPORT.md`.

## State at hand-off (2026-11-02, ~13:45 UTC)
- Preregistration frozen at `2e92b8b`. All clarifications and deviations are in its Deviations table.
- Full runs on Drive (`MyDrive/child-detection-audit/results/phase1/runs/`): 6 of 7 done.
  **Pending: `mivolo_face__fairface_val__full.json`**, running in Isaac's open Colab tab
  (`phase12_full_runs.ipynb`). It takes about 20 min.
- D5 (`phase13_vlm.ipynb`) has not been run yet. Isaac **pre-authorised** the subsample run IF the
  automatic gate passes: mean yes/no mass ≥ 0.9 and minors flagged > adults, on both datasets,
  with no errors.
- Analysis code is tested on synthetic data (`tests/test_analysis_synthetic.py` passes).

## Tools
- **Google Drive connector**: poll for files (`search_files`) and read small aggregate JSONs
  (`read_file_content`). Folder ids: project root `1y003pKDHTbtRcvB6AbhiemzH-8UY87xL`,
  `results/phase1/runs` `1SdpHvyNNuEMCGrIhpKhs_P33kOWyHK7O`.
- **Claude in Chrome**: drive Colab (colab.research.google.com, signed in as isaacmutuma@pusan.ac.kr).
  Open notebooks via File → Open notebook → GitHub → `isaacmutuma/child-skintone-detection`.

## Steps (do them in order, and log each one to `docs/OVERNIGHT_LOG.md` with a UTC time)
1. **Watch the pending run.** Poll Drive every ~10 min for `mivolo_face__fairface_val__full.json`.
   - If it appears, read it and check `n_images == 10954` and `git.dirty == false`. Log n, flagged and fail_reasons.
   - If it hasn't appeared after 60 min, look at the Colab tab's last cell output in Chrome.
     - If it errored or disconnected: reconnect with a T4 GPU runtime and **Run all**. The notebook is
       resumable and skips the 6 finished runs.
     - Retry at most twice.
   - **Do not disconnect Isaac's running runtime while the run is still going.**
2. **Confirm 7/7.** Check that all 7 `__full.json` files are on Drive.
3. **D5.** Free the GPU first (Runtime → Disconnect and delete runtime on the phase 12 tab). Then
   open `notebooks/phase13_vlm.ipynb`, set a T4 runtime, and **Run all**.
   - Wait for `results/phase1/d5_gate_dev.json` on Drive and read it.
   - If the gate passed, wait for `vlm__utkface__full_sub.json` and `vlm__fairface_val__full_sub.json`.
     This takes about 1–2 h. Use a long polling interval (~20 min).
   - If the gate failed, record why and continue without D5. The analysis then counts D5 as p = 1, which is conservative.
4. **Analysis.** Open `notebooks/phase14_analysis.ipynb` on a runtime (CPU is fine) and **Run all**.
   - Wait for `MyDrive/child-detection-audit/phase1_outputs.zip`.
   - Copy the printed primary-endpoint table into the log.
5. **Bring outputs to the Mac.**
   - Download `phase1_outputs.zip` from Drive in Chrome. It lands in `~/Downloads`.
   - Verify that it contains only `results/phase1/*.json`, `results/phase1/runs/*.json` and
     `figures/phase1_*.png`.
   - Unzip it into the repo root, run `git status`, and confirm no per-image file was added.
   - Check one run JSON's `git.commit` against `git log`.
6. **Report.** Write `docs/PHASE1_REPORT.md` (about 1 page):
   - the table of White−Black TPR gaps per detector × dataset (estimate, 95% CI, p_perm, p_holm, n);
   - H1 count per dataset;
   - 3 bullet findings;
   - H2a/H2b/H3/H5 headline numbers;
   - the confounder-regression OR for Black vs White;
   - caveats.

   **Every number must be copied from a JSON in `results/phase1/`, citing the file.** Mark
   anything else `(unverified)`. Required caveats:
   - cropped, aligned faces are not in-the-wild images; this is a pilot;
   - perceived-race labels are noisy, and race is not skin tone;
   - D3 is a substitute model;
   - D5 ran on a subsample, so its MDE is larger;
   - FairFace ages are bins.

   Null results get reported as nulls, with the minimum detectable effect (`results/phase0_counts.json`).
7. **Wrap up.** Append to the CLAUDE.md Session Log. Finish `docs/OVERNIGHT_LOG.md` with:
   - what ran;
   - what failed and why;
   - the headline table;
   - what Isaac needs to decide.

   **Do not commit or push** (Isaac reviews first). End the loop (ScheduleWakeup `stop: true`).

## Stop conditions (log, end the loop, wait for Isaac)
- The same step fails after 2 retries, or a page doesn't respond.
- A CAPTCHA, a login prompt, a "Colab Pro / usage limit" dialog, or any payment prompt.
- Any assert in a notebook fails: n_images, dirty, sha256, or the consistency checks in `run_analysis.py`.
- Anything that seems to need viewing an image, or would change the preregistered analysis.

## Ethics (absolute; see EXPERIMENT_PLAN.md §7)
- Never open image files or image folders in Drive, Colab's file browser or anywhere else. Never display images.
- Never open per-image CSVs. Read only aggregate JSONs.
- Don't copy per-image data to the Mac except via the zip above, which contains none.
- Don't commit, push, share, or change Drive sharing settings.
