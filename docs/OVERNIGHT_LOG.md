# Overnight log: Phase 1 completion

Follows `docs/OVERNIGHT_RUNBOOK.md`. Times are UTC.

| Time | Step | Result |
|---|---|---|
| 2026-11-02 13:54 | 1. Pending run | `mivolo_face__fairface_val__full.json` landed on Drive. Checked by Claude (main session) via the Drive connector: n_images 10,954, git.dirty false, commit 74ae7b6, prereg 2e92b8b, no_face 851, n_flagged_child 2,640, 1,266 s. Role counts match Phase 0 (minor 1,555, adult_fpr 7,779) |
| 2026-11-03 | 2. Confirm 7/7 | All 7 `__full.json` files are on Drive (`results/phase1/runs`). **Next: step 3 (D5).** |
| 2026-11-03 | Isaac | Overnight agent cancelled (already morning). Isaac running phase13_vlm.ipynb (D5 smoke + gate + subsample) manually; Claude follows up in this session |
| 2026-11-02 22:39 | 3. D5 gate | `d5_gate_dev.json`: **PASS**. UTKFace dev yes/no mass mean 0.994 (min 0.979), minors 0.84 vs adults 0.05; FairFace dev 0.994 (min 0.972), 0.98 vs 0.04; 0 errors |
| 2026-11-02 22:44 | 3. D5 subsample | `vlm__utkface__full_sub.json`: n 1,844 (1,044 minors + 800 adults), flagged 923, 0 fails, 80 s. `vlm__fairface_val__full_sub.json`: n 1,949 (969 + 980), flagged 989, 0 fails, 110 s. Commit 095dac0, dirty false. Matches the approved design. **Next: step 4 (phase14 analysis)** |
