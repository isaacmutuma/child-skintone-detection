# Text-side probe of CLIP's text encoder: experiment plan

**Status:** exploratory side experiment, not part of the preregistered study (H1–H5). Written on 2026-11-10, before any probe output existed. Changes after the first run are listed at the end.

## Question
Does CLIP's **text encoder** place race-marked descriptions of young people closer to "adult" than the same descriptions with a different race word? For example, does "a photo of a Black teenager" sit further toward the adult side than "a photo of a White teenager"?

## Why
- **Where to focus.** Phase 1 found that every detector misses Black teenagers on UTKFace. The CLIP detector (D4) uses race-neutral prompts ("a photo of a child", "... a teenager", against "... an adult", "... a man", "... a woman"), so its gap must come from how CLIP embeds the **face image**. This probe tests whether the **text** side carries a similar age shift. If it does not, the course project's focus on the image side (models A–F) is right. If it does, text matters for caption-based filtering (Phase 2) and for text-to-image models that use this encoder (Stable Diffusion 1.x uses CLIP ViT-L/14's text encoder).
- **A claim to check.** A separate conversation asserted, without evidence, that CLIP's text vector for "Black teenager" sits closer to "adult". This probe tests that directly.

## What it does and does not do
- **Text only.** CLIP's text encoder turns each sentence into a vector. Nothing is drawn: **no images are generated, loaded or shown.** Ethics rule 1 (no generated images of children) is untouched.
- Prompts are descriptions, not people. Results describe the text encoder, not any population.

## Model
- CLIP **ViT-B/32**, OpenAI weights, from the local file `~/.cache/clip/ViT-B-32.pt`, sha256 `40d36571…` (matches OpenAI's published hash, checked before loading). These are the same OpenAI weights D4 used in Phase 1, in OpenAI's original file format.
- CLIP **ViT-L/14** (the Stable Diffusion 1.x text encoder and D4's second model) needs a 1.7 GB download. The Mac has under 1 GB free, so L/14 runs on Colab later with the same script (`--arch ViT-L-14`).

## Prompt grid
`{template}` × `{modifier}` × `{age noun}`:
- **Templates (6):** "a photo of {x}", "a close-up photo of the face of {x}", "a portrait of {x}", "a picture of {x}", "an image of {x}", "a photograph of {x}". The article ("a" or "an") is chosen to fit the next word.
- **Modifiers (10):**
  - none: no modifier, the baseline
  - race words, the UTKFace groups: White, Black, Asian, Indian
  - extra race and skin wording: African American, light-skinned, dark-skinned. "Black" and "White" are also colour words, so these check whether results depend on that double meaning.
  - neutral controls: tall, smiling. These measure how much **any** extra word moves a prompt.
- **Age nouns (16):**
  - minor (11): child, kid, boy, girl, teenager, teenage boy, teenage girl, 13-year-old, 15-year-old, 16-year-old, 17-year-old
  - adult (5): adult, man, woman, 25-year-old, 40-year-old

That gives 6 × 10 × 16 = 960 prompts.

## Metrics
1. **Detector margin (M1).** For each prompt q: max cosine similarity to D4's four child prompts minus max similarity to D4's three adult prompts. This is D4's exact decision rule, applied to text instead of an image. Above 0 means "child" side.
2. **Implied age (M2).** Similarity of q to 80 anchors "a photo of a {N}-year-old person" (N = 1…80). These are turned into probabilities with CLIP's own logit scale (softmax), then averaged into an expected age. The anchors contain no race word.

## Analysis
- **Paired differences.** For every template × age noun (one "item"), compute Δ = metric(modifier) − metric(White). The main contrast is **Black − White** on minor nouns (66 items). All other modifiers are reported too, plus dark-skinned − light-skinned.
- **Uncertainty.** Mean Δ with a bootstrap 95% interval over items (2,000 resamples, seed 0). Items share templates and words, so they are not independent. Intervals are descriptive, not tests about people.
- **Scale.** Δ M1 is also expressed as a fraction of the **child–adult distance**: the mean M1 of "teenager" minus the mean M1 of "adult", unmodified, over the 6 templates.
- **Control spread.** The largest |mean Δ| of the neutral controls (tall, smiling) against "none".

## Decision rule (fixed before running)
A **material text-side shift toward adult** for Black minors means all three of these hold for M1 on minor nouns:
1. The mean Black − White Δ is below 0 and its 95% interval excludes 0.
2. |mean Δ| ≥ 10% of the child–adult distance.
3. |mean Δ| is larger than the control spread.

The 10% threshold is a judgement call, set before seeing data. M2 (implied age) is reported as supporting evidence and does not decide the result.

**What follows:**
- **No material shift:** keep the course focus on the image side (models A–F as planned). The text encoder is not a priority.
- **Material shift:** add text-side effects to the Phase 2 plan (caption-based CLIP filtering) and to the course write-up. It still would not explain D4's Phase 1 gap, because D4's prompts contain no race word. The effect on text-to-image models cannot be tested here, because that would need generated images of children.

## Outputs
- `scripts/text_probe.py` (code), `results/text_probe/text_probe_<arch>.json` (aggregate numbers + config + git state).
- No per-image data exists in this experiment. The JSON holds per-prompt aggregates only.

## Changes after the first run
None to the design. One fix to the loader call (open_clip 3.3.0 has no `jit` argument).

**2026-11-10, activation bug.** The first Colab run (`--weights hf-pinned`, both models) loaded OpenAI's weights with nn.GELU instead of QuickGELU. Its cross-check failed (B/32 child–adult distance 0.1079 against 0.1387 locally), which is how the bug was found. Those two Colab results are invalid. The script now forces QuickGELU and refuses to run otherwise. The local B/32 result below was always correct, because OpenAI's original file loads with QuickGELU. ViT-L/14 is rerun in `notebooks/phase15_clip_quickgelu.ipynb`.

## First result: ViT-B/32 (2026-11-10)
Source: `results/text_probe/text_probe_ViT-B-32.json`. The script was uncommitted when it ran (`dirty: true`), on top of commit ef0c8d8.

**Decision: no material text-side shift.** None of the three criteria is met.

| Contrast (minor nouns, 66 items) | M1 mean Δ [95% CI] | Fraction of child–adult distance | M2 mean Δ, years [95% CI] |
|---|---|---|---|
| Black − White | −0.0017 [−0.0048, 0.0013] | −0.01 | −3.60 [−4.87, −2.38] |
| African American − White | −0.0119 [−0.0156, −0.0082] | −0.09 | −1.88 [−3.38, −0.55] |
| dark-skinned − light-skinned | −0.0019 [−0.0029, −0.0008] | −0.01 | +0.45 [−0.15, 1.07] |
| tall − none (control) | −0.0136 [−0.0181, −0.0093] | −0.10 | +4.69 [3.25, 6.19] |

Child–adult distance on M1: 0.1387. Control spread on M1: 0.0136.

What the numbers show:
- **Black vs White:** the margin difference is about 1% of the child–adult distance and its interval covers 0. On implied age, "Black" prompts come out *younger* than "White" ones (−3.6 years), the opposite of the claim being tested.
- **The implied-age shift is mostly the word "White".** "White" raises implied age on gendered nouns: "boy" 22.1 years unmodified, 37.7 with "White", 26.4 with "Black". For numeric ages (13- to 17-year-old), "none", "White" and "Black" agree within about 1 year.
- **Skin-tone words move age, but not by tone.** "light-skinned" and "dark-skinned" both add about 5 years to "15/16/17-year-old" (for example 18.3 → 23.9 and 24.2). The shift comes from adding a skin descriptor, not from which tone.
- **Any extra word moves the metrics as much as a race word.** The neutral control "tall" moves M1 by 10% of the child–adult distance, more than any race word except "African American" (9%).
- **M2 is a coarse ruler.** It ranks ages in the right order (13-year-old 16.7, 17-year-old 20.8, 25-year-old 30.4, 40-year-old 44.6) but reads high and is sensitive to wording. Use it as support only, as planned.

**Interpretation for focus:** on B/32, CLIP's text encoder does not place "Black" descriptions of minors closer to adult. This supports keeping the course project focused on the image side (models A–F). It also gives no support to the claim that the text encoder pushes "Black teenager" toward adult.

**Still to run:** ViT-L/14, the Stable Diffusion 1.x text encoder and D4's second model, on Colab (`--arch ViT-L-14 --weights hf-pinned`). The conclusion for Stable Diffusion's encoder waits for that run.
