"""Text-side probe of CLIP's text encoder (plan: docs/TEXT_PROBE_PLAN.md).

Embeds race-marked and unmarked descriptions of young people and adults with CLIP's
text encoder only (no images anywhere) and measures, per prompt:
  M1  D4's decision margin: max sim to D4 child prompts - max sim to D4 adult prompts
  M2  implied age: softmax(logit_scale * sim) over "a photo of a {N}-year-old person"
Then paired differences modifier - White per template x age noun, bootstrapped.

  .venv/bin/python scripts/text_probe.py --arch ViT-B-32 --weights local-openai
  python scripts/text_probe.py --arch ViT-L-14 --weights hf-pinned      # Colab
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs import config as C  # noqa: E402
from src.provenance import write_result  # noqa: E402
from src.detectors._weights import sha256  # noqa: E402

# OpenAI's published sha256 (part of the official download URL), checked before loading.
LOCAL_OPENAI = {"ViT-B-32": (Path.home() / ".cache/clip/ViT-B-32.pt",
                             "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af")}

TEMPLATES = ["a photo of {x}", "a close-up photo of the face of {x}", "a portrait of {x}",
             "a picture of {x}", "an image of {x}", "a photograph of {x}"]
MODIFIERS = ["", "White", "Black", "Asian", "Indian", "African American",
             "light-skinned", "dark-skinned", "tall", "smiling"]
RACE_SKIN = ["Black", "Asian", "Indian", "African American", "light-skinned", "dark-skinned"]
CONTROLS = ["tall", "smiling"]
MINOR_NOUNS = ["child", "kid", "boy", "girl", "teenager", "teenage boy", "teenage girl",
               "13-year-old", "15-year-old", "16-year-old", "17-year-old"]
ADULT_NOUNS = ["adult", "man", "woman", "25-year-old", "40-year-old"]
AGES = list(range(1, 81))
N_BOOT, SEED, MATERIAL_FRAC = 2000, 0, 0.10


def article(phrase):
    """'a' or 'an' for the phrase's first word (all words used here follow spelling)."""
    return "an" if phrase[0].lower() in "aeiou" else "a"


def phrase(mod, noun):
    p = f"{mod} {noun}".strip()
    return f"{article(p)} {p}"


def load(arch, weights):
    """Return (model, tokenizer, weights_info); weights are hash-checked before loading."""
    import open_clip
    if weights == "local-openai":
        path, expected = LOCAL_OPENAI[arch]
        got = sha256(path)
        if got != expected:
            raise SystemExit(f"sha256 mismatch for {path}: {got}")
        model = open_clip.load_openai_model(str(path), precision="fp32", device="cpu")
        info = {"source": "OpenAI original .pt (local cache)", "path": str(path), "sha256": got}
    else:
        from huggingface_hub import hf_hub_download
        from src.detectors.clip_zeroshot import WEIGHTS, WEIGHT_FILE
        repo, rev, expected = WEIGHTS[arch]
        path = hf_hub_download(repo, WEIGHT_FILE, revision=rev, cache_dir=str(C.HF_CACHE_DIR))
        got = sha256(path)
        if got != expected:
            raise SystemExit(f"sha256 mismatch for {path}: {got}")
        model = open_clip.create_model(arch, pretrained=path)
        info = {"source": "HF pinned (same as Phase 1 D4)", "repo": repo, "revision": rev,
                "sha256": got}
    model.eval()
    return model, open_clip.get_tokenizer(arch), info


@torch.no_grad()
def embed(model, tok, texts, bs=256):
    out = []
    for i in range(0, len(texts), bs):
        e = model.encode_text(tok(texts[i:i + bs])).float()
        out.append(e / e.norm(dim=-1, keepdim=True))
    return torch.cat(out).numpy()


def boot_ci(x, rng):
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    m = x[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="ViT-B-32", choices=["ViT-B-32", "ViT-L-14"])
    ap.add_argument("--weights", default="local-openai", choices=["local-openai", "hf-pinned"])
    a = ap.parse_args()
    torch.manual_seed(0)
    model, tok, winfo = load(a.arch, a.weights)
    scale = float(model.logit_scale.exp())

    child_e = embed(model, tok, C.CLIP_CHILD_PROMPTS)
    adult_e = embed(model, tok, C.CLIP_ADULT_PROMPTS)
    anchor_e = embed(model, tok, [f"a photo of a {n}-year-old person" for n in AGES])

    nouns = MINOR_NOUNS + ADULT_NOUNS
    keys = [(t, m, n) for t in TEMPLATES for m in MODIFIERS for n in nouns]
    texts = [t.format(x=phrase(m, n)) for t, m, n in keys]
    q = embed(model, tok, texts)
    m1 = (q @ child_e.T).max(1) - (q @ adult_e.T).max(1)
    logits = scale * (q @ anchor_e.T)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    m2 = p @ np.array(AGES, dtype=float)
    val = {k: (float(x1), float(x2)) for k, x1, x2 in zip(keys, m1, m2)}

    # child-adult distance on M1, unmodified prompts, averaged over templates
    dist = float(np.mean([val[(t, "", "teenager")][0] - val[(t, "", "adult")][0] for t in TEMPLATES]))

    rng = np.random.default_rng(SEED)
    contrasts = [(m, "White") for m in [""] + RACE_SKIN + CONTROLS] + \
                [("dark-skinned", "light-skinned")] + [(c, "") for c in CONTROLS]
    summary = {}
    for mod, ref in contrasts:
        for group, ns in (("minor", MINOR_NOUNS), ("adult", ADULT_NOUNS)):
            d1 = np.array([val[(t, mod, n)][0] - val[(t, ref, n)][0] for t in TEMPLATES for n in ns])
            d2 = np.array([val[(t, mod, n)][1] - val[(t, ref, n)][1] for t in TEMPLATES for n in ns])
            name = f"{mod or 'none'} - {ref or 'none'}"
            summary.setdefault(name, {})[group] = {
                "n_items": int(len(d1)),
                "m1_mean": float(d1.mean()), "m1_ci": boot_ci(d1, rng),
                "m1_frac_of_child_adult_distance": float(d1.mean() / dist),
                "m1_share_items_negative": float((d1 < 0).mean()),
                "m2_mean_years": float(d2.mean()), "m2_ci": boot_ci(d2, rng),
            }
    control_spread = max(abs(summary[f"{c} - none"]["minor"]["m1_mean"]) for c in CONTROLS)
    bw = summary["Black - White"]["minor"]
    crit = {"mean_below_0_ci_excludes_0": bw["m1_mean"] < 0 and bw["m1_ci"][1] < 0,
            "at_least_10pct_of_distance": abs(bw["m1_mean"]) >= MATERIAL_FRAC * dist,
            "larger_than_control_spread": abs(bw["m1_mean"]) > control_spread}

    # per-prompt-cell means over templates (descriptions only; no people or images)
    cells = {f"{m or 'none'} | {n}": {
        "m1_mean_over_templates": float(np.mean([val[(t, m, n)][0] for t in TEMPLATES])),
        "m2_mean_over_templates": float(np.mean([val[(t, m, n)][1] for t in TEMPLATES]))}
        for m in MODIFIERS for n in nouns}

    cfg = {"arch": a.arch, "weights": winfo, "templates": TEMPLATES, "modifiers": MODIFIERS,
           "minor_nouns": MINOR_NOUNS, "adult_nouns": ADULT_NOUNS,
           "d4_child_prompts": C.CLIP_CHILD_PROMPTS, "d4_adult_prompts": C.CLIP_ADULT_PROMPTS,
           "age_anchor": "a photo of a {N}-year-old person, N=1..80", "logit_scale": scale,
           "n_boot": N_BOOT, "seed": SEED, "material_frac": MATERIAL_FRAC,
           "plan": "docs/TEXT_PROBE_PLAN.md", "torch": torch.__version__}
    payload = {"n_prompts": len(texts), "child_adult_distance_m1": dist,
               "control_spread_m1_minor": control_spread,
               "decision": {"criteria": crit, "material_text_side_shift": all(crit.values())},
               "contrasts": summary, "cells": cells}
    out = write_result(ROOT / f"results/text_probe/text_probe_{a.arch}.json", payload, cfg, ROOT)
    print(f"wrote {out}\nprompts {len(texts)}  child-adult distance (M1) {dist:.4f}  "
          f"control spread {control_spread:.4f}")
    print(f"{'contrast':<30}{'group':<7}{'M1 mean':>9}{'95% CI':>22}{'frac':>7}{'M2 yrs':>8}{'95% CI':>18}")
    for name, g in summary.items():
        for grp, s in g.items():
            print(f"{name:<30}{grp:<7}{s['m1_mean']:>9.4f}  [{s['m1_ci'][0]:.4f}, {s['m1_ci'][1]:.4f}]"
                  f"{s['m1_frac_of_child_adult_distance']:>7.2f}{s['m2_mean_years']:>8.2f}"
                  f"  [{s['m2_ci'][0]:.2f}, {s['m2_ci'][1]:.2f}]")
    print("decision:", crit, "->", "MATERIAL" if all(crit.values()) else "not material")


if __name__ == "__main__":
    main()
