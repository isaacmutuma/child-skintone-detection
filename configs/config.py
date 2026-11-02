"""
configs/config.py
-----------------
Single source of truth for environment, paths, seeds, analysis constants and
detector prompts. Values marked FROZEN come from EXPERIMENT_PLAN.md and must not
change after the preregistration freeze (log any change in PREREGISTRATION.md
Deviations).

Environment: set CHILD_AUDIT_ENV=colab|local (default local), or edit ENV below.
"""

import os
from pathlib import Path

# ── Environment ───────────────────────────────────────────────────────────────
ENV = os.environ.get("CHILD_AUDIT_ENV", "local")  # 'local' | 'colab'

if ENV == "colab":
    REPO_DIR = Path("/content/child-detection-skin-tone-audit")
    DRIVE_ROOT = Path("/content/drive/MyDrive/child-detection-audit")
    RAW_DIR = DRIVE_ROOT / "raw"          # pinned archives (tar / parquet) + provenance
    DATA_DIR = Path("/content/data")      # unpacked from RAW_DIR each session (fast local disk)
    WEIGHTS_DIR = DRIVE_ROOT / "weights"
    # Per-image outputs never go in the repo; on Colab they live on Drive.
    PER_IMAGE_DIR = DRIVE_ROOT / "per_image"
    # HF model cache on Colab's local disk: Drive's FUSE mount corrupts large cached
    # safetensors / breaks HF cache symlinks ("header too large", 2026-11-02 smoke run).
    HF_CACHE_DIR = Path("/content/hf_cache")
elif ENV == "local":
    REPO_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = REPO_DIR / "data" / "dev"          # gitignored; <= DEV_SUBSET_MAX images
    WEIGHTS_DIR = REPO_DIR / "data" / "weights"   # gitignored
    PER_IMAGE_DIR = REPO_DIR / "data" / "per_image"
    HF_CACHE_DIR = WEIGHTS_DIR / "hf"
else:
    raise ValueError(f"Unknown ENV: {ENV!r}. Must be 'local' or 'colab'.")

# ── Pinned sources (verified 2026-11-02; see docs/DATA_CARD.md) ────────────────
SOURCES = {
    "utkface": {
        "hf_repo": "py97/UTKFace-Cropped", "revision": "e80f21fb21fa3631380c3cb634b79be3ac2f4cba",
        "file": "UTKFace.tar.gz",
        "sha256": "2c0655397b498e81b82d685ee74bee8c2e4364a5897caf3afec6ea48929024a7",
        "licence": "non-commercial research only (original UTKFace terms; mirror's MIT tag is wrong)",
        "crosscheck_repo": "nu-delta/utkface",
    },
    "fairface_val": {
        "hf_repo": "mutuma00/FairFace", "revision": "e8e670de53803408a4eaebf503d0a35749907f54",
        "file": "0.25/validation-00000-of-00001-951dbd63c8724ee1.parquet",
        "sha256": "d0cfac2888356b3d3fa3d837bc30961d8c860934e4c5df862e8cb87812bfa137",
        "licence": "CC BY 4.0",
        "upstream": "HuggingFaceM4/FairFace (identical LFS sha256)",
    },
}

UTKFACE_DIR = Path(os.environ.get("UTKFACE_DIR", DATA_DIR / "UTKFace"))   # flat folder of *.jpg(.chip.jpg)
FAIRFACE_DIR = Path(os.environ.get("FAIRFACE_DIR", DATA_DIR / "fairface"))
FAIRFACE_LABELS = FAIRFACE_DIR / "fairface_label_val.csv"                 # official CSV format
FAIRFACE_PADDING = "0.25"                                                 # official 0.25-padding crops

RESULTS_DIR = REPO_DIR / "results"   # aggregate JSON only
FIGURES_DIR = REPO_DIR / "figures"   # charts only
DOCS_DIR = REPO_DIR / "docs"

DEV_SUBSET_MAX = 500                 # hard cap on images kept on this Mac

# ── Randomness / stats (FROZEN) ───────────────────────────────────────────────
SEED = 0
N_BOOTSTRAP = 2000
ALPHA = 0.05
MIN_CELL = 50                        # primary cell below this -> fallback strata

# ── Age definitions (FROZEN) ──────────────────────────────────────────────────
MINOR_MAX_AGE = 17                   # minor = age 0-17
CHILD_AGE_THRESHOLD = 18             # detector says child if min est. age < 18
ADULT_FPR_RANGE = (20, 59)           # adults used for FPR
AGE_STRATA = [(0, 4), (5, 9), (10, 14), (15, 17)]          # primary
AGE_STRATA_FALLBACK = [(0, 9), (10, 17)]                   # if any primary cell < MIN_CELL

# FairFace ships age bins. Minors = 0-2, 3-9. 10-19 straddles 18 -> excluded from TPR.
FAIRFACE_MINOR_BINS = ["0-2", "3-9"]
FAIRFACE_AMBIGUOUS_BINS = ["10-19"]
FAIRFACE_ADULT_FPR_BINS = ["20-29", "30-39", "40-49", "50-59"]
FAIRFACE_BIN_MIDPOINTS = {           # approximate ages; age_is_exact=False
    "0-2": 1.0, "3-9": 6.0, "10-19": 14.5, "20-29": 24.5, "30-39": 34.5,
    "40-49": 44.5, "50-59": 54.5, "60-69": 64.5, "more than 70": 75.0,
}

# ── Groups (FROZEN, dataset-provided perceived race labels) ──────────────────
UTKFACE_RACES = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
UTKFACE_TESTED_GROUPS = ["White", "Black", "Asian", "Indian"]   # Others reported, not tested
FAIRFACE_RACES = ["White", "Black", "Latino_Hispanic", "East Asian",
                  "Southeast Asian", "Indian", "Middle Eastern"]
PRIMARY_CONTRAST = ("White", "Black")   # (lightest, darkest) for the primary endpoint
MST_BINS = {"light": (1, 3), "medium": (4, 7), "dark": (8, 10)}  # Phase 2

# ── Detector prompts (FROZEN once PREREGISTRATION.md is committed) ───────────
CLIP_CHILD_PROMPTS = [
    "a photo of a child",
    "a photo of a kid",
    "a photo of a baby",
    "a photo of a teenager",
]
CLIP_ADULT_PROMPTS = [
    "a photo of an adult",
    "a photo of a man",
    "a photo of a woman",
]
# Rule: child if max child-prompt similarity > max adult-prompt similarity; log margin.
CLIP_MODELS = [("ViT-B-32", "openai"), ("ViT-L-14", "openai")]

VLM_PROMPT = ("Is there a child (a person under 18 years old) in this image? "
              "Answer yes or no.")


def as_dict():
    """Serialisable snapshot of this config, stored in every results JSON."""
    return {k: (str(v) if isinstance(v, Path) else v)
            for k, v in globals().items()
            if k.isupper() and not k.startswith("_")}
