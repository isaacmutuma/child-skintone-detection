"""Per-image confounders (EXPERIMENT_PLAN.md §5): resolution, brightness,
contrast (RMS), blur (variance of Laplacian). Face size comes from D1's detector."""

import numpy as np
from PIL import Image
from scipy.ndimage import laplace


def image_stats(path):
    im = Image.open(path).convert("L")
    w, h = im.size
    a = np.asarray(im, dtype=np.float32) / 255.0
    return {"width": w, "height": h, "brightness": float(a.mean()),
            "contrast": float(a.std()), "blur_lapvar": float(laplace(a).var())}
