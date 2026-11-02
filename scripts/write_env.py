"""Write results/env.json: Python, package versions, device, git state.
Model weights are added under `weights` as each detector is pinned (Phase 1)."""

import json
import platform
import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from configs import config as C  # noqa: E402
from src.provenance import write_result  # noqa: E402

PKGS = ["torch", "torchvision", "open_clip_torch", "timm", "pandas", "numpy",
        "scipy", "scikit-learn", "matplotlib", "pillow"]


def pkg_versions():
    """Installed versions of the key packages (None if missing)."""
    out = {}
    for p in PKGS:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def main():
    out_path = C.RESULTS_DIR / "env.json"
    weights = {}
    if out_path.exists():  # keep pinned weights from earlier runs
        weights = json.loads(out_path.read_text()).get("weights", {})
    mps_ok = torch.backends.mps.is_available()
    if mps_ok:
        x = torch.ones(2, device="mps")
        mps_ok = bool((x + x).sum().item() == 4)
    payload = {
        "env": C.ENV,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": pkg_versions(),
        "device": {"mps_available": torch.backends.mps.is_available(),
                   "mps_smoke_test_passed": mps_ok,
                   "cuda_available": torch.cuda.is_available()},
        "weights": weights,
    }
    print(write_result(out_path, payload, {"ENV": C.ENV, "SEED": C.SEED}, C.REPO_DIR))


if __name__ == "__main__":
    main()
