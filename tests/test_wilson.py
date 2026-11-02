"""Unit checks for the face-fail score intervals (stratified Wilson + Newcombe)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.metrics import effective_n, newcombe_gap, stratified_wilson, wilson  # noqa: E402


def main():
    # Textbook values: Wilson for 0/10 -> [0, 0.2775]; 5/10 -> [0.2366, 0.7634]
    lo, hi = wilson(0.0, 10)
    assert lo == 0.0 and abs(hi - 0.2775) < 1e-3, (lo, hi)
    lo, hi = wilson(0.5, 10)
    assert abs(lo - 0.2366) < 1e-3 and abs(hi - 0.7634) < 1e-3, (lo, hi)

    # Newcombe (1998) Table II example: 56/70 vs 48/80 -> d 0.2, CI [0.0524, 0.3339]
    p1, p2 = 56 / 70, 48 / 80
    lo, hi = newcombe_gap(p1, wilson(p1, 70), p2, wilson(p2, 80))
    assert abs(lo - 0.0524) < 1e-3 and abs(hi - 0.3339) < 1e-3, (lo, hi)

    # Unstratified case: n_eff equals n
    w = {"a": 1.0}
    assert abs(effective_n(np.array(["a"] * 280), w) - 280) < 1e-9

    # Zero failures in a stratified group: CI must have a positive upper bound
    w = {"0-9": 0.7234, "10-17": 0.2766}
    strata = np.array(["0-9"] * 154 + ["10-17"] * 126)
    p, (lo, hi) = stratified_wilson(np.zeros(280), strata, w)
    assert p == 0.0 and lo == 0.0 and hi > 0.005, (p, lo, hi)
    print("wilson tests PASSED")


if __name__ == "__main__":
    main()
