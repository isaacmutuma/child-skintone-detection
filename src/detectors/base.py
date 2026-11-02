"""Common detector interface (EXPERIMENT_PLAN.md §4, §8).

Every detector maps a batch of Records to one output dict per record:
  is_child   bool        final decision (no face/person found -> False)
  score      float|None  continuous score for threshold analysis (higher = more child-like)
  age_est    float|None  estimated age (min over detected people), if the detector gives one
  face_found bool|None   None when the detector has no face stage (CLIP, VLM)
  fail_reason str|None   e.g. 'no_face', 'load_error'
"""

from abc import ABC, abstractmethod

import torch


def pick_device(prefer=None):
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_output(**kw):
    out = {"is_child": False, "score": None, "age_est": None,
           "face_found": None, "fail_reason": None}
    out.update(kw)
    return out


class Detector(ABC):
    name: str = "base"

    @abstractmethod
    def predict(self, records):
        """records: list[Record] -> list[dict] (same order, same length)."""

    def weights_info(self):
        """Pinned weight provenance for results/env.json."""
        return {}
