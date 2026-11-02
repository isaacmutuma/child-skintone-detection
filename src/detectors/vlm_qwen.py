"""D5: open VLM yes/no detector, Qwen2-VL-2B-Instruct (fits a Colab T4; PREREGISTRATION.md Deviations).

Prompt (frozen, configs/config.py VLM_PROMPT) is sent as one user turn with the image.
We do NOT sample text: one forward pass, and read the next-token distribution.
  P_yes = sum P over {"Yes","yes"} first tokens; P_no likewise for {"No","no"}
  score = log(P_yes) - log(P_no)   (> 0  <=>  P(yes | yes-or-no) > 0.5)
  is_child = score > 0
  yesno_mass = P_yes + P_no  (format check: should be close to 1)
Needs Colab's recent transformers (Qwen2VLForConditionalGeneration). Runs on CUDA only.
"""

import math

import torch
from PIL import Image

from configs import config as C
from .base import Detector, empty_output, pick_device
from ._weights import sha256

MODEL = ("Qwen/Qwen2-VL-2B-Instruct", "895c3a49bc3fa70a340399125c650a463535e71c")
SHARDS = {"model-00001-of-00002.safetensors":
          "994ac2b03f97de8bc647d0fe5eba2e4b632b3e28dc03574c29bdfc36cf47e1b9",
          "model-00002-of-00002.safetensors":
          "92540d8353c8d226a589a3b179bdb33851c970ee2cc2ac7ba035f79425e7b833"}


class QwenVLM(Detector):
    """D5: asks Qwen2-VL whether the image shows a child and compares P(yes) with P(no)."""

    name = "vlm"

    def __init__(self, device=None, cache_dir=None):
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.device = pick_device(device)
        if self.device.type != "cuda":
            raise SystemExit("D5 (Qwen2-VL) runs on a CUDA GPU only (Colab T4)")
        mdir = snapshot_download(MODEL[0], revision=MODEL[1],
                                 cache_dir=str(cache_dir or C.HF_CACHE_DIR))
        for f, want in SHARDS.items():
            got = sha256(f"{mdir}/{f}")
            if got != want:
                raise RuntimeError(f"Qwen2-VL shard {f} sha256 mismatch ({got})")
        self.proc = AutoProcessor.from_pretrained(mdir)
        self.proc.tokenizer.padding_side = "left"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            mdir, torch_dtype=torch.float16).to(self.device).eval()
        tok = self.proc.tokenizer

        def first_ids(words):
            return sorted({tok.encode(w, add_special_tokens=False)[0] for w in words})
        self.yes_ids, self.no_ids = first_ids(["Yes", "yes"]), first_ids(["No", "no"])
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": C.VLM_PROMPT}]}]
        self.text = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        self._info = {"vlm": {"hf_repo": MODEL[0], "revision": MODEL[1], "dtype": "float16",
                              "sha256_shards": SHARDS, "prompt": C.VLM_PROMPT,
                              "yes_token_ids": self.yes_ids, "no_token_ids": self.no_ids,
                              "decision": "log P(yes) - log P(no) > 0, next-token, no sampling"}}

    def weights_info(self):
        return self._info

    @torch.no_grad()
    def predict(self, records):
        outs, imgs, idx = [empty_output() for _ in records], [], []
        for i, r in enumerate(records):
            try:
                imgs.append(Image.open(r.path).convert("RGB"))
                idx.append(i)
            except Exception as e:
                outs[i] = empty_output(fail_reason=f"load_error:{type(e).__name__}")
        if not imgs:
            return outs
        batch = self.proc(text=[self.text] * len(imgs), images=imgs, padding=True,
                          return_tensors="pt").to(self.device)
        logits = self.model(**batch).logits[:, -1, :].float()
        p = torch.softmax(logits, dim=-1)
        p_yes = p[:, self.yes_ids].sum(dim=1).cpu()
        p_no = p[:, self.no_ids].sum(dim=1).cpu()
        for j, i in enumerate(idx):
            py, pn = float(p_yes[j]), float(p_no[j])
            score = math.log(max(py, 1e-12)) - math.log(max(pn, 1e-12))
            outs[i] = empty_output(is_child=score > 0, score=score,
                                   p_yes=py / (py + pn) if py + pn > 0 else None,
                                   yesno_mass=py + pn)
        return outs
