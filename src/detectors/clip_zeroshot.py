"""D4: CLIP zero-shot child detector (prompts and rule frozen in PREREGISTRATION.md).

child if max cosine sim over child prompts > max over adult prompts.
score = that margin (child_max - adult_max), logged for threshold analysis.
Weights are downloaded from a pinned HF revision and their SHA256 recorded.
"""

import torch
from PIL import Image

from configs import config as C
from .base import Detector, empty_output, pick_device
from ._weights import sha256 as _sha256

# OpenAI CLIP weights as packaged by timm/open_clip on the HF hub, pinned revisions
WEIGHTS = {
    "ViT-B-32": ("timm/vit_base_patch32_clip_224.openai", "a6f597a30f7b82c51704746581f9a4e41421e878"),
    "ViT-L-14": ("timm/vit_large_patch14_clip_224.openai", "18d0535469bb561bf468d76c1d73aa35156c922b"),
}
WEIGHT_FILE = "open_clip_model.safetensors"


class ClipZeroShot(Detector):
    def __init__(self, arch="ViT-B-32", device=None, cache_dir=None):
        import open_clip
        from huggingface_hub import hf_hub_download

        self.arch = arch
        self.name = f"clip_{arch.lower().replace('-', '')}"
        self.device = pick_device(device)
        repo, rev = WEIGHTS[arch]
        self.weight_path = hf_hub_download(repo, WEIGHT_FILE, revision=rev,
                                           cache_dir=cache_dir or C.WEIGHTS_DIR / "hf")
        self._info = {"hf_repo": repo, "revision": rev, "file": WEIGHT_FILE,
                      "sha256": _sha256(self.weight_path), "open_clip_arch": arch}
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=self.weight_path)
        self.model = model.eval().to(self.device)
        tok = open_clip.get_tokenizer(arch)
        prompts = C.CLIP_CHILD_PROMPTS + C.CLIP_ADULT_PROMPTS
        with torch.no_grad():
            t = self.model.encode_text(tok(prompts).to(self.device)).float()
        self.text = t / t.norm(dim=-1, keepdim=True)
        self.n_child = len(C.CLIP_CHILD_PROMPTS)

    def weights_info(self):
        return {self.name: self._info}

    @torch.no_grad()
    def predict(self, records):
        outs, imgs, idx = [empty_output() for _ in records], [], []
        for i, r in enumerate(records):
            try:
                imgs.append(self.preprocess(Image.open(r.path).convert("RGB")))
                idx.append(i)
            except Exception as e:  # unreadable image: logged, counted as not child
                outs[i] = empty_output(fail_reason=f"load_error:{type(e).__name__}")
        if not imgs:
            return outs
        f = self.model.encode_image(torch.stack(imgs).to(self.device)).float()
        f = f / f.norm(dim=-1, keepdim=True)
        sims = (f @ self.text.T).cpu()
        child_max = sims[:, :self.n_child].max(dim=1).values
        adult_max = sims[:, self.n_child:].max(dim=1).values
        for j, i in enumerate(idx):
            margin = float(child_max[j] - adult_max[j])
            outs[i] = empty_output(is_child=margin > 0, score=margin)
        return outs
