"""D3: FairFace-trained age-bin classifier, UTKFace only (never FairFace: trained on it).

Replacement (PREREGISTRATION.md Deviations, 2026-11-02): the original dchen236/FairFace
ResNet-34 weights are no longer downloadable (Drive folder 404), so D3 uses
dima806/fairface_age_image_detection (ViT-B/16 fine-tuned on FairFace, same 9 age bins).

Pipeline: dlib CNN face detector (image shrunk to max side 800) + 5-point landmarks ->
get_face_chips(size=300, padding=0.25), the crop style FairFace images were made with ->
ViT image processor (224, mean/std 0.5) -> softmax over 9 age bins.
Decision rule (unchanged): age_est = sum_b P(b) * midpoint_b (70+ -> 75); min over faces;
child if age_est < 18; score = 18 - age_est; no face -> not child ('no_face').
"""

import numpy as np
import torch

from configs import config as C
from .base import Detector, empty_output, pick_device
from ._weights import sha256

MODEL = ("dima806/fairface_age_image_detection", "4e02ab8057ea7fd74b1670940995c5dfda3e6ec0")
MODEL_SHA256 = "1265d8e5a8a363a615898be873c224aebc6f310654bf23b1745da0975a7b7555"
DLIB_FILES = {"cnn": "mmod_human_face_detector.dat", "sp5": "shape_predictor_5_face_landmarks.dat"}


class FairFaceClassifier(Detector):
    name = "fairface_clf"

    def __init__(self, device=None, weights_dir=None, cache_dir=None):
        import dlib
        from huggingface_hub import snapshot_download
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.dlib = dlib
        self.device = pick_device(device)
        wd = weights_dir or (C.WEIGHTS_DIR / "fairface")
        dpaths = {k: str(wd / v) for k, v in DLIB_FILES.items()}
        self.cnn = dlib.cnn_face_detection_model_v1(dpaths["cnn"])
        self.sp = dlib.shape_predictor(dpaths["sp5"])

        mdir = snapshot_download(MODEL[0], revision=MODEL[1],
                                 cache_dir=str(cache_dir or C.HF_CACHE_DIR),
                                 allow_patterns=["config.json", "preprocessor_config.json",
                                                 "model.safetensors"])
        got = sha256(f"{mdir}/model.safetensors")
        if got != MODEL_SHA256:
            raise RuntimeError(f"D3 weights sha256 mismatch: {got}")
        self.proc = AutoImageProcessor.from_pretrained(mdir)
        self.model = AutoModelForImageClassification.from_pretrained(mdir).to(self.device).eval()
        labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self.midpoints = np.array([C.FAIRFACE_BIN_MIDPOINTS[b] for b in labels], dtype=np.float64)
        self._info = {"fairface_clf": {
            "hf_repo": MODEL[0], "revision": MODEL[1], "sha256_model_safetensors": got,
            "age_bins": labels,
            **{f"sha256_dlib_{k}": sha256(p) for k, p in dpaths.items()},
            "dlib_source": "http://dlib.net/files/"}}

    def weights_info(self):
        return self._info

    def _chips(self, path):
        img = self.dlib.load_rgb_image(path)
        h, w = img.shape[:2]
        s = 800 / max(h, w)
        if s < 1:
            img = self.dlib.resize_image(img, rows=int(h * s), cols=int(w * s))
        dets = self.cnn(img, 1)
        if len(dets) == 0:
            return [], []
        objs = self.dlib.full_object_detections()
        areas = []
        for d in dets:
            objs.append(self.sp(img, d.rect))
            areas.append(d.rect.area() / float(img.shape[0] * img.shape[1]))
        return self.dlib.get_face_chips(img, objs, size=300, padding=0.25), areas

    @torch.no_grad()
    def predict(self, records):
        outs = []
        for r in records:
            try:
                chips, areas = self._chips(r.path)
            except Exception as e:
                outs.append(empty_output(face_found=False, fail_reason=f"load_error:{type(e).__name__}"))
                continue
            if not chips:
                outs.append(empty_output(face_found=False, fail_reason="no_face", n_faces=0))
                continue
            x = self.proc(images=[np.asarray(c) for c in chips], return_tensors="pt")["pixel_values"]
            logits = self.model(pixel_values=x.to(self.device)).logits.float()
            p = torch.softmax(logits, dim=1).cpu().numpy()
            ages = p @ self.midpoints
            k = int(np.argmin(ages))
            age = float(ages[k])
            outs.append(empty_output(is_child=age < C.CHILD_AGE_THRESHOLD,
                                     score=C.CHILD_AGE_THRESHOLD - age, age_est=age,
                                     face_found=True, n_faces=len(chips),
                                     face_area_frac=areas[k],
                                     p_age_0_19=float(p[k, :3].sum())))
        return outs
