"""D3: FairFace ResNet-34 classifier (github.com/dchen236/FairFace), UTKFace only.

Pipeline as in the FairFace repo's predict.py: dlib CNN face detector (image resized to
max side 800) + 5-point landmarks -> get_face_chips(size=300, padding=0.25) -> resize 224,
ImageNet normalisation -> fairface_alldata_20191111.pt (18 outputs; [9:18] = 9 age bins).
Decision rule (PREREGISTRATION.md, clarification 2026-11-02):
  age_est = sum_b softmax(age)_b * midpoint_b (70+ -> 75); min over faces
  child if age_est < 18; score = 18 - age_est; no face -> not child ('no_face')
Weights/dlib models are placed in WEIGHTS_DIR/fairface by the Colab notebook.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from configs import config as C
from .base import Detector, empty_output, pick_device
from ._weights import sha256

AGE_BINS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
MIDPOINTS = np.array([C.FAIRFACE_BIN_MIDPOINTS[b] for b in AGE_BINS], dtype=np.float64)
FILES = {"model": "fairface_alldata_20191111.pt",
         "cnn": "mmod_human_face_detector.dat",
         "sp5": "shape_predictor_5_face_landmarks.dat"}


class FairFaceClassifier(Detector):
    name = "fairface_clf"

    def __init__(self, device=None, weights_dir=None):
        import dlib
        self.dlib = dlib
        wd = weights_dir or (C.WEIGHTS_DIR / "fairface")
        paths = {k: str(wd / v) for k, v in FILES.items()}
        self.device = pick_device(device)
        self.cnn = dlib.cnn_face_detection_model_v1(paths["cnn"])
        self.sp = dlib.shape_predictor(paths["sp5"])
        m = torchvision.models.resnet34(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 18)
        m.load_state_dict(torch.load(paths["model"], map_location="cpu"))
        self.model = m.to(self.device).eval()
        self.tf = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self._info = {"fairface_clf": {
            "source": "github.com/dchen236/FairFace (Google Drive models folder "
                      "1F_pXfbzWvG-bhCpNsRj6F_xsdjpesiFu); dlib.net models",
            **{f"sha256_{k}": sha256(p) for k, p in paths.items()}}}

    def weights_info(self):
        return self._info

    def _chips(self, path):
        img = self.dlib.load_rgb_image(path)
        h, w = img.shape[:2]
        s = 800 / max(h, w)
        if s < 1:  # predict.py only shrinks large images
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
            x = torch.stack([self.tf(c) for c in chips]).to(self.device)
            p = torch.softmax(self.model(x)[:, 9:18].float(), dim=1).cpu().numpy()
            ages = p @ MIDPOINTS
            k = int(np.argmin(ages))
            age = float(ages[k])
            outs.append(empty_output(is_child=age < C.CHILD_AGE_THRESHOLD,
                                     score=C.CHILD_AGE_THRESHOLD - age, age_est=age,
                                     face_found=True, n_faces=len(chips),
                                     face_area_frac=areas[k],
                                     p_age_0_19=float(p[k, :3].sum())))
        return outs
