"""D1: MiVOLO v2 (face-only) with the YOLOv8x person/face detector.

Decision rule (PREREGISTRATION.md, clarification 2026-11-02):
  faces = YOLO 'face' boxes (conf 0.4, IoU 0.7); none -> not child, fail_reason 'no_face'
  age_est = min MiVOLO age over faces; child if age_est < 18; score = 18 - age_est
Body input is left empty (processor given None), i.e. face-only MiVOLO.
Needs (Colab): pip install git+https://github.com/WildChlamydia/MiVOLO.git@<pinned>
               transformers==4.51.0 ultralytics==8.1.0 accelerate
"""

import os

# Headless: Colab exports MPLBACKEND=module://matplotlib_inline..., which the isolated
# MiVOLO venv doesn't have; ultralytics imports matplotlib at import time and crashes.
os.environ["MPLBACKEND"] = "Agg"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from configs import config as C  # noqa: E402
from .base import Detector, empty_output, pick_device  # noqa: E402
from ._weights import sha256  # noqa: E402

MIVOLO = ("iitolstykh/mivolo_v2", "e8bdd8aec21ae468cbfe73b41f52550105ad8fbf")
YOLO = ("iitolstykh/YOLO-Face-Person-Detector", "b3d071aedccd46a3b2d4b40609da6880a815f395",
        "yolov8x_person_face.pt")
YOLO_CONF, YOLO_IOU = 0.4, 0.7


class MivoloFace(Detector):
    name = "mivolo_face"

    def __init__(self, device=None, cache_dir=None):
        from huggingface_hub import hf_hub_download, snapshot_download
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification
        from ultralytics import YOLO as UYOLO

        self.device = pick_device(device)
        cache = str(cache_dir or C.HF_CACHE_DIR)
        ypath = hf_hub_download(YOLO[0], YOLO[2], revision=YOLO[1], cache_dir=cache)
        self.yolo = UYOLO(ypath)
        mdir = snapshot_download(MIVOLO[0], revision=MIVOLO[1], cache_dir=cache)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.cfg = AutoConfig.from_pretrained(mdir, trust_remote_code=True)
        self.model = AutoModelForImageClassification.from_pretrained(
            mdir, trust_remote_code=True, torch_dtype=dtype).to(self.device).eval()
        self.proc = AutoImageProcessor.from_pretrained(mdir, trust_remote_code=True)
        self.face_cls = [k for k, v in self.yolo.names.items() if v == "face"][0]
        self._info = {
            "mivolo_face": {
                "mivolo": {"hf_repo": MIVOLO[0], "revision": MIVOLO[1],
                           "sha256_model_safetensors": sha256(f"{mdir}/model.safetensors"),
                           "dtype": str(dtype)},
                "yolo": {"hf_repo": YOLO[0], "revision": YOLO[1], "file": YOLO[2],
                         "sha256": sha256(ypath), "conf": YOLO_CONF, "iou": YOLO_IOU},
            }
        }

    def weights_info(self):
        return self._info

    @torch.no_grad()
    def _ages(self, crops):
        faces = self.proc(images=crops)["pixel_values"]
        bodies = self.proc(images=[None] * len(crops))["pixel_values"]
        faces = faces.to(self.device, dtype=self.model.dtype)
        bodies = bodies.to(self.device, dtype=self.model.dtype)
        out = self.model(faces_input=faces, body_input=bodies)
        return out.age_output.float().cpu().numpy().reshape(-1)

    def predict(self, records):
        outs = []
        for r in records:
            img = cv2.imread(r.path)  # BGR, as the MiVOLO processor expects
            if img is None:
                outs.append(empty_output(face_found=False, fail_reason="load_error"))
                continue
            res = self.yolo.predict(img, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False,
                                    device=self.device.index if self.device.type == "cuda"
                                    else self.device.type)[0]
            boxes = [b for b in res.boxes if int(b.cls) == self.face_cls]
            if not boxes:
                outs.append(empty_output(face_found=False, fail_reason="no_face", n_faces=0))
                continue
            crops, areas = [], []
            H, W = img.shape[:2]
            for b in boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                crops.append(img[y1:y2, x1:x2])
                areas.append((x2 - x1) * (y2 - y1) / float(W * H))
            ages = self._ages(crops)
            k = int(np.argmin(ages))
            age = float(ages[k])
            outs.append(empty_output(is_child=age < C.CHILD_AGE_THRESHOLD,
                                     score=C.CHILD_AGE_THRESHOLD - age, age_est=age,
                                     face_found=True, n_faces=len(boxes),
                                     face_area_frac=areas[k]))
        return outs
