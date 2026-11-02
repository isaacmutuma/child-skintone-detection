"""Detector registry. Constructors are lazy so importing this never loads weights."""


def build(name, **kw):
    if name == "clip_vitb32":
        from .clip_zeroshot import ClipZeroShot
        return ClipZeroShot("ViT-B-32", **kw)
    if name == "clip_vitl14":
        from .clip_zeroshot import ClipZeroShot
        return ClipZeroShot("ViT-L-14", **kw)
    if name == "mivolo_face":
        from .mivolo_face import MivoloFace
        return MivoloFace(**kw)
    if name == "fairface_clf":
        from .fairface_clf import FairFaceClassifier
        return FairFaceClassifier(**kw)
    raise ValueError(f"unknown detector {name!r} (D5 not implemented yet)")


# which detector may run on which dataset (PREREGISTRATION.md)
ALLOWED = {
    "clip_vitb32": {"utkface", "fairface_val"},
    "clip_vitl14": {"utkface", "fairface_val"},
    "mivolo_face": {"utkface", "fairface_val"},
    "fairface_clf": {"utkface"},          # trained on FairFace -> never on FairFace
    "vlm": {"utkface", "fairface_val"},
}
