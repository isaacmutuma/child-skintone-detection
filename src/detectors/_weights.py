"""Helpers for pinned weight files."""

import hashlib


def sha256(path, bs=1 << 20):
    """Hex SHA-256 of a file, read in 1 MB blocks (weights are checked before loading)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()
