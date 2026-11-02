"""Provenance helpers: every results JSON stores config + git state."""

import datetime as _dt
import json
import subprocess
from pathlib import Path


def git_state(repo_dir):
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=repo_dir, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    commit = run("rev-parse", "HEAD")
    if commit is not None:
        return {"commit": commit, "dirty": bool(run("status", "--porcelain")),
                "source": "git"}
    stamp = Path(repo_dir) / "GIT_COMMIT"   # written by scripts/pack_repo_for_colab.sh
    if stamp.exists():
        return {"commit": stamp.read_text().strip(), "dirty": None, "source": "GIT_COMMIT file"}
    return {"commit": "NO_COMMIT_YET", "dirty": None, "source": "none"}


def write_result(path, payload, config_dict, repo_dir):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(repo_dir),
        "config": config_dict,
        **payload,
    }
    path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    return path
