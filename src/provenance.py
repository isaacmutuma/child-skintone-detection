"""Provenance helpers: every results JSON stores config + git state."""

import datetime as _dt
import json
import subprocess
from pathlib import Path


def git_state(repo_dir):
    """Commit hash and dirty flag of `repo_dir`, stored in every results JSON.

    `dirty` ignores results/ and figures/, which runs write to themselves.
    """
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=repo_dir, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    commit = run("rev-parse", "HEAD")
    if commit is not None:
        # dirty = uncommitted CODE changes; outputs written by runs (results/, figures/)
        # don't count, otherwise every run would flag itself as dirty.
        status = run("status", "--porcelain", "--", ".", ":!results", ":!figures")
        return {"commit": commit, "dirty": bool(status), "source": "git"}
    stamp = Path(repo_dir) / "GIT_COMMIT"   # legacy: zip uploads without .git (Phase 0 only)
    if stamp.exists():
        return {"commit": stamp.read_text().strip(), "dirty": None, "source": "GIT_COMMIT file"}
    return {"commit": "NO_COMMIT_YET", "dirty": None, "source": "none"}


def write_result(path, payload, config_dict, repo_dir):
    """Write `payload` as JSON together with a UTC timestamp, git state and config.

    This is how every number in results/ stays traceable to the code that made it.
    """
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
