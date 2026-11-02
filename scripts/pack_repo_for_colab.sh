#!/usr/bin/env bash
# Zip the code (no data, venv, images) so it can be uploaded to Drive for Colab.
# Usage: scripts/pack_repo_for_colab.sh [out.zip]   (default: ../child-audit-repo.zip)
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:-../child-audit-repo.zip}"
git rev-parse HEAD > GIT_COMMIT 2>/dev/null || echo NO_COMMIT_YET > GIT_COMMIT
rm -f "$out"
zip -qr "$out" . -x '.venv/*' 'data/*' '.git/*' '*.jpg' '*.jpeg' '*.png' '*.pt' '*.pth' \
  '*.safetensors' '*__pycache__*' '.DS_Store' '.claude/*'
rm GIT_COMMIT
echo "wrote $out ($(du -h "$out" | cut -f1))"
