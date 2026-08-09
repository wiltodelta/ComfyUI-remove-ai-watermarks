#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${LIBRARY_VERSION:-}" ]]; then
  python -m pip install "remove-ai-watermarks[qwen-zimage,heif]==$LIBRARY_VERSION" pytest
else
  python -m pip install --requirement requirements.txt pytest
fi
python -m pytest
