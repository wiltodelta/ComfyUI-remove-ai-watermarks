#!/usr/bin/env bash
set -euo pipefail

install_with_retry() {
  local attempt
  for attempt in $(seq 1 5); do
    if python -m pip install --no-cache-dir "$@"; then
      return 0
    fi
    if [[ "$attempt" -lt 5 ]]; then
      echo "pip install failed (attempt $attempt/5); waiting 15s for PyPI CDN propagation..." >&2
      sleep 15
    fi
  done
  return 1
}

if [[ -n "${LIBRARY_VERSION:-}" ]]; then
  install_with_retry "remove-ai-watermarks[all,classify-onnx,text-draft]==$LIBRARY_VERSION" pytest
else
  python -m pip install --requirement requirements.txt pytest
fi
python -m pytest
