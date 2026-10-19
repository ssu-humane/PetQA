#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BACKEND=${1:-}

if [[ -z "$BACKEND" ]]; then
  echo "Usage: $0 {vllm|transformers|api} --modality {text|multimodal} [args]" >&2
  exit 2
fi
shift
cd "$ROOT_DIR"

case "$BACKEND" in
  vllm|transformers|api)
    exec python -m "generation.$BACKEND" "$@"
    ;;
  *)
    echo "Usage: $0 {vllm|transformers|api} --modality {text|multimodal} [args]" >&2
    exit 2
    ;;
esac
