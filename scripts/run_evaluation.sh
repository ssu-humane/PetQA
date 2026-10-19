#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EVALUATOR=${1:-}

if [[ -z "$EVALUATOR" ]]; then
  echo "Usage: $0 {auto|judge} --modality {text|multimodal} [args]" >&2
  exit 2
fi
shift
cd "$ROOT_DIR"

case "$EVALUATOR" in
  auto)
    exec python -m evaluation.automatic "$@"
    ;;
  judge)
    exec python -m evaluation.llm_judge "$@"
    ;;
  *)
    echo "Usage: $0 {auto|judge} --modality {text|multimodal} [args]" >&2
    exit 2
    ;;
esac
