#!/usr/bin/env bash
# Downloads a model file to $SCRATCH_DIR/models and checks its sha256 against the laptop's copy
# (docs/design/llm_models.md), so run.json hashes and response cache keys match across machines.
# Usage, on a login node: cluster/fetch_model.sh qwen3-8b   (or qwen3.5-4b, ministral3-8b)
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/cluster.env

case "${1:-qwen3-8b}" in
  qwen3-8b)
    file=Qwen3-8B-Q4_K_M.gguf
    url=https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/$file
    sha=d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785
    ;;
  qwen3.5-4b)
    file=Qwen3.5-4B-Q4_K_M.gguf
    url=https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/$file
    sha=00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4
    ;;
  ministral3-8b)
    file=Ministral-3-8B-Instruct-2512-Q4_K_M.gguf
    url=https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF/resolve/main/$file
    sha=33e7a72cf5e6e2cfc2f2847075acc013d68bba023e35310cef86b5cf8fdca761
    ;;
  *)
    echo "unknown model '$1' (known: qwen3-8b, qwen3.5-4b, ministral3-8b)" >&2
    exit 2
    ;;
esac

mkdir -p "$SCRATCH_DIR/models"
target=$SCRATCH_DIR/models/$file
if [ ! -f "$target" ]; then
  curl -L --fail -o "$target.part" "$url"
  mv "$target.part" "$target"
fi
echo "checking sha256 of $target"
if ! echo "$sha  $target" | sha256sum -c -; then
  echo "sha256 mismatch: not the file the laptop runs; delete it and rerun" >&2
  exit 1
fi
