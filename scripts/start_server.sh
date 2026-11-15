#!/usr/bin/env bash
# Starts the laptop's model server (llama.cpp llama-server) in the foreground. Leave this terminal
# open while LLM episodes run; Ctrl+C stops the server.
# Flags follow docs/design/llm_setup.md (cluster/run_shard.sh uses the same flags on another port).
#
# Usage:  bash scripts/start_server.sh [model-file]      (default: Qwen3-8B-Q4_K_M.gguf)
# Folders: LLAMA_DIR (holds llama.cpp/) and MODEL_DIR (holds the .gguf); both default to ~/llm.
#   e.g.  MODEL_DIR=/d/models bash scripts/start_server.sh Ministral-3-8B-Instruct-2512-Q4_K_M.gguf
# Check it from another terminal:  curl http://127.0.0.1:8080/health   ->  {"status":"ok"}
set -euo pipefail

llama_dir=${LLAMA_DIR:-$HOME/llm}
model_dir=${MODEL_DIR:-$HOME/llm}
model=$model_dir/${1:-Qwen3-8B-Q4_K_M.gguf}

server=$llama_dir/llama.cpp/llama-server.exe
if [ ! -f "$server" ]; then
  server=$llama_dir/llama.cpp/llama-server
fi
if [ ! -f "$server" ]; then
  echo "llama-server not found under $llama_dir/llama.cpp (install it: README, 'LLM runs')" >&2
  exit 1
fi
if [ ! -f "$model" ]; then
  echo "model not found: $model (download it: bash scripts/fetch_model.sh qwen3-8b $model_dir)" >&2
  exit 1
fi

# The episode configs must point llm.model_path at this same file, or run.json records another
# file's hash.
echo "starting $server with $model on http://127.0.0.1:8080"
exec "$server" -m "$model" --host 127.0.0.1 --port 8080 -ngl 99 --jinja \
  --parallel 6 --kv-unified --ctx-size 32768 -b 512 -ub 256 \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --cache-ram 0
