# LLM setup: one llama.cpp server, many agents

Every LLM agent shares one model server (hard rule 6). The sim speaks the OpenAI-compatible chat API (`llm/client.py`), so the same code runs against llama.cpp on the laptop and llama.cpp or vLLM on the cluster. Cluster runs (Apptainer image with the same llama.cpp build, one server per Slurm array task) are described in `cluster.md`.

## Pinned versions

| Piece | Version | Where |
|---|---|---|
| llama.cpp | release `b12211`, Windows CUDA 12.4 x64 build plus its `cudart` runtime | `C:\Users\domin\llm\llama.cpp\` |
| Model | Qwen3-8B, `Qwen3-8B-Q4_K_M.gguf` from `Qwen/Qwen3-8B-GGUF` on Hugging Face (about 5 GB) | `C:\Users\domin\llm\` |

The model file's sha256 is written to every LLM run's `run.json` (`model_file_hash`, hard rule 9) when the episode config's `llm.model_path` points at it.

## Install (Windows, native)

```bash
mkdir -p /c/Users/domin/llm && cd /c/Users/domin/llm
curl -L -o llama.zip https://github.com/ggml-org/llama.cpp/releases/download/b12211/llama-b12211-bin-win-cuda-12.4-x64.zip
curl -L -o cudart.zip https://github.com/ggml-org/llama.cpp/releases/download/b12211/cudart-llama-bin-win-cuda-12.4-x64.zip
mkdir -p llama.cpp && cd llama.cpp && unzip -o ../llama.zip && unzip -o ../cudart.zip && cd ..
curl -L -o Qwen3-8B-Q4_K_M.gguf https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf
```

WSL2 is a fallback only if the Windows build lacks something we need. The Ubuntu WSL distributions on this machine can run the Linux CUDA build with the same flags.

## Run the server

```bash
/c/Users/domin/llm/llama.cpp/llama-server.exe -m /c/Users/domin/llm/Qwen3-8B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 -ngl 99 --jinja \
  --parallel 6 --kv-unified --ctx-size 32768 -b 512 -ub 256 \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --cache-ram 0
```

- **`--parallel 6`:** one slot per agent in a house, so the requests of one tick are batched. Each agent is pinned to its own slot (`llm.pin_slots`), so its system prompt stays in that slot's prompt cache.
- **`--kv-unified --ctx-size 32768`:** one KV buffer shared by all slots. Each slot keeps its last prompt cached, so the buffer must hold six full house prompts at once. A kitchen executor prompt is about 0.7k tokens. A house prompt (prompt plus reply) is 2.8k tokens at the median, 4.0k at the 99th percentile, and 4.4k at most (house LLM pilot, Nov 12-14). Six slots therefore need up to about 27k tokens. Confirm VRAM with `nvidia-smi` after loading; at 16k the server used 5.9 of the RTX 4070's 8 GB, and the extra 16k of q8 KV adds about 1.2 GB.
  - *Measured, Nov 12-14:* with `--ctx-size 16384`, the server answered HTTP 500 "Context size has been exceeded" 107 times in the house pilot, and 3 of 24 episodes stopped after their retries ran out.
  - *Measured, Nov 8:* split per-slot buffers of 36,864 and then 30,720 tokens (2.2 GiB of q8 KV beside the 4.7 GiB model) both ran out of CUDA memory mid-episode, after 12 and 19 minutes. Those crashes now look like host memory exhaustion (see the zombie-process leak below), not VRAM, but recheck VRAM if 32k crashes.
- **`-b 512 -ub 256`:** smaller batch buffers, which shrink the compute buffer.
- **`--cache-ram 0`:** turns off the server's host-RAM prompt cache (the default is 8 GB; 2048 was used on Nov 8). Each agent's prefix stays in its slot, so the cache matters only when slots evict each other, and host memory is the scarce resource on this laptop.
  - *Measured, Nov 8:* a third crash came with VRAM flat at 5.9 GB. All three crashes happened while heavy CPU jobs ran beside the server (a 4-process regression, the house generator), and the host prompt cache sat at its 8 GB cap. With the Windows (WDDM) driver, GPU allocations also count against system memory commit, so host memory pressure is the likely cause. This is not proven.
  - **Do not run heavy jobs beside a live pilot** (regressions, generation, dev refreshes); run them before or after.
  - *Nov 9, stronger evidence:* with only 3.6 GB of RAM free (11 days of uptime, and `dwm.exe` holding 5 GB of shared GPU memory on the integrated GPU), every server crashed. That included a 4B model at 3.1 GB of VRAM with 3 slots, no host cache, and CUDA graphs off. VRAM was never the limit.
- **Keep Windows awake during a live run.** `run_single` does not (the sweep runner does): a probe slept for hours on Nov 10. Hold `experiments.sweep.keep_awake(True)` from a helper process, as the bake-off scripts did.
- **Before a live run:**
  - **Check free memory:** `Get-Counter '\Memory\Available MBytes'` should be well above 8 GB.
  - **Check the Desktop Window Manager's shared GPU memory:** `(Get-Counter '\GPU Process Memory(*)\Shared Usage').CounterSamples | Sort-Object CookedValue -Descending | Select-Object -First 3`. A large `pid_<dwm>` entry is the leak.
  - **Check for zombie processes:** `(Get-Counter '\Objects\Processes').CounterSamples[0].CookedValue` should be close to `(Get-Process).Count`. Each exited process that is never fully released keeps about 90 KB of kernel memory.
    - *Measured, Nov 12-14:* 279,058 process objects against 312 live processes. Free RAM fell from 14.9 GB to 0.5 GB over the 28-hour pilot (about 1 GB per hour; committed memory reached 46 of 54 GB), and stopping the server freed nothing.
    - With nothing else running, the count grew by 53 a minute. Most of that was `docker.exe` and its `conhost.exe`, started by the VS Code extension host: the Docker, Docker DX, Container Tools, and Dev Containers extensions poll `docker`, and Docker Desktop was not running. Growth was about three times faster during the pilot; our code starts only two `git` calls per episode, so the rest is likely VS Code reacting to files written under `runs/`.
    - **Before a long run:** disable the Docker extensions (or start Docker Desktop), and keep `runs/` out of VS Code's file watcher (`.vscode/settings.json`).
  - **If any check looks bad, restart Windows first** (Restart, not Shut down). Signing out also restarts DWM. Updating the Intel graphics driver may stop the DWM leak.
- **Monitoring during a run:**
  - `nvitop` (`pip install nvitop`);
  - `nvidia-smi dmon -s pucm`;
  - `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 1`;
  - Task Manager → GPU 1, with an engine graph set to "Cuda".

  Per-process memory reads "N/A" under WDDM; that is expected.
- **`--jinja`:** uses the model's chat template, which honors `enable_thinking: false`. The client sends this, so Qwen3 does not spend hundreds of tokens thinking before each reply.
- **Structured output:** the client sends a JSON schema (`response_format`), which llama-server enforces with a grammar, so replies always parse.

## Episode config

Set `executor: llm` and point the `llm:` block at the server and the model file:

```yaml
executor: llm
llm:
  endpoint: http://127.0.0.1:8080
  model: qwen3-8b
  model_path: C:/Users/domin/llm/Qwen3-8B-Q4_K_M.gguf
  temperature: 0.0
  seed: 0
  max_tokens: 256
  timeout_seconds: 120.0
  retries: 2
  cache_dir: runs/llm_cache
  replay_only: false
  history_keep: 8
  pin_slots: true
```

## Reproducibility

- **Temperature 0 and a fixed seed.** Batched decoding can still vary slightly between runs, so the response cache (`llm.cache_dir`) is the reproducibility guarantee: a rerun sends the same prompts and gets the cached replies. Set `replay_only: true` to re-score a run without a GPU; a missing reply then stops the episode.
- **Wall-clock seconds are costs only** (hard rule 1). Simulated latency comes from token counts at the fixed reference rates in the `latency:` block (hard rule 3).
