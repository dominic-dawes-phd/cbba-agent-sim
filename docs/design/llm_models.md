# Local model candidates

Which open model the LLM agents run on (`decisions.md`, open question 3). The paper freezes one model; this file records the candidates and the bake-off that picks it. Serving is in `llm_setup.md`.

## What matters here

- **Speed changes wall time, never results.** Ticks come from token counts at fixed reference rates (hard rule 3), so a faster model or server only shortens runs. A different model does change behavior, which is why one model is frozen for every condition.
- **Native tool calling is not used.** The client sends a JSON schema that llama-server enforces with a grammar, so every reply parses whatever the model. What matters is following the rules and the recipe, grounding in the observation, and short outputs. A model's tool-calling score is only a proxy for that.
- **Budget:** about 7 GB of the RTX 4070's 8 GB, beside a unified KV cache of 16-32k tokens for six agents. A Q4_K_M file up to about 5.7 GB fits.

## Candidates

"Checked" means the GGUF file, size, and license were read from Hugging Face on Nov 9, 2026. "Reported" means a third-party page or model card summary; verify before relying on it.

| # | Model | Params | Q4_K_M GGUF | Fits 8 GB | Tool calling | License | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | Qwen3-8B (current) | 8B | 5.0 GB | yes (5.9 GB used with 16k KV) | native | Apache-2.0 | in use | Kitchen pilots: many plain errors (locking before walking over, releasing when blocked) |
| 2 | Qwen3.5-9B | 9B | 5.68 GB (`unsloth/Qwen3.5-9B-GGUF`) | tight | native | Apache-2.0 | checked | Feb 2026; multimodal; reported BFCL-V4 66.1 (self-reported) |
| 3 | Qwen3.5-4B | 4B | 2.74 GB (`unsloth/Qwen3.5-4B-GGUF`) | easily | native | Apache-2.0 | checked | Mar 2026; the efficiency candidate in the bake-off |
| 4 | Qwen3.5-2B | 2B | about 1.3 GB | easily | native | Apache-2.0 | reported | A third-party suite reports 14/14 tool-call tests; likely too weak for long recipes |
| 5 | Qwen3-4B-Instruct-2507 | 4B | about 2.5 GB | easily | native | Apache-2.0 | reported | Non-thinking instruct variant of Qwen3 |
| 6 | Ministral 3 8B Instruct (2512) | 8B | 5.2 GB (`mistralai/Ministral-3-8B-Instruct-2512-GGUF`) | yes | native | Apache-2.0 | checked | Official Mistral GGUF; a different family from Qwen; the quality candidate in the bake-off |
| 7 | Ministral 3 3B | 3B | about 2 GB | easily | native | Apache-2.0 | reported | Small sibling of #6 |
| 8 | IBM Granite 4.x 8B | 8B | about 5 GB | yes | native | Apache-2.0 | reported | Tool calling and agentic use listed as primary uses |
| 9 | IBM Granite 4.x 8B | 7B MoE, 1B active | about 4 GB | yes | native | Apache-2.0 | reported | Hybrid Mamba; fast decode for its size |
| 10 | Phi-4-mini-instruct | 3.8B | about 2.5 GB | easily | native | MIT | reported | Reasoning and function calling |
| 11 | Gemma 4 E4B (or Gemma 3n E4B) | about 4B effective | about 4.6 GB | yes | native | Gemma terms | reported | Sizes disagree across sources |
| 12 | LFM2.5-2.6B (Liquid) | 2.7B dense | about 1.7 GB | easily | native (own call tokens) | LFM Open License | reported | 128k context |
| 13 | LFM2.5-8B-A1B (Liquid) | 8B MoE, 1B active | about 5 GB | yes | reported unreliable | LFM Open License | reported | A third-party suite: 4/14 tool calls at Q4_K_M; older GGUFs had parser issues |
| 14 | Llama 3.2 3B Instruct | 3B | about 2 GB | easily | native | Llama 3.2 Community | reported | Older (2024) |
| 15 | SmolLM3-3B | 3B | about 1.9 GB | easily | native | Apache-2.0 | reported | Fully open training recipe |
| 16 | Salesforce xLAM-2 3B / 8B (fc-r) | 3B / 8B | about 2 / 5 GB | yes | function-calling fine-tune | CC-BY-NC-4.0 | reported | Non-commercial, which is fine for coursework but limits reuse |
| 17 | Qwen3-30B-A3B with expert offload | 30B MoE, 3B active | about 18 GB (experts in system RAM) | with `--n-cpu-moe` | native | Apache-2.0 | reported | Bigger model at 3B-active speed; 8 GB users report 30-40 tok/s with fast RAM; slower prefill |

### Added Nov 9 (small and reasoning-capable models)

Sizes and licenses checked on Hugging Face Nov 9, 2026. IBM's BFCL v4 figures are IBM-reported.

| # | Model | Params | Q4_K_M GGUF | Reasoning | License | Notes |
|---|---|---|---|---|---|---|
| 18 | Granite 4.2 3B (`ibm-granite/granite-4.2-3b-GGUF`) | 3B | 2.24 GB | switchable (`enable_thinking`; 3 modes) | Apache-2.0 | IBM-reported BFCL v4 52.41. One hands-on test found tool execution failing, since the 3B skips the agentic training stages the 8B and 30B get. IBM recommends temperature 1.0 and at least 2,048 max tokens when thinking |
| 19 | DeepSeek-R1-0528-Qwen3-8B (`unsloth/...-GGUF`) | 8B | 5.03 GB | always on (long traces) | MIT | Reasoning-heavy; a Qwen3 backbone; tool-call adherence unverified |
| 20 | LFM2.5-1.2B-Thinking (`LiquidAI/...-GGUF`) | 1.2B | 0.73 GB | always on | LFM Open License ("other") | Size floor for the study; no documented llama.cpp tool-call setup (docs point to vLLM) |
| 21 | Qwen3-4B-Thinking-2507 (`unsloth/...-GGUF`) | 4B | 2.50 GB | always on | Apache-2.0 | Thinking sibling of #5 |

**Reasoning column for the earlier rows:** Qwen3-8B, Qwen3.5 (0.8B-9B), and Gemma 4 E4B switch thinking on or off; Qwen3.5's small models default to off. A community report (April 2026) says llama.cpp forced thinking on for Qwen3.5 whatever `enable_thinking` said, and put tool calls inside the thinking block; recheck on b12211. The rest are non-reasoning instruct models. Qwen3.5-9B's BFCL-V4 66.1 is self-reported, and I found no official 4B score.

**Reasoning costs ticks.** Thinking tokens are output tokens, so latency `L` (hard rule 3) grows with them. A reasoning model pays more simulated time per decision, which is a faithful cost, but the model must be frozen together with its thinking setting.

## Bake-off (Nov 9, 2026)

Kitchen (`m2_home`), seed 0, 900 ticks, C3 and G-llm, same prompts and server flags. Configs: `configs/m2_home_{llm_cbba,greedy_llm}_{qwen35_4b,ministral3_8b}.yaml`. Qwen3-8B's numbers come from the v5.1.00 pilots.

**Downloaded and hashed** (sha256 matches the Hugging Face LFS ids):

| File | Size | sha256 |
|---|---|---|
| `Qwen3.5-4B-Q4_K_M.gguf` | 2.74 GB | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| `Ministral-3-8B-Instruct-2512-Q4_K_M.gguf` | 5.20 GB | `33e7a72cf5e6e2cfc2f2847075acc013d68bba023e35310cef86b5cf8fdca761` |

**Results** (kitchen `m2_home`, C3, seed 0, 900 ticks, identical final code, after the Windows restart; `runs/final/`). One seed each, so these are a pilot, not results.

| Model | Tasks done | LLM calls | Output tokens per call | Mean `L` (ticks) | Rejected replies | Wall s per call | VRAM (6 slots, 16k) |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | 7/10 | 412 | 61 | 2.8 | 4 | 2.2 | 5.9 GB |
| Qwen3.5-4B | 3/6 | 288 | 87 | 3.4 | 0 | 2.1 | 3.4 GB |
| Qwen3.5-4B, thinking on | 1/6 | 89 | 1,028 | 26.9 | 25 | 16.3 | 3.4 GB |
| Ministral 3 8B | 8/11 | 269 | 108 | 3.9 | 12 | 3.6 | 6.1 GB |

- **No server crashed in any run** (zero CUDA "out of memory"), so the earlier crashes were host memory, as diagnosed. Decode speed: Qwen3.5-4B about 53 tok/s, Ministral 34, Qwen3-8B 37.
- **Size did not buy quality here.** The 4B finished half as many tasks as the 8B models, with no rejected replies after the schema fix. Ministral 3 8B did the most work (8/11) and duplicated less than Qwen3-8B (24 vs 76 ticks); Qwen3-8B was fastest per decision.
- **Thinking is expensive, and did not pay.** About 1,000 output tokens per call, against 87, makes each decision cost about 27 ticks instead of 3. In 900 ticks the team made 89 decisions, not 288, and finished 1 task of 6 against 3. Stale-belief errors fell from 0.26 to 0.02, so the reasoning is more careful, but the time cost dominates. 25 replies were refused (a reasoning trace running past the token cap).
- **Thinking works under the JSON grammar.** llama-server returns the reasoning separately, and the JSON answer still obeys the schema.
- **One seed, one kitchen:** do not pick the paper's model from this. It shows which models are worth a house pilot: Ministral 3 8B and Qwen3-8B.

**Found on the way, fixed for every model** (v5.2.00):
- The reply schema now fixes each tool's number of arguments (and whole ticks for `wait`). Before, Qwen3.5-4B often wrote `"args": [` and then only whitespace until it ran out of tokens, and 159 of its 434 replies were refused. The same run now has none refused.
- A refused call's message now includes the tool's usage line.

**To rerun the bake-off** (one model at a time, nothing heavy beside it; check free RAM first, `llm_setup.md`):
1. Start the server as in `llm_setup.md` with `-m C:/Users/domin/llm/<file>` and `--cache-ram 0`.
2. Run `python -m experiments.run_single configs/m2_home_llm_cbba_<tag>.yaml --run-name bakeoff`, then the same with `greedy_llm_<tag>`. The tags are `qwen35_4b` and `ministral3_8b`.
3. Stop the server and repeat for the other model.
4. Compare with `python -m experiments.analyze runs/bakeoff`, and read decode tok/s from the server log.
