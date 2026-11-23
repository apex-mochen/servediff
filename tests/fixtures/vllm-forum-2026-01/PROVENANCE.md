# Provenance: vllm-forum-2026-01

One post from `discuss.vllm.ai`, read through the forum's public JSON API (`/t/2254.json`),
not by copying a browser page.

The forum does **not** render long logs as one `<pre><code>` block: it splits a pasted log
into paragraphs and terminates each line with `<br>`. This file is therefore extracted, not
verbatim: `<br>` became a newline, HTML tags were stripped, and only lines carrying a real
log preamble (`[module.py:NN]`, `Traceback`, `File "…"`) were kept. Prose between the log
lines was dropped; nothing was re-wrapped. The typographic punctuation the forum returned
(the model path printed as `models–Qwen–…`, i.e. Hugging Face's `models--` separator
smarty-pantsed into an en dash) was left exactly as served — that is what a user copying
from the page pastes, and `cli.parse` now folds en/em dashes back to `--`/`---` before
matching.

| file | topic | posted | why it is here |
|---|---|---|---|
| `nemotron3-nano-v100-tp4.log` | [2254 — vLLM Tensor Parallel Workers Not Completing Initialization](https://discuss.vllm.ai/t/2254) | 2026-01-21 | vLLM 0.14.0 on Nemotron-3-Nano-30B-A3B, TP4 over four Tesla V100-SXM2. The first log in the corpus from a pre-Ampere card, so it is the only one that shows the engine casting weights down (`torch.bfloat16 -> torch.float16`), FA2 being refused for capability, and custom allreduce switching itself off because the box is PCIe-only. It also carries a hybrid-model rewrite nothing on the command line asks for (`mamba_ssm_cache_dtype` moved to float16), the padding layer the KV cache pays for, a missing fused-MoE config, and DP sync moved off NCCL by async scheduling. |

The post is public and the topic page is still up. If it is edited later this snapshot stays
as it was read.

Facts this file is expected to produce (asserted in `tests/test_cli.py`):

- `engine.version=0.14.0`, `model.architecture=NemotronHForCausalLM`,
  `model.cast=torch.bfloat16 -> torch.float16`, `gpu.bf16_support=no (Tesla V100-SXM2-32GB)`
- `kv_cache.mamba_ssm_dtype=float16 (set for NemotronH)`, `kv_cache.padding_layers=1`,
  `kv_cache.padding_waste=4.35%`, `kv_cache.attention_block_size=544 (page >= mamba page)`
- `moe.config=default (config file not found)`,
  `parallel.dp_sync_nccl=disabled (async scheduling)`,
  `backend.allreduce_reason=it's not supported on more than two PCIe-only GPUs`,
  `backend.flash_attn_blocked=FA2 needs sm80+`,
  `chat.template_warmup=["failed", "completed in 1419.3ms"]`, `server.url=http://0.0.0.0:8000`
- `parallel.roles` carries two values because the log holds a failed attempt and a retry;
  the rank string ends at `EP rank` here — this vLLM build no longer prints an EPLB rank,
  which is why the rule takes whatever rank list follows.

It audits to zero shapes worth a look. The symmetric-memory capability warning, the
`vllm serve` argument advice, the chat-template start line and its exception text are noise
with reasons in `audit.NOISE`.
