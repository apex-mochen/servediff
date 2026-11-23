# Provenance: vllm-forum-2026-05

Three posts from `discuss.vllm.ai`, read through the forum's public JSON API
(`/t/<id>.json`), not by copying a browser page. The two older files are one `<pre><code>`
block verbatim, including the emoji and the Chinese punctuation; nothing is re-wrapped and no
line is added or removed. `gemma4-nvfp4-hetero-heads.log` is **extracted**, not verbatim: that
post has no `<pre>` block, so `<br>` became a newline and only lines carrying `[module.py:NN]`
were kept. Its typographic quotes (`‘nvidia/Gemma-4-31B-IT-NVFP4’`) were left as served.

| file | topic | posted | why it is here |
|---|---|---|---|
| `pd-nccl-sample-tokens-timeout.log` | [2637 — vLLM 多节点的PD分离部署](https://discuss.vllm.ai/t/2637) | 2026-05-14 | Prefill/decode disaggregation over P2P NCCL: the only log we have where the whole `VllmConfig` is printed by a crash (`Dumping input data for V1 LLM engine (v0.20.1) with config:`) and the only one with `set_p2p_nccl_context` overrides. Dies with `TimeoutError: RPC call to sample_tokens timed out`. |
| `gemma4-nvfp4-hetero-heads.log` | [2674 — OOM Trying to run Gemma 4 31B NVFP4 on 2x16GB](https://discuss.vllm.ai/t/2674) | 2026-05-30 | vLLM 0.21.0 on Gemma-4-31B-IT-NVFP4, TP2 over 2×16 GB. The heterogeneous-head case: `head_dim=256` for the local layers and `global_head_dim=512` for the global ones, so the engine overrides the backend with `TRITON_ATTN` to keep one set of numerics. Also the only log where a ModelOpt NVFP4 checkpoint is detected by name, where the multimodal model is declared text-only because every modality limit is 0, and where `--disable_chunked_mm_input` is forced for bidirectional attention. |
| `broadcast-60s-tool-parser.log` | [2426 — 60秒内没找到可用的内存广播块](https://discuss.vllm.ai/t/2426) | 2026-03-07 | Same 60 s broadcast wait on v0.14.0, plus two behaviour facts no flag records: a Qwen3-Coder tool parser that was imported, and `{'enable_thinking'}` being accepted then ignored. |

Both posts are public; the topic pages are still up. If a post is edited later, this
snapshot stays as it was read, which is what a corpus needs.

Facts each file is expected to produce (asserted in `tests/test_cli.py`):

- `pd-nccl-sample-tokens-timeout.log`: `engine.version=0.20.1`, `weights.model`,
  `weights.dtype=bfloat16`, `kv_cache.max_seq_len=8192`, `parallel.tp=2`, `parallel.pp=1`,
  `parallel.dp=1`, `kv_cache.dtype=auto`, `cache.prefix=enabled`,
  `cache.chunked_prefill=enabled`, `compile.backend=inductor`,
  `nccl.env_overrides={NCCL_BUFFSIZE=4194304}`,
  `startup.cause=TimeoutError: RPC call to sample_tokens timed out.`, and no `startup` ready
  value (the run never got there).
- `gemma4-nvfp4-hetero-heads.log`: `engine.version=0.21.0`,
  `model.architecture=Gemma4ForConditionalGeneration`,
  `backend.attention_forced=TRITON_ATTN (head_dim 256 vs 512)`,
  `quant.checkpoint_format=ModelOpt NVFP4 (NVFP4), experimental`,
  `mm.text_only=all modality limits 0`,
  `mm.chunked_input=disabled (multimodal-bidirectional attention)`,
  `attn.backend=TRITON_ATTN`, `kv_cache.dtype=fp8`.
- `broadcast-60s-tool-parser.log`: `api.tool_parser=Qwen3CoderToolParser`,
  `api.ignored_request_fields={'enable_thinking'}`, `engine.version=0.14.0`.

Both audit to zero shapes worth a look. The tracebacks, scheduler dumps and
`ncclCommInitRank Success` lines are noise with reasons in `audit.NOISE`: they repeat the
same crash per rank/step and carry nothing a diff between two runs could use.
