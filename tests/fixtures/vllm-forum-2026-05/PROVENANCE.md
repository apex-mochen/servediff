# Provenance: vllm-forum-2026-05

Two posts from `discuss.vllm.ai`, read through the forum's public JSON API
(`/t/<id>.json`), not by copying a browser page. Each file is one `<pre><code>` block
verbatim, including the emoji and the Chinese punctuation; nothing is re-wrapped and no
line is added or removed.

| file | topic | posted | why it is here |
|---|---|---|---|
| `pd-nccl-sample-tokens-timeout.log` | [2637 — vLLM 多节点的PD分离部署](https://discuss.vllm.ai/t/2637) | 2026-05-14 | Prefill/decode disaggregation over P2P NCCL: the only log we have where the whole `VllmConfig` is printed by a crash (`Dumping input data for V1 LLM engine (v0.20.1) with config:`) and the only one with `set_p2p_nccl_context` overrides. Dies with `TimeoutError: RPC call to sample_tokens timed out`. |
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
- `broadcast-60s-tool-parser.log`: `api.tool_parser=Qwen3CoderToolParser`,
  `api.ignored_request_fields={'enable_thinking'}`, `engine.version=0.14.0`.

Both audit to zero shapes worth a look. The tracebacks, scheduler dumps and
`ncclCommInitRank Success` lines are noise with reasons in `audit.NOISE`: they repeat the
same crash per rank/step and carry nothing a diff between two runs could use.
