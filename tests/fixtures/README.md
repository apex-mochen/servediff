# Fixtures

Most files are log lines copied from public issue threads, one file per side of each
comparison. They are excerpts the reporters chose to post, not full logs: the tests prove
the parser recognises these formats, not that it finds them inside a full noisy log. The
exception is `vllm-0.33.0-real-rtx4060.log`, a full log captured from a server run here.

| File | Source (checked 2026-10-28) |
|---|---|
| `vllm-47047-v0.23-default.log` | vllm-project/vllm#47047 body, "Notes from the logs": v0.23.0 default selection |
| `vllm-47047-v0.23-marlin.log` | same, v0.23.0 with `--moe-backend marlin` |
| `vllm-53472-v0.23.log` | vllm-project/vllm#53472 body, "Runtime evidence", v0.23.0 (line wrap kept as posted) |
| `vllm-53472-v0.26.log` | same, v0.26.0 |
| `vllm-53472-v0.26-flag.log` | same issue, "Ablation": v0.26.0 + `--moe-backend flashinfer_cutlass` (lines quoted inline in the issue) |
| `vllm-0.23.1-tp8-allreduce.log` | vllm-project/vllm#48591, v0.23.1 TP=8 startup excerpt (dispatch chain, FlashInfer auto-selection, fusion workspaces, model runner) |
| `vllm-0.29-tp4-ep-h200-allreduce.log` | vllm-project/vllm#45604, v0.29.0 TP4+EP on 4×H200: the same all-reduce line for group `tp:0` and group `ep:0` |
| `sglang-0.4.4-tp4.log` | sgl-project/sglang#4862, v0.4.4 TP=4, Qwen2.5-Omni-7B — `server_args=ServerArgs(...)` form plus a startup failure |
| `vllm-0.33.0-real-rtx4060.log` | Not an excerpt: a full 170-line log from a live vLLM 0.33.0 server on one RTX 4060 Laptop (8 GB, WSL2, Qwen2.5-0.5B), written by `servediff capture` on 2026-11-08. Reached `/health` 200, served a completion, then Ctrl-C → exit 130. PIDs in the prefixes belong to that one run. |
