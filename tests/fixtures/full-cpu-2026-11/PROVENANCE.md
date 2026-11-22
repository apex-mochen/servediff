# Full CPU log, November 2026

The corpus' first log from a server with no GPU, and the first from vLLM 0.34.

| file | source | shape |
|---|---|---|
| `vllm-0.34.0rc1-cpu-avx2-qwen3.5.log` | vllm-project/vllm#67598 body, code block 2 (posted 2026-11-14 by the reporter) | full startup, 173 lines, `startup ready` |

Run: vLLM `0.34.0rc1.dev398+g153dc0464`, one AVX2 CPU (no CUDA), `vadery/Qwen3.5-0.8B-W8A8`,
`device_config=cpu`, `enable_prefix_caching=True`, TP/PP/DP = 1.

## Why it is not in the root fixture folder

`test_a_directory_is_audited_as_one_corpus_with_a_line_per_log` asserts that every root
fixture parses with **zero** shapes worth a look. This log is the first *full* noisy one
from a GPU-less build and does not meet that bar yet: 22 shapes remain unread. Keeping it
here keeps that invariant honest instead of lowering it, and the remainder is written down
below rather than discovered later.

## What it guards

Before 0.4.62 this log parsed as `startup ready` with **no backend facts at all** — the
GPU-shaped `Using X attention backend` line never prints on a CPU build. A diff of two CPU
runs therefore reported "no difference" whether or not one of them had asked for another
backend. The facts now read from it:

| fact | value | line |
|---|---|---|
| `backend.attention_vit` | `TORCH_SDPA (default)` | 42 |
| `backend.attention_mm` | `TORCH_SDPA` | 43 |
| `backend.gdn_prefill` | `CPU (head_k_dim=128)` | 44 |
| `backend.linear` | `CPUInt8ScaledMMLinearKernel (CompressedTensorsW8A8Int8)` | 45 |
| `backend.linear_fallback` | `oneDNN -> torch` | 53 |

The last row is the one worth keeping: oneDNN refused a primitive descriptor, so the run
silently fell back to torch linear. Nothing on the command line says so, and a benchmark
only shows the slower result.

## Remaining 22 shapes (audit at 0.4.62)

Real gaps, not shipped as facts yet:

- `kv_cache group sizes (N, N, N, N)` (hybrid KV cache layout), `Mamba cache mode is set to 'align'`
- `Multi-modal warmup completed in N.Ns`
- `Warming up model for the compilation...` / `Warming up done.` (a pair; the delta is startup cost)
- `[backbone] Compiling model again due to a load failure from ...torch_compile_cache/...`
- `OMP_NUM_THREADS=N is set; leaving Torch threads at N for serving`
- `CPU backend doesn't allow to use torch.set_num_threads after the thread binding, skip it`
- `OpenMP thread binding info:` + `VLLM_CPU_OMP_THREADS_BIND='auto', ...`
- `Warning: NUMA binding: Using MEMBIND policy ...` / `Warning: numa_migrate_pages failed`
  (two sources disagree — the API says automatic NUMA binding is unavailable, the C++
  layer says MEMBIND — so a single fact would have to pick a side)
- `[transformers] QwenNVL video processing does not apply the per-frame pixel cap ...`

Noise the classifier still misses (progress-bar variants): `Download complete:` (4),
`Reconstruction complete: |` (3).
