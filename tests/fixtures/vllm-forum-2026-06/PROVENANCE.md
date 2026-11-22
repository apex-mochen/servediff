# A CPU-only server on 384 threads, June 2026

One report, from the vLLM forum rather than a GitHub issue: the poster runs
`Qwen/Qwen3-8B` with `device_config=cpu` on two AMD EPYC 9655 sockets (384 threads, 1.5 TB RAM)
and observes that the process uses only about two threads at 100%.

| file | source | shape |
|---|---|---|
| `vllm-cpu-2x-epyc-9655.log` | discuss.vllm.ai thread 2708, first post (2026-06-09) | the pasted startup log, 22 lines, verbatim |

The paste is the whole log the poster attached, so it stops after the second broadcast
timeout — there is no `startup: ready` line because the post did not include one.

## What it added to the reader

The line ordering here is a variant nothing else in the corpus prints: the CPU KV cache is
announced as `Auto set (675.87/755.19) GiB for KV cache on node 0` rather than
`Explicitly set (…)`, and the run drops to the v1 model runner because Triton found no driver.

| fact | value |
|---|---|
| `cpu.core_ids` | 96 cores, `192`–`286`, with `reserved_cpus=[287]` left for the control plane |
| `cpu.threads_bind_mode` | `auto (auto_setup=True, skip_setup=False)` |
| `cpu.numa_membind` | `MEMBIND nodes [0]` |
| `kv_cache.cpu_node` | `675.87/755.19 GiB (node 0, auto)` |
| `kv_cache.requested_gib` / `kv_cache.non_kv_gib` | `694.77` / `18.9` |
| `kv_cache.tokens` / `kv_cache.max_concurrency` | `4921472` / `150.19` |
| `model_runner` | `v1 (v2 requires Triton)` |
| `cpu.reserved_cpus` | `[287]` |
| `triton.drivers` | `0 active (Triton disabled)` |

## Current state

`audit tests/fixtures/vllm-forum-2026-06 --max-shapes 0` → `1 log(s): 0 with shapes worth a look`.
