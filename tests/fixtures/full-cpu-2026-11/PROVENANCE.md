# Full CPU log, November 2026

The corpus' first log from a server with no GPU, and the first from vLLM 0.34.

| file | source | shape |
|---|---|---|
| `vllm-0.34.0rc1-cpu-avx2-qwen3.5.log` | vllm-project/vllm#67598 body, code block 2 (posted 2026-11-14 by the reporter) | full startup, 173 lines, `startup ready` |

Run: vLLM `0.34.0rc1.dev398+g153dc0464`, one AVX2 CPU (no CUDA), `vadery/Qwen3.5-0.8B-W8A8`,
`device_config=cpu`, `enable_prefix_caching=True`, TP/PP/DP = 1.

## Why it is not in the root fixture folder

`test_a_directory_is_audited_as_one_corpus_with_a_line_per_log` walks the root folder, so a log
kept here is audited by its own test. At 0.4.64 that test is the strict one — `audit
tests/fixtures/full-cpu-2026-11 --max-shapes 0` exits 0, the same bar as the root corpus and
the held-out directories, and CI runs it.

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
| `kv_cache.mamba_mode` | `align (default for Qwen3_5ForConditionalGeneration)` | 16 |
| `mm.warmup_s` | `9.45` | 33 |
| `compile.recompile_reason` | `cache load failure (backbone)` | 76 |
| `kv_cache.group_sizes` | `[640, 640, 640, 640]` | 83 |
| `cpu.threads_bind_mode` | `auto (auto_setup=True, skip_setup=False)` | 25 |
| `cpu.core_ids` | `[1, 3, 5, 7, 9, 11, 13]` | 27 |
| `cpu.omp_threads` | `7 (torch left at 7)` | 55 |
| `kv_cache.attention_block_size` | `640 (page >= mamba page)` | 56 |
| `mm.pixel_cap` | `not applied (cap_pixels_per_frame unset)` | 32 |
| `mm.encoder_cache` | `16384 tokens, 1 image item(s)` | 63 |
| `kv_cache.mamba_page_pad` | `20.75% (to match attention page)` | 57 |
| `kv_cache.cpu_node` | `4.0/30.69 GiB (node 0)` | 81 |
| `kv_cache.lcm_block_sizes` | `640` | 84 |
| `compile.inductor_disabled` | `user settings` | 126 |
| `api.chat_template_order` | `system-first required (inline system messages rewritten)` | 131 |
| `cpu.numa_auto` | `unavailable (started unbound)` | 18 |
| `cpu.numa_membind` | `MEMBIND nodes [0]` | 35 |
| `cpu.numa_migrate` | `failed (errno 1)` | 34 |
| `cpu.set_num_threads` | `skipped (thread binding owns affinity)` | 61 |

The last row is the one worth keeping: oneDNN refused a primitive descriptor, so the run
silently fell back to torch linear. Nothing on the command line says so, and a benchmark
only shows the slower result.

## What is left: 121 noise lines, 0 shapes worth a look (audit at 0.4.64)

`audit` reports `121 unread line(s), 58 shape(s)` and `nothing worth a look`. That sentence is
only as good as the noise table, and this log is where the table was caught lying.

The "prose" rule — a capitalised English sentence, no colon or equals, 40+ characters, ending
in a period — was written for text people write *around* a pasted log. Engine messages are
English sentences too, so it was silently eating settings:

| swallowed line | what it actually says |
|---|---|
| `[interface.py:1174] Padding mamba page size by 20.75% to ensure that mamba page size and attention page size are exactly equal` | the hybrid cache pads the mamba page, which is why the KV numbers do not add up to what you configured |
| `[encoder_runner.py:131] Encoder cache will be initialized with a budget of 16384 tokens, and profiled with 1 image items` | the vision encoder's cache budget and the profile batch |
| `[cpu_worker.py:255] Explicitly set (4.0/30.69) GiB for KV cache on node 0.` | the per-node CPU KV allocation |
| `[kv_cache_utils.py:851] kv lcm block sizes 640` | the block size the scheduler actually uses |

All four are facts now (`kv_cache.mamba_page_pad`, `mm.encoder_cache`, `kv_cache.cpu_node`,
`kv_cache.lcm_block_sizes`), the prose rule only applies to lines with no log preamble, and a
test pins that (`test_a_sentence_a_log_actually_printed_is_never_prose_noise`).

What remains is genuinely noise: routes and banners, the two `torch.set_num_threads` / NUMA
warnings (now facts, `cpu.set_num_threads` and `cpu.numa_*`), progress bars, access logs, and
the broadcast wait — `No available shared memory broadcast block found in 60 seconds`, 49
occurrences, kept as noise with its own reasoned rule: it is a load signature, not a
configuration one, and its text is identical every time.

`Warming up model for the compilation...` / `Warming up done.` are phase markers, so the pair's
delta is not measured — a known cost, not a hidden one.
