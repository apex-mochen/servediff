# Six SGLang reports, October 2026

The SGLang half of the corpus: excerpts from six public reports, reassembled from the issue
bodies so the numbers in `writeups/audit-noise-real-logs.md` have inputs anyone can re-read.

| file | source | blocks taken from the issue body (verbatim) |
|---|---|---|
| `sglang-41654.log` | sgl-project/sglang#41654 | env dump, the type error line, the mamba capping line |
| `sglang-42684.log` | sgl-project/sglang#42684 | the startup log, the post-start log, the `slot_layer_ids` error, the env dump |
| `sglang-43741.log` | sgl-project/sglang#43741 | the FP8 warning, the KV allocation line, the scheduler traceback |
| `sglang-46756.log` | sgl-project/sglang#46756 | the two per-rank mamba lines (DP3/TP3/EP3 and DP7/TP7/EP7) |
| `sglang-48502.log` | sgl-project/sglang#48502 | the `eager_runner.py` traceback and the `IndexError` line |
| `sglang-49206.log` | sgl-project/sglang#49206 | the `torch.compile cannot JIT compile during CUDA graph capture` traceback and its error line |

Nothing was rewritten: each file is the named code blocks joined in order, so a line that
looks truncated is truncated in the report. Filenames carry the issue number, and `PROVENANCE`
is the only place that says which block a run came from.

## What they added to the reader

These reports are the first SGLang logs in the corpus that start a server and stay up (the
named fixtures are a 0.4.4 startup, an 0.5.20 env dump, an IndexError and a NIXL failure). Six
facts came out of them:

| fact | value | why it matters |
|---|---|---|
| `mamba.cache_size` / `mamba.state_gib` | `63` / `0.18/9.00` | the cache the scheduler sizes before admitting requests |
| `sched.max_running_requests` | `12 (capped by mamba cache: 63 slots, 5 per request)` | the concurrency you set is not the concurrency you get |
| `mamba.effective_size` | `72 (raised from 24, needed 72 on DP3/TP3/EP3)` | per rank, and it differs between ranks in one launch |
| `pd.warmup` | `completed (all 1 DP ranks)` | prefill/decode disaggregation reaching ready |
| `scheduler.exit` | `code -3 (reason 11)` | the scheduler's own death sentence, beside the traceback |
| `kv_cache.fp8_scaling` | `defaulted to 1.0 (accuracy risk)` | FP8 KV cache with an implicit scale |

Also reclassified as noise, each with a reason rather than a wildcard: the Uvicorn listening
banner, the second line of the FP8 warning (`This may lead to less accurate results!` — the
first line is the fact), and `Start of pd disaggregation warmup ...` (a phase marker whose
completion line is the fact).

## Current state

`audit tests/fixtures/sglang-2026-10 --max-shapes 0` → `6 log(s): 0 with shapes worth a look`.
CI gates this directory with the same `--max-shapes 0`.
