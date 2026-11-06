# Analysis: vLLM #63705 MiniMax-M2.7 TP4+EP regression (0.29/0.31/0.32)

Source: https://github.com/vllm-project/vllm/issues/63705
Analyzed: 2026-11-02. All numbers are from the reporter's benchmark table (official
vLLM docker images, TP=4, EP enabled, 4×H100, 80 prompts, max-concurrency 8).

## The regression is entirely in the decode path

Baseline taken as v0.26.0 (last good version; also the one that needed a
compilation-config workaround to start).

| version | ttft (prefill) Δ | tpot (decode) Δ | total throughput Δ |
|---|---|---|---|
| 0.17.1 | +33.0% | +14.0% | −12.9% |
| 0.20.2 | +31.2% | +14.5% | −13.2% |
| 0.21.0 | +6.0% | +9.2% | −8.3% |
| 0.25.1 | +106.7% | +0.2% | −4.2% |
| 0.26.0 | 0.0% | 0.0% | 0.0% |
| **0.29.0** | **−0.9%** | **+22.7%** | **−17.9%** |
| **0.31.0** | **−1.8%** | **+27.9%** | **−21.1%** |
| **0.32.0** | **−0.8%** | **+22.1%** | **−17.5%** |

Three findings:

1. **Prefill is unaffected.** ttft from 0.29.0 onward is within noise of 0.26.0
   (616/611/617 vs 622 ms). Attention backend, prefill scheduling and prefix-cache
   behaviour are not suspects.
2. **Decode is uniformly worse and stable across 0.29/0.31/0.32** (+22.7 / +27.9 /
   +22.1 %). A stable ~18 ms tpot (vs 14.7 ms) across three releases points at a
   structural server-side change, not one-off scheduling jitter.
3. **The throughput drop (−18~21%) matches the decode regression**, not the old
   prefill-heavy cost of 0.17–0.21. This is a different regression from earlier
   versions' problem.

## Server-side components on the decode path to compare

On TP4+EP with a MoE model, decode latency is dominated by the **MoE expert
allreduce** (EP exchanges expert activations across TP ranks each step), **CUDA
graph replay** (decode kernel sequence captured once, replayed every step) and the
**compilation passes** that fuse kernels inside those graphs.

### Timeline facts (verified, not inferred)

- vLLM 0.28.0 released 2026-08-26. The day before,
  [vllm#51292](https://github.com/vllm-project/vllm/pull/51292) merged: *"Disable
  fuse_allreduce_rms under VLLM_BATCH_INVARIANT (non-deterministic under TP)"*.
  So `fuse_allreduce_rms` is a known source of **TP non-determinism** as of 0.28.
- The reporter's own note: v0.25.1 / v0.26.0 only started with
  `--compilation-config {"pass_config":{"fuse_allreduce_rms":false}}`. The pass
  affects startup behaviour on this exact configuration (TP4+EP).
- The regression appears in 0.29.0 — the first release after 0.28 — and persists
  through 0.31 / 0.32.

None of these server-side facts appear in benchmark result files; backend
selection, CUDA graph captures and applied fusions are only visible in the startup
log.

### Hypothesis (to A/B, not asserted)

The decode regression is consistent with a change in how `fuse_allreduce_rms` (or
the surrounding allreduce/RMSNorm fusion) behaves on TP≥2 from 0.29 onward. This is
a hypothesis: the decisive test is running the same 0.29.0 benchmark with
`--compilation-config {"pass_config":{"fuse_allreduce_rms":false}}` and comparing
tpot. If tpot returns to ~14.7 ms, the fuse path is the regression point. If not,
the startup-log diff narrows the next suspect.

## Independent confirmation of the two default changes (2026-11-05)

An external contributor grepped startup logs from the same node (4×H100, TP4+EP,
MiniMax-M2.7, this issue's serve config with a smaller `--max-model-len`), one run
per version, and reported the first cross-version server-side diff for this
configuration:
[comment](https://github.com/vllm-project/vllm/issues/63705#issuecomment-6656162111).

| fact | v0.26.0 | v0.29.0 |
|---|---|---|
| TP group all-reduce chain | `['PYNCCL']` | `['FLASHINFER', 'PYNCCL']` |
| EP group all-reduce chain | `['PYNCCL']` | `['PYNCCL']` |
| model runner | V1 (line absent) | V2 (`Using V2 Model Runner`) |
| `Enabled custom fusions` | `norm_quant, act_quant` | `norm_quant, act_quant` |

Three things follow, all from logs that were already being produced:

1. FlashInfer all-reduce **did** enter the TP dispatch chain in 0.29.0
   ([#52998](https://github.com/vllm-project/vllm/pull/52998)), while the EP group
   stayed on PYNCCL. The change is TP-side, not EP-side.
2. Model Runner V2 became active in 0.29.0
   ([#53183](https://github.com/vllm-project/vllm/pull/53183)) on this configuration.
3. `allreduce_rms` was **off in both versions**, which removes the
   `fuse_allreduce_rms` hypothesis above as a variable *between these two runs* —
   for this configuration, with these logs.

Both toggles (`VLLM_ALLREDUCE_USE_FLASHINFER=0`, `VLLM_USE_V2_MODEL_RUNNER=0`) are
queued on H100s. Whether either default costs the +22.7% tpot still needs those
numbers: a log says what is configured, not what it costs.

## How to verify (and how servediff automates the diff)

Reproduce 0.26.0 vs 0.29.0 under the same flags, keep both startup logs, diff the
server-side facts:

```
pip install servediff
servediff capture -o v026.log -- docker run ... vllm/vllm-openai:v0.26.0 ...   # with the compilation-config workaround
servediff capture -o v029.log -- docker run ... vllm/vllm-openai:v0.29.0 ...
servediff diff v026.log v029.log --markdown
```

The report lists engine version, MoE backend, CUDA graph captures and pool size,
compile fusions, model runner and startup outcome — each fact linked to the log
line it came from. `servediff` only reads logs; it does not judge which change
caused the slowdown — it tells you what to A/B.

---
*Servediff project note: G3 growth experiment. Status 2026-11-05 — an external
developer applied the log-based checks proposed in this thread to their own node and
reported the first 0.26-vs-0.29 server-side diff (section above); the checks were
found useful, but they grepped by hand rather than running servediff, so the
"external developer runs servediff on a real difference" signal is still pending.
Comment drafts live in `docs/`; posting requires user approval.*
