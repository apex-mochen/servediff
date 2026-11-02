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
*Servediff project note: this is G3 growth experiment — evidence that an external
developer uses servediff to locate a real difference. Comment draft lives in
`docs/issue-63705-comment-draft.md`; posting requires user approval.*
