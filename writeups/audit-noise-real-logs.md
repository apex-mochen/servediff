# What `servediff audit` is for, measured on four full logs

`servediff diff` answers "what changed between these two startups". `servediff audit`
answers the question that makes `diff` trustworthy: **which lines did the parser fail to
read?** Without it, a fact missing after an upgrade is indistinguishable from a setting the
server never printed — servediff reports both as `<absent>`, and that ambiguity is the
failure this tool exists to remove.

So the audit has one job: take a log nobody wrote rules for and separate

- **noise** — route tables, banner art, progress bars, access logs, tracebacks, shutdown
  chatter: lines that could never be a setting;
- **shapes worth a look** — lines that look like configuration and no rule reads them.

If that split is wrong, the report is useless in exactly the situation people need it: a
startup that broke.

## Before and after, on real logs

Four full startup logs, all of them live-server output rather than excerpts. Counts are
"shapes worth a look", the only group the report lists.

| log | engine · version | before 0.4.22 | after 0.4.25 |
|---|---|---|---|
| `vllm-0.28.0.log` (died at startup, UVA unavailable) | vLLM 0.28.0, 1 GPU, WSL2 | 146 | **0** |
| `vllm-0.28.0-v1runner.log` (same flags, V1 model runner) | vLLM 0.28.0 | 66 | **0** |
| `vllm-0.30.0.log` (started, then killed) | vLLM 0.30.0 | 89 | **0** |
| `vllm-0.23.1-tp8.log` (from vllm#48591) | vLLM 0.23.1, 8 GPUs, DP+EP, Eagle3 | 190 | **0** |

The last three shapes standing after 0.4.23 were real conditions rather than noise, so they
became facts in 0.4.25 instead of being classified away:

| line | fact |
|---|---|
| `Casting torch.bfloat16 to torch.float32` | `model.cast` — the checkpoint dtype was coerced; nothing in the flags says so and every result changes |
| `Module deep_gemm was found but failed to import` / `ImportError: libnvrtc.so.3` | `kernel.module_missing`, `kernel.missing_lib` — an optional kernel library is installed but unimportable, which changes the kernel set a run can choose |
| `AsyncLLM output_handler failed` / `EngineDeadError` | `startup: failed (output handler)`, `startup.error` |

A bare `AssertionError` under a failed import check stays noise: the detail is the line
above, which is now read, so counting it again would double-report one condition.

The TP=8 log reaching zero is the stricter result: every one of those 190 lines was either a
traceback frame or a configuration line, and the configuration lines are now facts
(`attn.backend`, `scheduler.chunked_prefill`, `scheduler.async`, `worker.mp_method`,
`cpu.torch_parallelism`, `spec_decode.eagle_layers`, `compile.cached_range`,
`sampling.topk_topp_backend`, `kernel.flashinfer_dtypes`, `nccl.version`,
`parallel.dp_leader`, `kernel.cuda_error`, `startup: failed (worker died)`).

## What zero does and does not mean

Zero here is a claim about **what these six logs print**, not about log space. The four above
plus two fixtures (a crashed startup and a clean 170-line run) all report
`nothing worth a look`; a log from a stack none of them covers — other engines, other
parallelism layouts, another vendor's kernels — may well print shapes no rule knows. That is
precisely why the audit prints the second group at all, and why
`servediff audit --max-shapes 20` exists as a gate: zero is measured per log, and the next
unfamiliar log is expected to break it.

## Why the first version got it wrong

Two separate bugs, both invisible on excerpt fixtures:

1. **The noise patterns described the message, but were matched against the whole line.**
   Every line of a real multi-process log looks like

   ```
   (EngineCore pid=3811) ERROR 10-28 11:26:34 [core.py:1346]   File "/opt/vllm-env/…", line 34
   ```

   and the patterns were anchored (`^\s*File "`, `^Traceback`, `\[shutdown\]`). Anchored at
   the start of a line that starts with `(EngineCore`. So a whole traceback counted as
   shapes worth a look — the report became a stack trace.

   The fix strips the preamble first — process, level, `MM-DD HH:MM:SS` (vLLM prints **no
   year**, which is why the first attempt only halved the count), `- INFO -`,
   `module.py:line` — and then matches the message. A third pass drops a trailing
   `# comment`, because traceback echoes carry one (`_find_cuda_home()  # CUDA home`).

2. **Rules pinned to the newest wording silently lost the older log.** Two found by this
   same audit:
   - `Model loading took 1.14 GiB and 36.34 seconds` — 0.23 prints no `memory`, and the rule
     required it. On 0.23 both `model_loading.gib` and `model_loading.seconds` vanished, and
     `diff` would have called that "not printed".
   - `Filesystem type for checkpoints: FUSE.OFS.` — `\w+` stopped at the dot and took the
     checkpoint size and available RAM with it: one regex, three facts.

## What guards it now

- A 43-line card of a **real crashed startup** is a fixture
  (`tests/fixtures/vllm-crash-traceback.log`). The test asserts the audit reports
  `nothing worth a look` on it, and that `RuntimeError: UVA is not available` stays consumed
  by the `startup.error` rule rather than appearing as unread.
- The same test asserts three prefixed **real settings** are *not* noise
  (`Using AttentionBackendEnum.FLASHINFER backend.`, `Chunked prefill is enabled with
  max_num_batched_tokens=2048.`, `Model loading took 1.2 GiB and 1.2 seconds`). Widening a
  noise pattern has to break the suite — a rule that quietly hides a missed fact is worse
  than a report that lists too much.
- A test asserts the TP=8 lines above become facts **and** that the audit reports none of
  them as unread, so parser and audit cannot drift apart.
- `servediff audit --max-shapes N` turns this into a gate: it exits `1` when more shapes need
  a look than you allow, so a pipeline fails on the upgrade that starts printing
  configuration the tool still cannot read.

## Limits of the claim

These four logs are from one machine's WSL2 runs plus one posted TP=8 log.
`nvcc`, other engines, and other parallelism layouts are unmeasured, and a log from a
different stack may well print shapes none of these rules know. That is what the audit is
for: run it on your own log, and paste the report into
[the issue form](https://github.com/apex-mochen/servediff/issues/new?template=missed-log-line.yml)
— the second group is the useful half.
