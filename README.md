# servediff

[![PyPI version](https://img.shields.io/pypi/v/servediff)](https://pypi.org/project/servediff/)
[![Python versions](https://img.shields.io/pypi/pyversions/servediff)](https://pypi.org/project/servediff/)
[![License](https://img.shields.io/github/license/apex-mochen/servediff)](https://github.com/apex-mochen/servediff/blob/main/LICENSE)

**Same flags, different server.** `servediff` shows what your vLLM or SGLang server
*actually* ran with — and what changed between two runs.

Benchmark result files only record the client side. After an upgrade, the server can
silently pick a different MoE or attention backend, change its CUDA graph capture,
fall back to another code path, or run against different torch / triton / flashinfer
builds. Today you find that by reading startup logs and bisecting by hand.
`servediff` does the reading for you.

- **Who it is for:** anyone comparing two vLLM / SGLang runs — after an upgrade, a flag
  change, or between machines — who needs to know *what differs on the server side*.
- **What it reads:** the server's startup log, and optionally a snapshot of the
  server's Python environment. Offline, no GPU, no dependencies.
- **What it does not do:** it does not say *which* difference caused a performance
  change. That still needs an A/B run. It tells you what to A/B.

## Try it in one minute (no GPU)

```bash
pip install servediff
git clone https://github.com/apex-mochen/servediff && cd servediff
servediff diff examples/issue-excerpts/vllm-47047-v0.23-default.log examples/issue-excerpts/vllm-47047-v0.23-marlin.log
```

```text
fact                             baseline                                 candidate                                status
backend.moe                      VLLM_CUTLASS (NvFp4)                     MARLIN (NvFp4)                           changed

1 difference(s); 0 fact(s) identical. Absent means the log did not print it, not that it was off.
```

That is the difference the reporter of
[vllm#47047](https://github.com/vllm-project/vllm/issues/47047) found by hand after a
v0.21 → v0.23 throughput drop. The example files are the log lines posted in that
issue (see [tests/fixtures/README.md](https://github.com/apex-mochen/servediff/blob/main/tests/fixtures/README.md)).

## Real example: same flags, vLLM 0.28.0 vs 0.30.0

Two startup logs from one RTX 4060 Laptop under WSL2, identical flags
(Qwen2.5-0.5B-Instruct, fp16, `--max-model-len 2048`, `--gpu-memory-utilization 0.45`):

```bash
servediff diff vllm-0.28.0.log vllm-0.30.0.log
```

```text
warning: baseline server failed to start: RuntimeError: UVA is not available; facts after the failure are absent because startup stopped, not because they were off

fact                             baseline                                 candidate                                status
backend.attention                <absent>                                 FLASH_ATTN                               only_candidate
backend.flash_attn_version       <absent>                                 2                                        only_candidate
backend.sampling_flashinfer      <absent>                                 disabled                                 only_candidate
cuda_graph.captures              <absent>                                 ["PIECEWISE x51", "FULL x2", "FULL x35"] only_candidate
cuda_graph.pool_gib              <absent>                                 0.17                                     only_candidate
engine.version                   0.28.0                                   0.30.0                                   changed
startup                          failed                                   ready                                    changed
startup.error                    ["RuntimeError: UVA is not available", " <absent>                                 only_baseline
uva_buffers                      <absent>                                 fallback: device memory                  only_candidate

9 difference(s); 8 fact(s) identical. Absent means the log did not print it, not that it was off.
```

0.28.0 does not start on WSL2 with default settings; 0.30.0 does, because it falls
back to device memory when pinned memory is unavailable. Upstream fixed this in
[vllm#56908](https://github.com/vllm-project/vllm/pull/56908), first released in
v0.30.0. Forcing the V1 model runner on 0.28.0 (`VLLM_USE_V2_MODEL_RUNNER=0`) makes it
start, and the remaining server-side differences are:

```text
fact                             baseline                                 candidate                                status
cuda_graph.captures              ["mixed prefill-decode, PIECEWISE x51",  ["PIECEWISE x51", "FULL x2", "FULL x35"] changed
cuda_graph.pool_gib              0.36                                     0.17                                     changed
engine.version                   0.28.0                                   0.30.0                                   changed
model_runner                     <absent>                                 V2                                       only_candidate
uva_buffers                      <absent>                                 fallback: device memory                  only_candidate

5 difference(s); 11 fact(s) identical. Absent means the log did not print it, not that it was off.
```

## Validated against real TP>1 logs

The parallel-config rules were written and checked against a real TP=8 startup log
posted in [vllm#48591](https://github.com/vllm-project/vllm/issues/48591) (v0.23.1,
8 workers, one `world_size=N` line per rank, role assignment without an ETP field):

```text
parallel.world_size              8
parallel.roles                   DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
compile.fusions                  allreduce_rms
startup                          failed
```

### When the all-reduce path changes silently

vLLM chooses an all-reduce implementation per TP group out of a candidate list, and
FlashInfer can pick its own backend on top of that. These lines are printed once at
startup and never mentioned again — which is how a version bump can move decode
through a different communication path without any flag changing. Since 0.3.7 they
are facts (this is `servediff show` on the same v0.23.1 TP=8 log):

```text
backend.allreduce_dispatch       tp:0: ['CUSTOM', 'SYMM_MEM', 'PYNCCL']                       line 37
backend.allreduce_flashinfer     mnnvl                                                        line 102
backend.allreduce_fusion         ["norm: mnnvl", "norm quantization: trtllm"]                 line 106
backend.allreduce_potential      ['NCCL_SYMM_MEM', 'QUICK_REDUCE', 'FLASHINFER', 'AITER_CU... line 37
model_runner                     V2                                                           line 39
```

The first two separate *chosen* from *possible*: FlashInfer sitting in `potential`
but not in `dispatch` means it was not used for that group. The last two record
whether FlashInfer's fused workspaces were actually built, but read their absence
carefully: since 0.29.0 the FlashInfer auto-selection line is logged at DEBUG
(`debug_once`), so on a default INFO log it may simply not have been printed. The
`all-reduce backends (in dispatch order)` line is `info_once` in 0.29.0, which makes
the dispatch chain the reliable cross-version signal. `model_runner` covers V1 vs V2
(so `VLLM_USE_V2_MODEL_RUNNER=0` runs are identifiable), and `compile.fusions` carries
`allreduce_rms` when `fuse_allreduce_rms` is on.

That makes the question raised in
[vllm#63705](https://github.com/vllm-project/vllm/issues/63705) — *did 0.29 turn
FlashInfer all-reduce on by default (#52998)?* — answerable from logs you already
keep, one per version, same TP:

```bash
servediff diff v0.28.log v0.29.log
```

On a real 4×H100 TP4+EP node (MiniMax-M2.7, one run per version) that is exactly what
the diff showed between v0.26.0 and v0.29.0: the TP group chain went from `['PYNCCL']`
to `['FLASHINFER', 'PYNCCL']` while the EP group stayed `['PYNCCL']`, V2 Model Runner
appeared, and `allreduce_rms` was off in both runs — so the two changed defaults are
the TP-side FlashInfer path and the model runner ([reported
here](https://github.com/vllm-project/vllm/issues/63705#issuecomment-6656162111)).

So a TP4+EP comparison (like the one in the
[#63705 analysis](https://github.com/apex-mochen/servediff/blob/main/ISSUE-63705-analysis.md)
— a decode-only regression across vLLM 0.29–0.32) can be scripted as:

```bash
servediff capture -o v026.log -- docker run ... vllm/vllm-openai:v0.26.0 ...
servediff capture -o v029.log -- docker run ... vllm/vllm-openai:v0.29.0 ...
servediff diff v026.log v029.log --markdown
```

## Dependency versions: `servediff env`

Logs do not print torch / triton / flashinfer versions, and upgrades of those change
server behaviour too. Snapshot each server environment with its own interpreter —
nothing needs to be installed into it:

```bash
servediff env --python /path/to/old-venv/bin/python -o old.json
servediff env --python /path/to/new-venv/bin/python -o new.json
servediff diff old.json new.json
```

Excerpt from the same two environments (20 inference-relevant rows; 74 other packages
hidden unless you pass `--all`):

```text
pkg.flashinfer-python            0.6.16.post3                             0.6.18.post1                             changed
pkg.nvidia-cuda-nvcc             13.3.73                                  13.0.88                                  changed
pkg.nvidia-cutlass-dsl           4.6.2                                    4.7.1                                    changed
pkg.torch                        2.13.0                                   2.13.0+cu130                             changed
pkg.transformers                 5.16.1                                   5.17.0                                   changed
pkg.vllm                         0.28.0                                   0.30.0                                   changed
```

Note the older vLLM environment carries *newer* CUDA toolkit wheels (13.3 vs 13.0).
Whether that matters for performance is not known — that is what an A/B run is for.
Versions are read from package metadata, so the same torch build can show as `2.13.0`
in one environment and `2.13.0+cu130` in another.

The snapshot records Python, GPU name and driver (via `nvidia-smi`), all installed
package versions, and env vars starting with `VLLM_`, `SGLANG_`, `SGL_`, `CUDA_`,
`NCCL_`, `TORCH_`, `PYTORCH_`, `FLASHINFER_`, `TRITON_`. Values of variables whose name
contains `TOKEN`, `KEY`, `SECRET`, `PASSWORD`, `CREDENTIAL` or `AUTH` are replaced with
`<redacted>`. Snapshots and logs contain local paths: review them before sharing.

## What it extracts

| Fact | vLLM | SGLang |
|---|---|---|
| version | banner / engine init | — (not printed) |
| arguments | `non-default args` | `server_args` (all args) |
| attention backend | `Using X attention backend` | `attention_backend` arg |
| MoE backend | three log formats seen in 0.23–0.30 | `moe_runner_backend` arg |
| CUDA graph | eager mode, `cudagraph_mode`, capture sets, pool size | prefill CUDA graph disabled |
| model runner, UVA fallback | `Using V2 Model Runner`; V1 runner detected via `gpu_model_runner.py` module; pinned-memory fallback | — |
| all-reduce path | chosen chain per TP group + candidate pool (`all-reduce backends (in dispatch order) … out of potential backends`), FlashInfer's auto-selected backend and its norm / norm-quantization fusion workspaces (0.3.7+) | — |
| parallel config | `world_size`, DP/PP/PCP/TP/ETP/EP/EPLB role assignment | — |
| KV cache layout | `Using LBNHC KV cache layout` (0.30+) | — |
| model loading | `Model loading took … GiB memory and … seconds` | — |
| torch.compile | `collected artifacts: …`, `torch.compile took … s in total` | — |
| startup outcome | ready / failed + root-cause error | ready / failed + root-cause error |

Every fact keeps the log line it came from (`servediff show <log>` or `--json`).
A fact the log did not print is `<absent>`: servediff never fills in defaults.
Options shared by both engines (dtype, context length, memory fraction, prefix cache,
CUDA graph, TP size) are mapped to common names, so a vLLM log and an SGLang log can be
compared, though the result is noisier than a same-engine diff.

### Pasting a report into a GitHub issue

```bash
servediff diff old.log new.log --markdown
```

Renders a GitHub-flavored table with a collapsible **Evidence** section quoting the
exact log line behind each fact — ready to paste into a regression report.

### Capturing a run in one step

```bash
servediff capture -o run1.log --python /path/to/server/bin/python -- vllm serve <model>
```

Runs the server command, tees stdout+stderr to `run1.log`, and snapshots the
environment to `run1.log.env.json` before the server starts. After two runs:

```bash
servediff diff run1.log run2.log
servediff diff run1.log.env.json run2.log.env.json
```

Exit codes: `0` no differences, `1` differences, `2` nothing recognised.

## Limits

- Tested on: log excerpts posted in vllm#47047 (v0.23) and vllm#53472 (v0.23, v0.26);
  full startup logs of vLLM 0.28.0 and 0.30.0 and one SGLang log (single-GPU dense);
  a real TP=8 startup log from vllm#48591 (v0.23.1, one line per worker + role
  assignment); a real SGLang startup log from sgl-project/sglang#4862 (v0.4.4, TP=4,
  Qwen2.5-Omni-7B — `server_args=ServerArgs(...)` keyword-call form parsed, startup
  failure + root-cause error recorded). The parallel-config and all-reduce rules are
  checked against real TP>1 logs; MoE models in full logs, recent SGLang versions and
  other engines are not yet verified.
- Parsing is regex-based on log text, which changes between versions. A missed line
  shows up as `<absent>`, never as a wrong value — please report it.

## License

Apache-2.0.
