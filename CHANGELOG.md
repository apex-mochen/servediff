# Changelog

## 0.4.1 — 2026-11-07

- Table cells mark truncated values with `...` instead of cutting them mid-word. A
  cell that ended in `TP ran` read like the value itself rather than a clipped column.
- README examples regenerated from this build. The quickstart block now shows the two
  `note:` lines the tool actually prints, and the 0.28-vs-0.30 blocks carry the real
  `15 difference(s); 9 fact(s) identical` and `10 difference(s); 13 fact(s) identical`
  — the old numbers predated several fact rules.

## 0.4.0 — 2026-11-06

- `servediff series a.log b.log c.log` compares three or more runs in one table with
  one column per log, so a middle version is not lost between two pairwise diffs —
  the shape regression threads like vllm#53472 actually have. Supports `--markdown`,
  `--json`, `--all`, `--raw-args`, and `-` for stdin on at most one entry. Exit codes
  match `diff`.

## 0.3.10 — 2026-11-06

- `show` and `diff` accept `-` for stdin, so a log can be read straight off the
  machine that produced it: `ssh node 'cat serve.log' | servediff show -`. Only one
  side may be `-` (stdin is read once), and `--markdown` labels it `stdin`.

## 0.3.9 — 2026-11-06

- Keep the all-reduce dispatch chain when the log line was truncated. The statement
  carries a seven-item candidate list, and log sinks cut long lines; the previous
  rule required the trailing `out of potential backends: [...]` and produced nothing
  silently when it was missing. The fallback is anchored to a real log prefix
  (`[module.py:NN]`), so the same sentence quoted as prose in a pasted issue body
  does not turn into a fact.
- Pin multi-group parsing with a verbatim 4×H200 TP4+EP fixture from vllm#45604: the
  fact now carries one value per group (`tp:0` and `ep:0`) instead of collapsing to
  the last line.

## 0.3.8 — 2026-11-06

- `diff` now names why a fact could not be observed, for the two cases the tool can
  actually tell: a single-rank log (`world_size=1`, so no all-reduce group exists and
  no backend is chosen) and a v0.29.0+ log whose FlashInfer auto-selection fact is
  missing because that line is logged at DEBUG. Previously a fact absent from *both*
  logs produced no row at all and was silently counted as identical, which let
  "nothing could be seen" read as "nothing changed". Reported in text, JSON (`notes`)
  and `--markdown` output.

## 0.3.7 — 2026-11-05

- Extract the all-reduce backend selection from TP startup logs: the chosen
  dispatch list per TP group (`backend.allreduce_dispatch` /
  `backend.allreduce_potential`), FlashInfer's auto-selected allreduce backend
  (`backend.allreduce_flashinfer`) and its norm-fusion workspaces
  (`backend.allreduce_fusion`). Verified against a real vLLM 0.23.1 TP=8 log
  (vllm#48591). These lines are the only place the fused communication path shows
  up, so a silent switch from custom all-reduce to FlashInfer's path becomes
  visible in a diff instead of staying invisible.

## 0.3.6 — 2026-11-03

- Parse SGLang's `server_args=ServerArgs(...)` keyword-call form (not just the
  `{...}` dict literal): verified against a real SGLang startup log (sgl-project/sglang#4862,
  v0.4.4, TP=4) — model, TP size, memory fraction, port, attention/sampling backends
  and startup failure with root cause are now extracted instead of nothing.
- Record `startup=failed` for AttributeError/KeyError/TypeError as well as
  RuntimeError/ValueError/OOM/AssertionError; expose the SGLang `port` arg.
- Add `servediff --version` (prints the installed package version).

## 0.3.5 — 2026-11-02

- README: document the real TP>1 validation (vLLM 0.23.1 TP=8, issue #48591) and
  the #63705 decode-only regression analysis as usage examples.

## 0.3.4 — 2026-11-02

- Verified the parallel-config rules against a real TP=8 startup log (vLLM 0.23.1,
  issue #48591): every rank prints its own `world_size=N` line and the role
  assignment has no ETP field. `parallel.world_size`, `parallel.roles`,
  `compile.fusions` and `model_runner` all parse correctly; added a regression test
  distilled from that log.

## 0.3.3 — 2026-11-02

- Extract `model_loading.gib` / `model_loading.seconds` and the torch.compile
  outcome (`compile.artifacts`, `compile.total_s`); detect the V1 runner via the
  `gpu_model_runner.py` module when the "Using VN Model Runner" line is absent
  (it is V2-only). All found by continuing to dogfood the full 0.28-V1 / 0.30-V2
  startup logs: a V1-vs-V2 diff now shows the runner change, model-loading time,
  compile artifacts and CUDA-graph capture differences.

## 0.3.2 — 2026-11-02

- Extract the actual parallel configuration (`parallel.world_size` and the
  DP/PP/PCP/TP/ETP/EP/EPLB role assignment) and the KV cache layout
  (`Using LBNHC KV cache layout`, 0.30+). Found by dogfooding the parser against
  full vLLM 0.28/0.30 startup logs; the role-assignment line is the only place
  TP>1 and expert parallel appear in the log, so it makes TP4+EP configs visible
  to `servediff diff`.

## 0.3.1 — 2026-10-31

- `--markdown` now also works on `servediff env` snapshots, so a `capture`-based
  environment diff can be rendered as a paste-ready report.

## 0.3.0 — 2026-10-31

- `servediff capture -o run.log -- <server command>`: run a server, tee its
  stdout+stderr to a log file, and snapshot the environment to `run.log.env.json`
  before the server starts. Two captures become a one-command `servediff diff`.

## 0.2.0 — 2026-10-31

- `servediff diff --markdown`: GitHub-flavored report with a table and a collapsible
  **Evidence** section quoting the log line behind each fact. Ready to paste into a
  regression issue.
- PyPI badges in the README.

## 0.1.0 — 2026-10-29

First version.

- `servediff show <log>`: effective configuration facts from a vLLM or SGLang startup log,
  each with the line it came from.
- `servediff diff <a.log> <b.log>`: version, arguments, attention / MoE backend, CUDA graph
  mode and captures, model runner, UVA fallback, startup outcome and root-cause error.
  Warns when one side failed to start. Errors after the server is ready are hidden unless
  `--all`.
- `servediff env [--python PATH]`: snapshot package versions, GPU and engine env vars of a
  server environment, with secret-looking values redacted. `servediff diff` on two
  snapshots lists inference-relevant changes first.
