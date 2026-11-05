# Changelog

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
