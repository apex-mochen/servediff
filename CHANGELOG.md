# Changelog

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
