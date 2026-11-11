# Changelog

## 0.4.50 — 2026-11-11

Positioning only — no code change.

- **The first sentence now leads with one log.** It said *"Same flags, different server"* and
  put the two-log comparison at the door. Measured over 200 open issues (100 vLLM + 100
  SGLang, created 2026-10-20..11-05, all comments read): exactly **1** thread carried anything
  `servediff` can read, and **0** carried two. The old pitch was asking for raw material that
  issue threads do not contain. It now reads *"One startup log in — the settings the server
  actually ran with come out"*, and the two-log comparison is the second step.
- `demo` stays as the zero-input first command (it needs no log at all); a second block,
  **"Already have a log? `servediff show serve.log`"**, now sits right under it, because that
  is the path a reader with a crash behind them actually takes. `show` prints each fact with
  its line number and exits `0` as soon as it read something — verified against the shipped
  0.33.0 log.
- `README.zh-CN.md` carries the same change.

## 0.4.49 — 2026-11-11

Documentation only — no code change.

- **The exit-code table now matches what 0.4.46–0.4.48 actually do.** `diff` gained *one log
  with nothing to compare it to* as a documented `2`; `env` gained *or the path cannot be
  written*; `capture` now says `2` when the paths cannot be written or the command cannot
  start. Each of these was already true in the code, and the table was the part that lied.
- **`README.zh-CN.md` gets the one-log path.** Its "use your own log" section started with
  `capture`, which needs a server and a Ctrl-C, while most people already have a log. It now
  starts with `show serve.log` / `show serve.log --json > facts.json` /
  `diff today.log --against facts.json`, and leaves `capture` to *the failure is ahead of you*.
  Re-ran all three against the published 0.4.48 wheel: exit `0`, `0 difference(s); 56 fact(s)
  identical`.

## 0.4.48 — 2026-11-10

- **A single-byte console no longer turns a report into exit 1.** `audit` echoes log lines
  back, and a real log carries a box-drawing banner (`██▄█▀`). On a default Windows console
  (cp1252) that raised `UnicodeEncodeError`, so `audit` exited `1` — and `1` means *the inputs
  differ*, so a gate read a console limitation as a config change. `show --json` failed the
  same way; because it writes the payload in one `print()`, nothing was written and
  `--against` then read an empty facts file. Output streams now keep the console's own code
  page and let a character it cannot encode come out as `?`.
- Found by the step added in 0.4.47: CI's Windows job runs under that code page, and
  `python tools/verify_readme_exit_codes.py` reported `19/21` there while both Linux jobs
  reported `21/21`. The failure was real, not a test artifact.
- 2 new tests (116 total): `audit` and `show --json` under `PYTHONIOENCODING=cp1252` exit `0`
  with no `UnicodeEncodeError` and no `Traceback`.

## 0.4.47 — 2026-11-10

- **`env -o` and `capture -o` say so when the path cannot be written.** Pointing `-o` at a
  directory that does not exist raised `FileNotFoundError` and exited `1` — and `1` is documented
  as "the inputs differ", so a script that branches on it reads a typo as a finding. Both now
  exit `2` with `cannot write <path>`. For `capture` this happens before anything is launched,
  so it is a usage error, never the server's exit code.
- **`tools/verify_readme_exit_codes.py`: the command table is now a test.** It builds its own
  fixtures, runs every documented case (21 of them), and fails on a code that does not match or
  on a traceback. CI runs it on each matrix job, so a README promise cannot drift from the tool
  without going red.
- 2 new tests (114 total): `env` and `capture` to an unwritable path are exit `2`, with no
  `Traceback`.

## 0.4.46 — 2026-11-10

- **`diff` with one log now explains itself instead of raising.** `candidate` became optional
  in 0.4.42 only because `--against` supplies the other side; without either, the code reached
  `_read_log(None)` and a newcomer's very first command — `servediff diff startup.log` — died
  with `AttributeError: 'NoneType' object has no attribute 'read_text'`. It now exits `2` with
  `diff: one log is not a comparison -- pass the second log, or '--against facts.json'
  (record it from a log you kept: servediff show --json <log> > facts.json)`.
- Found by walking the README as a stranger would: fresh venv, `pip install servediff`,
  `servediff demo`, then "diff the one log I have". No code path was changed beyond that
  branch; every other exit code stays as it was.
- 1 new test (112 total): one log without `--against` is exit `2`, names both routes, and
  asserts `Traceback` is not in stderr.

## 0.4.45 — 2026-11-10

- **`diff <dir> --against facts.json`: gate a whole corpus in one command.** A CI matrix
  writes one startup log per row; six diffs bury the answer in six tables. Give `diff` the
  directory and it prints one line per log — `N difference(s); M fact(s) identical` plus the
  first changed keys — then one summary line and one exit code. `--only` / `--skip-absent` /
  `--all` narrow every row the same way they narrow a single diff, and `--json` returns
  `{against, facts, logs, logs_differing}` for a machine.
- **A blank or truncated artifact in that directory exits 2.** A log nothing can be read
  from is a broken gate input, not a run that happened to match; an empty directory is
  likewise "no *.log inside", not a clean pass.
- 4 new tests (111 total): per-log lines and one exit code, `--only` narrowing the corpus,
  the empty directory, and the blank artifact.

## 0.4.44 — 2026-11-10

- **`diff --only PREFIX` (repeatable): a gate on the decisions you own.** Package pins move
  every week; the startup decisions — attention/MoE backend, CUDA graph mode, parallelism —
  are what a version bump actually changes. `--only backend. --only cuda_graph.` compares
  those rows and nothing else, and the note says how many facts were left out, so a narrowed
  pass cannot be read as a clean one.
- **A prefix that matches nothing exits 2, not 0.** `--only cuda_graph` on two logs that
  never printed it is a green gate watching nothing — the same failure as an unreadable
  baseline. It says so and fails.
- `--only` is rejected on `servediff env` snapshots rather than silently ignored: those are
  compared whole today.
- 4 new tests (107 total): narrowing, two prefixes at once, the no-match exit 2, and the
  snapshot rejection.

## 0.4.43 — 2026-11-10

- **Corrected the size figures in the 0.4.42 example.** 0.4.42 said a facts file is 4 KB
  "versus 40 KB for the log"; measured on the fixtures it is **5 190 B for 19 facts**, the
  ROCm collect_env paste it came from is **9 904 B**, and a full startup log is 33–42 KB.
  The point still holds — you commit kilobytes instead of the log — but the numbers in
  the README and the release body now match what `Get-Item` reports.
- No code change: same behaviour, same 103 tests, same exit codes.

## 0.4.42 — 2026-11-10

- **`servediff diff <log> --against facts.json`: one log is enough to gate.** A CI job has
  today's startup log; nobody keeps the log from the run they accepted three months ago.
  `servediff show accepted.log --json > facts.json` records those facts in a few kilobytes
  (19 facts from a collect_env paste is 5.2 KB, against 9.7 KB for that excerpt and 33–42 KB
  for a full startup log), and `--against`
  diffs today's log against them with the same rows, the same `<absent>` semantics and the
  same exit codes as a two-log diff. Passing a second log alongside `--against` is a usage
  error rather than a guess.
- **`show --json` now also emits `values`, the lossless list.** `value` collapses to a
  scalar when a fact was printed once and to a list when it was printed several times, so
  reconstructing facts from it turns `cuda_graph.captures` (one list-valued fact) and
  `backend.attention` (two conflicting backends) into the same shape. `--against` reads
  `values`, so a round trip through the file cannot merge or split a fact; files written
  before 0.4.42 still load, with their `value` taken as one value.
- **Recorded facts cannot explain an absence.** In `--against` mode the baseline gets no
  absence note — a facts file has nothing to say about why a line was never written — and
  the output names how many facts the baseline holds, so a narrowed comparison is never
  read as a clean one. Unreadable or non-facts input exits `2`, not `1`: a broken baseline
  is a broken gate, not a finding.
- `--raw-args` is ignored when the baseline is a facts file, because a facts file records
  facts and not the resolved command line; the flag no longer invents `raw_arg.*` rows.
- 3 new tests (103 total) covering the round trip, the multi-valued fact, and the three
  new exit-2 cases.

## 0.4.41 — 2026-11-10

- **The second CI gate is now on the landing page.** `--baseline` shipped in 0.4.40 and was
  documented only inside the audit section; the README that PyPI renders now carries the
  two-step recipe near the top — write the baseline once on a log you have read, then gate
  with `--max-shapes 0` against it — with the exit codes spelled out: `1` a new shape, `2`
  the baseline or the log could not be read, which is a broken gate rather than a finding.
- No parser or CLI change: this release is the documentation the previous one needed, and
  the PyPI long description is this README.

## 0.4.40 — 2026-11-10

- **`audit --baseline` / `--write-baseline`: a gate that stays on.** `--write-baseline
  accepted.json` records the shapes a run saw; `--baseline accepted.json` reports and gates
  only on shapes that are not in it, printing how many were hidden (`N shape(s) in the
  baseline (M line(s)) were not counted`). Without this, `--max-shapes` goes red the first
  time the server prints a line no rule reads and stays red forever — a gate that is always
  red gets deleted, and then nothing is checked. A baseline turns it into the question that
  actually matters: did this upgrade make the server print configuration nothing reads.
- **A shape in the baseline is still reported as unread**, because it is: accepted once, not
  read. The single-log report says `nothing new: every shape worth a look is already in the
  baseline`, not `every unread line is noise`, which would misdescribe what happened — that
  wording is now pinned by a test.
- **`--json` gains two fields only when `--baseline` is given** (`baseline_shapes_not_counted`,
  `baseline_lines_not_counted`), so a pipeline reading the default document sees the same
  fields it saw in 0.4.38. `--max-shapes` and the exit code behave as before: `--json` still
  changes the format, not the code, and `--json` with `--markdown` is still exit 2.
- **A file that is not a baseline is exit 2**, not a traceback and not exit 1: an unreadable
  or malformed `--baseline` is an input error, the same rule the rest of `audit` follows.
- 100 tests.

## 0.4.39 — 2026-11-10

- **A third held-out batch, collected five days later, and what it found.** `created:>2026-11-05`
  across both repositories produced 28 usable pastes (3306 lines), now in
  `tests/fixtures/held-out-2026-11b/` with a `PROVENANCE.md`. The rules as of 0.4.38 reported
  17 of those 28 files: 191 shapes.
- **The `[conda]` package inventory.** 164 of the 191 shapes were one table — `collect_env`
  lists every installed wheel twice, once as `[pip3] name==version` and once as a `[conda]`
  row, and only the pip form was covered. Four reports carried ~41 rows each, and every row
  reads like a version fact. The `[conda]` table is the same inventory in another column
  format, so it gets the same policy; `[conda] Could not collect` was already there.
- **The rest of a pasted reproducer's expression lines**: an f-string (`f"…{…}"`), a tuple
  assignment with qualified names (`layer.output_size_per_partition, layer.input_size_per_partition
  = N, K`), the argument list that only closes the call above it, a one-line call chain
  (`tf.AutoModel…from_config(cfg).to(…).save_pretrained(path)`), a nested call closed twice
  (`asyncio.run(main(parser.parse_args()…))`), a bare name closing a parenthesis, a lone `[`,
  an annotation-only assignment, a `!= (…)` continuation, an `and … )` continuation, a tuple
  of layer specs ending the `for … in (` header, `async def`, and `finally:`.
- **One candidate measured and thrown away again.** `^\(.*\)$` — any parenthesized line —
  looked like it covered the reproducer's Path expression, and it ate 31 real vLLM log lines,
  every one a `(EngineCore pid=…)` entry. The adopted pattern requires the closing `)` to be
  followed by `.name`, which is how a value built in brackets gets used. The engine line is
  now a control in the tests.
- All five corpora report zero: 21 fixtures, 11 October pastes, 7 November pastes, these 28,
  and the 6 logs from a real machine. 96 tests.

## 0.4.38 — 2026-11-10

- **`servediff audit --json`.** The audit prints a table for a person; a CI gate should read
  fields. `--json` emits one object with, per log, `path`, `unread`, `shapes` and the
  `shapes_worth_a_look` entries (shape and count), plus `shapes_worth_a_look` (the number
  `--max-shapes` gates on) and `logs_with_shapes_worth_a_look`. The counts come from the
  same pass that renders the text, so the two cannot disagree.
- `--json` changes the format, not the exit code: `--max-shapes` still decides it. Passing
  `--json` with `--markdown` is an error (exit 2) rather than one silently winning.
- Internally the audit's summary now carries the full log label and shortens it only for the
  table, so the JSON path is the path that was read.
- 92 tests.

## 0.4.37 — 2026-11-10

- **A second held-out batch, a month later, to test whether the rules learned logs or learned
  October.** The same collection (`created:>2026-11-01`) produced twelve blocks; five were
  already in the October batch, so `tests/fixtures/held-out-2026-11/` holds the seven new
  ones (713 lines) with a `PROVENANCE.md`. Six were clean on the first try; the seventh had
  one line.
- **A conditional expression standing alone is not a log line.** `run(a) if a.cmd == "run"
  else compare(a)`, lifted out of a reproducer script, is the argument-list shape one form
  further out: it picks between two values, so it has no subject. The obvious pattern
  (contains `if` and `else`, ends in `)`) was measured and rejected — it also ate the warning
  *"Using FlashInfer all-reduce if available else falling back to the custom kernel (set
  VLLM_ALLREDUCE_USE_FLASHINFER=0 to disable)"*, which is a statement and belongs in the
  report. The adopted pattern requires the line to begin with a call; it matches one line
  across all four corpora.
- The old result is reproducible: `v0.4.36` reports one of seven on this directory.
- 89 tests.

## 0.4.36 — 2026-11-10

- **The held-out corpus ships, so the claim about it can be rerun instead of believed.**
  `tests/fixtures/held-out-2026-10/` holds the eleven pastes taken fresh from vLLM and SGLang
  issues in October 2026 (1291 lines), with a `PROVENANCE.md` naming each issue and stating
  what the sample does not prove. A test audits that directory and asserts
  `11 log(s): 0 with shapes worth a look`; checking out `v0.4.32` and rerunning the same
  command reports eight of eleven, which is the measurement 0.4.33 was about.
- Directory expansion stays non-recursive, so `audit tests/fixtures` still reports the
  twenty-one main-corpus logs and not thirty-two; the two corpora are audited separately.
- 87 tests.

## 0.4.35 — 2026-11-10

- **The last four "worth a look" lines in the held-out corpus are gone.** They were the same
  pasted script in a harder form: an argument list standing alone
  (`8, 96, 128, conv_kernel_size=4, num_spec=5`, `groups, KVCacheLayout.BLHNC`) and a
  continuation that closes parentheses without starting a statement
  (`…routing_input_mode=…), wp))(ml)`). A log line is a statement — it starts with a word and
  says something about it — so it never consists of arguments alone. Two patterns, both
  anchored on that: the line must be arguments only, and the first token must be a bare name.
  That anchor is what keeps `NVIDIA_DRIVER_CAPABILITIES=compute,utility` a fact.
- Eleven pastes taken fresh from vLLM and SGLang issues (October 2026): **0 with shapes worth
  a look.** 21 shipped fixtures: 0. Six real startup logs: 0.
- Measured before adopting: both patterns hit **zero** lines across the 21 fixtures and the six
  real startup logs, and the `NAME=value` counter-examples were checked explicitly. 86 tests.

## 0.4.34 — 2026-11-10

- **The log that caught the padded-colon bug is now fixture 21.** `vllm-66525` is a real
  `collect_env` pasted into a vLLM issue from a Debian 12 box with no GPU: unpadded version
  lines, `torch 2.10.0+cpu`, `HIP runtime version: N/A`, `GPU models and configuration: No
  CUDA`. Every fixture the repo carried was a hand copy that had kept the padded colons, so
  the 0.4.33 bug was invisible here; this one pins it down against the real form, and adds the
  first corpus entry from a machine with no accelerator at all.
- A test now asserts that log reads `env.torch`, `env.os`, `env.python` and `gpu.model`, and
  that it contributes nothing to "worth a look" — so a future rule regression fails in the
  suite instead of in someone's issue.
- Corpus count in README, README.zh-CN and the writeup: twenty → twenty-one. 85 tests.

## 0.4.33 — 2026-11-09

- **Every `collect_env` rule was pinned to the padded colon.** They were written
  `^PyTorch version\s+:\s*(\S+)`, which requires whitespace before the colon; ROCm's own dump
  aligns its colons with spaces, and vLLM's real `collect_env` prints
  `PyTorch version: 2.10.0+cpu` with none. So on the output every bug report actually pastes,
  `env.torch`, `env.torch_rocm`, `env.hip`, `env.os`, `env.python` and `engine.version` were
  never read. `\s*` on those rules; `env.allocator` reads `Caching allocator config`.
  Found by auditing eleven issue pastes taken fresh from vLLM and SGLang (created after
  2026-10-20) instead of the repo's own fixtures — 8 of 11 came back with lines worth a look.
  After this version: 2 of 11, both script pastes, four lines.
- Noise, same corpus: SGLang's macOS report (`MPS available`, `macOS Version`, `Apple
  Silicon`, `Metal Support`, `Unified Memory`, `CPU Cores (…)`, `GPU Cores`), a multi-GPU row
  written `GPU 0,1,2,3: NVIDIA H20`, a container runtime's `brand=…driver>=…` capability
  list, a topology-matrix header row, a reporter's parenthetical note, `Is XPU available`,
  `CPU:` / `Versions of relevant libraries:` headers with a colon, and a plugin source paste
  (`WTYPE = …`, `tensor.offset`, `{`, `sys.modules[…] = …`, `(Path(…) / "x").write_text(`).
- `MAX_MODEL_LEN=4096` inside a pasted script is still a fact (`env.var.MAX_MODEL_LEN`), and
  `GPU models and configuration: No CUDA` is still a fact (`gpu.model`): a script's code is
  noise, a script's call is not, and a CPU box saying it has no GPU is a statement.
- README numbers unchanged: 48/14, `--skip-absent` 3/14/45, 30/33, `--skip-absent` 22/33/8.
  The 20-log corpus and the six real startup logs still report `0 with shapes worth a look`.
  84 tests.

## 0.4.32 — 2026-11-09

- A reproducer script pasted into an issue body is no longer reported as lines worth a look.
  sglang#49123 pastes a 195-line benchmark harness, and the audit called 38 of its lines
  "worth a look" — every one of them the script: `def run_marlin(...)`,
  `handles[f] = safe_open(...)`, a module docstring. Those shapes (definition, tuple unpack,
  subscripted assignment, bare `print(...)`, a line ending in a comma, the prose between two
  `"""`) are noise now: that paste goes from 38 to 1, and the one left is a lambda's
  continuation. A shell-style `MAX_MODEL_LEN=4096` in the same paste is still read as
  `env.var.MAX_MODEL_LEN`: a script's code is noise, a script's call is not.
- Fixed a keyword pattern that was almost right: `^\s*(?:if |elif |else |…)\b` failed on
  `if "W4A4" in name:` because the space sat inside the alternative and `\b` tested a
  space→quote boundary. Loop and branch lines whose condition is a quoted string stayed in
  the report; now they do not.
- The 20-log corpus and the six real startup logs still report `0 with shapes worth a look`.
  82 tests.

## 0.4.31 — 2026-11-09

- A numeric difference now prints its magnitude as well as the word: `changed  +45.0%`,
  `changed  -4.6%`. `changed` alone reads the same whether torch.compile moved 4% or 400%,
  and the magnitude is what a reader decides on. Text values keep the bare status.
- A per-worker measurement is one fact again. On an 8-GPU box every worker logs
  `Model loading took 1.14 GiB and 36.3 seconds` with its own jitter, so one log held eight
  different values — and two runs of the same box could never agree. `model_loading.seconds`,
  `model_loading.gib`, `weights.load_s` and `weights.available_ram_gib` are now read from the
  lowest rank present, with the other ranks kept in the evidence line. Two *different*
  measurements from the same rank (TP0 loading a backbone and a draft head) are still kept
  apart.
- 80 tests. The two README comparisons still report 48 / 14 and 30 / 33 unchanged: the
  magnitude is added after the status, it does not move the counts.

## 0.4.30 — 2026-11-09

- `servediff audit` takes a directory (or several paths) and answers for the whole corpus:
  one line per log with its unread count, shape count and verdict, then only the rows that
  matter. `servediff audit tests/fixtures` audits all twenty logs in one command, which is
  the claim this project rests on, now checkable in one line instead of twenty runs.
- One more report from the field: vllm#65873, a GB10 (aarch64) box running
  `Nemotron-3.5-Super-VL-120B-A12B-NVFP4` with native MTP. Its nine shapes worth a look were
  all absolute paths to `libcudnn*.so` — an `ldd` dependency dump, which is machine
  inventory, not a setting. Now noise; the log reports nothing worth a look.
  A missing library in an `ImportError` stays a fact.
- Twenty fixtures, 76 tests.

## 0.4.29 — 2026-11-09

- Two more reports, both on hardware the earlier logs did not cover: a GB200 NVL node
  (vllm#66192, aarch64, SM100, torch 2.14.0) and an RTX PRO 6000 Blackwell box
  (vllm#66193). Each carried 9 and 13 "shapes worth a look" that were all machine
  topology — the InfiniBand NIC grid (`NIC0: mlx5_0`, one row per NIC, `NIC Legend:`) —
  and an empty `GPU models and configuration` fact that said nothing.
- An empty value is an absence now. Multi-node images print
  `GPU models and configuration :` with nothing after the colon and then one
  `GPU 0: NVIDIA GB200` per card; that produced `gpu.model = ""`, which compares equal to
  another log's empty string instead of being absent.
- The per-card line is read as the model when the aggregate is silent, and the aggregate's
  own per-card prefix is dropped: `GPU models and configuration : GPU 0: NVIDIA RTX PRO 6000
  Blackwell Server Edition` now yields `NVIDIA RTX PRO 6000 Blackwell Server Edition`, not
  `GPU 0: NVIDIA …`.
- Two new fixtures, two new tests, 72 tests.

## 0.4.28 — 2026-11-09

- The second class of misses, found the same way: twelve logs from issues the tool had never
  seen — six vLLM (a production image, an eight-card MI355X node, a gVisor run, three short
  excerpts) and six SGLang. Before: 61 / 45 / 44 / 51 "shapes worth a look" on the four full
  vLLM logs and 39 / 12 / 11 / 3 / 2 on SGLang. After: nothing worth a look on all twelve.
- Facts: the `[pip3]` rule now covers `flashinfer-python` (it decides kernels; the rest of
  the pip inventory does not). Every `NAME=value` the `collect_env` block ends with becomes
  `env.var.NAME` — the six-entry whitelist became a general rule — minus `CUDA_VISIBLE_DEVICES`,
  `NVIDIA_VISIBLE_DEVICES` and `VLLM_CACHE_ROOT`, which differ between two boxes started with
  the same flags.
- `Could not collect`, `N/A`, `Not Set` and `unknown` are absences, not values. A CUDA box
  was reporting `env.rocm = "Could"` because the rule captured one token of
  `ROCM Version : Could not collect`, so two CUDA boxes compared equal on a value neither had.
- SGLang: its own `name: value` environment dump (`sglang:`, `PyTorch:`, `sglang-kernel:`,
  `flashinfer_python:`, `triton:`, `CUDA_HOME:`, `NVCC:`, `CUDA Driver Version:`,
  `ulimit soft:`), the brackets around its timestamp (`[2026-10-06 00:15:19] …`, stripped as
  a prefix by `parse`, `audit.message` and the noise table alike), `cache.radix` from
  `Tree cache initialized: …`, `transfer.nixl_backend` from `NIXL KVManager initialized
  with backend: UCX`, `startup: failed (scheduler)`, and `IndexError` as a startup cause.
- Audit noise: the pip inventory, GPU topology grid / rocm-smi weight-hops-link matrices /
  `Legend:` / Hypervisor lines, `GPU N:` and `GPU[N] (Topology)` headers, variables printed
  with a name but no value, prose pasted between the logs (Markdown bullets, bold sentences,
  issue-template headings), and source echoed around a traceback (Triton patch diffs,
  frame assignments, bare closing brackets).
- Six new fixtures (`vllm-0.32.0-cuda-collectenv.log`, `vllm-0.33.0rc1-rocm-mi355x.log`,
  `vllm-0.30.0-gvisor-cuda-collectenv.log`, `sglang-0.5.20-env-dump.log`,
  `sglang-0.5-nixl-startup-failure.log`, `sglang-0.5-indexerror-per-request.log`), five new
  tests. 70 tests.
- `diff vllm-65250.log vllm-63505.log` now reports 23 differences / 5 identical and names
  the ones that matter between a gVisor run and a production image: `TORCH_CUDA_ARCH_LIST`,
  `VLLM_BUILD_COMMIT`, `VLLM_IMAGE_TAG`, `CUDA_HOME`, plus the threads and FlashInfer-sampler
  settings only one side printed.

## 0.4.27 — 2026-11-09

- Docs only: the audit writeup now carries the ROCm measurement (vllm#65034: 175 shapes
  worth a look → facts plus noise, nothing left over), and both READMEs name the covered
  stack. No behaviour change — 0.4.26 is the release that reads those lines.

## 0.4.26 — 2026-11-09

- ROCm support, found by auditing a log the tool had never seen: the `collect_env` block
  pasted into vllm#65034 is ~180 lines and each one reads like a setting, so `audit`
  reported 175 "shapes worth a look". It now yields facts — `engine.version` +
  `engine.git_sha`, `env.torch`, `env.torch_rocm`, `env.rocm`, `env.hip`, `env.os`,
  `env.python`, `gpu.model`, `env.pip.{torch,triton,transformers,numpy}` and
  `env.var.{TORCHINDUCTOR_CACHE_DIR,PYTORCH_ROCM_ARCH,MAX_JOBS,LD_LIBRARY_PATH,…}` — and
  the block's scaffolding (section rules, CPU inventory, the flag dump, per-arch
  `amd-torch-device-gfx*` wheels, the wrapped `LD_LIBRARY_PATH` tail) is counted as noise.
  That log now reports nothing worth a look.
- The generic `version X.Y.Z` banner rule no longer takes `CMake version : version 3.31.10`
  as the engine version.
- A launch command pasted into a report (`vllm serve …` plus `--flag value \` lines) is
  noise: the server prints its own args as "non-default args".
- New fixture `tests/fixtures/vllm-0.29.1-rocm-collectenv.log` (176 lines, vllm#65034).
  65 tests.

## 0.4.25 — 2026-11-09

- The last real shapes are facts now, so every full log in this repo audits clean: `Casting
  torch.bfloat16 to torch.float32` → `model.cast` (a coercion no flag records, and it changes
  every result), `AsyncLLM output_handler failed` → `startup: failed (output handler)`,
  `EngineDeadError` (fully qualified as `vllm.v1.engine.exceptions.EngineDeadError`) →
  `startup.error`, and an installed-but-unimportable optional kernel library (`Module …
  was found but failed to import`, `ImportError: … cannot open shared object file`) →
  `kernel.module_missing` / `kernel.missing_lib`.
- Audit: `outputs = await engine_core.get_output_async()` is a frame echo and counts as
  noise; a bare `AssertionError` under a failed import check does too, since the detail is
  the line above that is now read — counting it again would double-report one condition.
- All four full real logs (vLLM 0.28.0 crash, 0.28.0 V1 runner, 0.30.0 killed, 0.23.1 TP=8)
  and both real-log fixtures report `nothing worth a look`. That is a claim about what those
  logs print, not about every log: `writeups/audit-noise-real-logs.md` now says so.
- 63 tests.

## 0.4.24 — 2026-11-09

- New writeup [writeups/audit-noise-real-logs.md](writeups/audit-noise-real-logs.md): what
  `servediff audit` is for, and the before/after counts on all four full real logs
  (146 → 1, 66 → 5, 89 → 4, 190 → 0), with the two bugs that caused them — noise patterns
  matched against the prefixed line instead of the message, and rules pinned to the newest
  wording that silently dropped the older log's facts. Linked from both READMEs; `writeups/`
  now ships in the sdist so the links hold inside the tarball.
- No behaviour change.

## 0.4.23 — 2026-11-09

- Read the configuration a real TP=8 startup log prints. On the vLLM 0.23.1 log from
  vllm#48591, `servediff audit` went from **70 shapes worth a look to zero**: new facts for
  `Using AttentionBackendEnum.FLASHINFER backend.`, `Chunked prefill is enabled with
  max_num_batched_tokens=…`, `Asynchronous scheduling is enabled.`, the forced worker start
  method, `Reducing Torch parallelism from 96 threads to 1`, `Using Eagle3 auxiliary layers
  from config: (…)`, `Cache the graph of compile range (…)`, `Using FlashInfer for top-p &
  top-k sampling.`, `FlashInfer resolved query dtypes: …`, `vLLM is using nccl==…`, and the
  DP group leader's ranks.
- Two rules were quietly missing older logs rather than reporting them as absent:
  `Model loading took 1.14 GiB and 36.34 seconds` (0.23 prints no `memory`) and
  `Filesystem type for checkpoints: FUSE.OFS.` (`\w+` stopped at the dot, taking the
  checkpoint size and available RAM down with it). Both now read on 0.23 and 0.30.
- Recorded outcomes that degrade a run instead of stopping it: `CUDA error
  (…flash_api.cpp): invalid configuration argument` as `kernel.cuda_error`, and a worker
  that dies while the API server keeps printing as `startup: failed (worker died)`.
- Audit noise: a wrapper script's own output (`>>> …`, `[spec_bench] …`, `!!! …`) is noise —
  it is not a vLLM line and no rule could ever read it — along with annotated and
  tuple-assignment frame echoes (`compilation_times: list[…] = self.collective_rpc(`,
  `status, result = mq.dequeue(…)`), `[Autotuner]: Autotuning process starts`, and the
  `I want to run inference of a [specific model]` line from the issue template.
- 63 tests; the new test asserts every one of those lines becomes a fact **and** that the
  audit reports none of them as unread, so the two cannot drift apart again.

## 0.4.22 — 2026-11-09

- `servediff audit` now reads a crashed log. Every line of a real multi-process log carries
  a preamble (`(EngineCore pid=3811) ERROR 10-28 11:26:34 [core.py:1346] …`), and the noise
  patterns described the message, not the prefixed line — so a whole traceback came back as
  "shapes worth a look". The classifier now strips the preamble first (process, level,
  `MM-DD HH:MM:SS` timestamp, `- INFO -`, `module.py:line`), then matches the message, and
  also retries with a trailing `# comment` removed. On the four full real logs in this repo
  the count of shapes worth a look went 146 → 1 (vLLM 0.28.0 crash, UVA), 66 → 5 (0.28.0 with
  the V1 runner), 89 → 4 (0.30.0 killed after starting), 190 → 70 (the TP=8 0.23.1 report) —
  and what is left in the last one is configuration the parser genuinely does not read yet
  (`Using Eagle3 auxiliary layers from config`, `Model loading took … GiB and … seconds`).
- Traceback frames, `^^^` markers, `Process EngineCore:`, `--- Logging error ---` dumps,
  `[shutdown]` teardown lines, uvicorn's shutdown narration, `resource_tracker` and NCCL
  teardown warnings, `[Autotuner]: Autotuning process starts`, and text pasted from the issue
  template are all noise now. A real setting that carries the same preamble stays visible:
  the new test asserts three such lines are *not* noise, so widening a pattern has to break
  the suite.
- New fixture `tests/fixtures/vllm-crash-traceback.log`: a 43-line excerpt of a real crashed
  startup (engine and API-server tracebacks, a logging-error dump, shutdown chatter), with
  machine paths anonymised but line shapes untouched. `audit` on it reports `nothing worth a
  look`, while the `RuntimeError: UVA is not available` line stays consumed by the
  `startup.error` rule rather than appearing as unread.
- 62 tests.

## 0.4.21 — 2026-11-09

- Reads the last nine lines of the real vLLM 0.33.0 startup log that no rule matched. Four
  carry configuration and now have rules:
  `Reducing Torch threads from 16 to 1 for serving` → `cpu.torch_threads` (the pair, because
  `1` alone cannot say what it was cut from); `Dynamo bytecode transform time: 9.43 s` →
  `compile.dynamo_s`; `Compiling a graph for compile range (1, 2048) takes 13.12 s` →
  `compile.range` + `compile.graph_s`; `Initial profiling/warmup run took 1.02 s` →
  `compile.warmup_s`. `compile.total_s` is one number; an upgrade that doubles startup is
  usually one of these three moving, and the split is what says which.
- The other five are noise and are now classified as such instead of being offered back to
  the reader: `[backbone] Starting torch.compile` and `JIT kernel warmup starting.` (phase
  markers with no setting), `Using cache directory: …` and `saved AOT compiled function to …`
  (per-machine absolute paths that can never be equal across two boxes), and
  `Engine N: Avg prompt throughput …` (serving telemetry, not startup configuration).
- `servediff audit` on that log is now clean — 64 unread lines, every one of them noise,
  nothing worth a look — and a test holds the standard there: it fails, naming the shape, the
  day a version prints configuration this parser cannot read.
- README / write-up numbers recomputed from a fresh run, since the new facts change them:
  0.28.0 vs 0.30.0 is now **48 differences / 13 identical** (`--skip-absent`: 3 visible,
  45 hidden), and 0.28.0 with the V1 runner vs 0.30.0 is **27 / 33**. The fact table gained
  rows for the compile split and for CPU threads.
- 61 tests.

## 0.4.20 — 2026-11-08

- `servediff audit --max-shapes N` exits `1` when more than N shapes need a look, so the
  audit report can be a pipeline gate: run it against a log from your own upgrade and the
  build fails when the server starts printing configuration this parser cannot read yet.
  The count comes from the audit module, not from counting lines of the printed report
  (that report truncates shapes, so a text count would be wrong on a large log). `1` here
  means "the parser is behind this log", not "the inputs differ" — `diff`'s `1` keeps its
  own meaning, and `audit` without the flag still exits `0`.
- README: a `ci` badge (it existed and was not shown), and a short "why not just diff the
  two logs" — timestamps, pids and redrawn progress bars make a text diff of a startup log
  useless, and a value that moved is invisible inside a line that reworded itself.
- 60 tests.

## 0.4.19 — 2026-11-08

- Fixes the failed 0.4.18 release: the publish workflow runs the test suite from the
  unpacked sdist, and the new `tests/test_demo_svg.py` opens `tools/render_demo_svg.py` and
  `assets/demo.svg`, neither of which `MANIFEST.in` shipped. The sdist build itself was
  fine; only the tests-from-the-sdist step caught it, which is the point of that step.
  Both paths are now in `MANIFEST.in`, verified by building the sdist, unpacking it, and
  running `pytest` there before tagging.
- `CONTRIBUTING.md` rewritten to be usable by someone who has never seen the code: where a
  rule goes and what a handler returns, why one fact per independent thing, the
  trailing-period trap that has already broken this parser twice, asserting values *and*
  line numbers, and the sdist rule above. A visitor who can add a rule is worth more than
  a visitor who files an issue.
- Also recorded, because it was claimed and wrong: PyPI's project page ships a CSP of
  `img-src 'self' data:`, so the SVG cannot render there. It is a GitHub-surface asset
  only; the alt text and the text sections carry the PyPI page.
- 59 tests.

## 0.4.18 — 2026-11-08

- The README now opens on a picture instead of prose: `assets/demo.svg` is the real
  `servediff demo` output rendered as a terminal window — same flags, `VLLM_CUTLASS` →
  `MARLIN`, plus the one-sided KV cache row. Generated by `tools/render_demo_svg.py` from
  captured stdout (never hand-edited), so the image cannot drift from the command; a test
  regenerates it and fails if the committed copy loses the demo's rows.
  The renderer escapes the text (`<absent>` and `&` must go in escaped or the document
  breaks), wraps prose at 96 columns but never a table row, and lets a wrapped continuation
  inherit the line's colour.
- This release exists because the PyPI project page renders the README: the image and the
  `--skip-absent` section are what a first-time visitor now sees.
- 59 tests.

## 0.4.17 — 2026-11-08

- `servediff diff --skip-absent`: on the most common real pair — same flags, one run died —
  the crash truncates one log, so every later fact is a one-sided `<absent>` row. That was
  40 of the 43 rows on the 0.28.0-vs-0.30.0 pair and it buried the three that explain the
  outcome. The flag keeps only facts both logs printed: **43 → 3 rows**, with the failure
  warning kept and a line stating how many were hidden. Exit code stays 1 (the inputs do
  differ) and `--json` carries both `differences` and `compared_both_sides`, so a script
  cannot read a shorter table as a cleaner result.
- 57 tests (the new one also checks the hidden count is not silently folded into
  "identical").

## 0.4.16 — 2026-11-08

- The loading phase is now three facts instead of one: `weights.cache` (`miss (from
  scratch)` when the run had nothing to read from disk), `model_init.seconds` and
  `weights.load_s`. "Startup got slower" is not answerable until you can see whether the
  time went into constructing the module or reading shards. Audit on the real 170-line
  log: 71 unread / 12 worth a look → **68 unread / 9 worth a look**.
- `servediff audit --markdown`: wraps the report in a code fence sized around the longest
  run of backticks inside it (log messages quote flags with them, and a shorter fence ends
  the block early, silently cutting the lines that follow), so the third field of the
  issue form is one paste rather than a reformat.
- README: the English command table never listed `demo` or `audit` — the two commands a
  fresh install needs — and now does; a GitHub Actions snippet shows the `--markdown` diff
  going into a step summary with exit 1 failing the job. The extracts table gained rows
  for the split loading cost, the weight source/prefetch decision, GPU headroom including
  the CUDA-graph-profiling effective utilization, and the server surface facts.
- Both worked examples regenerated again, because three new facts move the counts:
  `43 difference(s); 13 fact(s) identical` (was 41/13) and `24 difference(s); 31 fact(s)
  identical` (was 22/31), plus the `writeups/wsl-uva-028-vs-030.md` table from the same run.
- 56 tests.

## 0.4.15 — 2026-11-08

- Eleven facts from lines the 0.4.14 audit still called worth a look, all of them things
  that change what a run costs or what it serves while every flag stays the same:
  - `gpu.utilization_effective` and `gpu.utilization_to_match_kv_cache`, from
    `CUDA graph memory profiling is enabled (default since v0.21.0)… equivalent to
    --gpu-memory-utilization=0.4141 without … increase … to 0.4859`. A version that
    changes only this default gives you less KV cache for the same flag, and the line
    says so in numbers.
  - `cuda_graph.profiling` — the mode itself, with the version it became the default.
  - `weights.filesystem`, `weights.checkpoint_gib`, `weights.available_ram_gib` and
    `weights.prefetch` — where the loader read from and whether it could prefetch
    (`Auto-prefetch is disabled because the filesystem (EXT4) is not a recognized
    network FS`). On a slow filesystem this is the whole startup-time difference.
  - `kernel.flashinfer_autotune` (`disabled` when autotune is skipped),
    `server.tasks`, `server.scale_out`, `serving.chat_template_content_format`.
- A trailing full stop bit the parser again: `increase --gpu-memory-utilization to
  ([\d.]+)` swallowed the sentence's period, so `float("0.4859.")` raised and
  `servediff show` died with `exit 1` on a real log. The rule matches the period
  explicitly now, and the test states the failure it prevents.
- Audit on the real 170-line log: 78 unread / 19 worth a look (0.4.14) → **71 unread /
  12 worth a look**, noise unchanged at 59.
- README's two diffs were regenerated with this parser, because eleven new facts move
  the counts: 0.28-vs-0.30 is now `41 difference(s); 13 fact(s) identical` (was 29/13),
  and the V1-runner pair `22 difference(s); 31 fact(s) identical` (was 16/25). The
  `writeups/wsl-uva-028-vs-030.md` table was regenerated from the same run.
- 54 tests.

## 0.4.14 — 2026-11-08

- `servediff audit` separates the gap into noise and shapes worth a look. On the real
  170-line vLLM 0.33.0 log it used to print all 80 unread lines, 59 of them route
  tables, banner art, deprecation warnings, progress bars and access logs — nobody
  pastes that into an issue. Now the noise is counted (grouped in full only under
  `--everything`) and the 19 remaining shapes are listed. Noise is matched against the
  raw line, not the digit-collapsed shape: collapsing turned `200` into `N`, so
  uvicorn access logs survived the first filter.
- Two facts the audit itself surfaced: `cuda_graph.capture_s` and
  `cuda_graph.capture_gib`, from `Graph capturing finished in 3 secs, took 0.25 GiB`.
  That line was previously unread even though a neighbouring line already gave
  `cuda_graph.pool_gib`; a capture pass that takes twice as long is startup-regression
  evidence on its own. Both passes are kept (`[3, 4]` / `[0.25, 0.18]` on the real log).
- A regression test now asserts that no line whose value `parse()` records is also
  reported as unread, so the audit report and the parser cannot drift apart again.
- 53 tests.

## 0.4.13 — 2026-11-08

- `servediff audit <log>` prints the lines no rule reads, grouped by shape, so a user
  whose fact came out `<absent>` can hand over the exact gap instead of describing it.
  The 0.4.9 release pointed at `tools/audit_unread_lines.py`, which only exists in a
  checkout — the capability was unusable for anyone on PyPI. The dev script remains as
  a wrapper over the same code.

## 0.4.12 — 2026-11-08

- `servediff demo` diffs two excerpt logs carried inside the package. Until now a fresh
  `pip install servediff` gave you a tool with nothing to run it on: every documented
  command needs two startup logs, which is exactly what someone who has not hit the
  problem yet does not have. The demo pair is public vllm#47047 text, and the command
  prints why its exit code is 1 rather than 0.
- The package now ships `servediff/examples/*.log` as package data.

## 0.4.11 — 2026-11-08

- The 0.4.9 note here claimed the unread set on the real log went "from 73 to 46
  shapes". It did not: the real movement was 87 lines / 80 shapes down to 80 lines /
  73 shapes, and 73 was the after number, not the before one. Corrected; the rest of
  that entry stands.
- `tools/audit_unread_lines.py` now applies the same ANSI normalisation the parser does
  before deciding a line is unread, so its report cannot count a line the parser reads.

## 0.4.10 — 2026-11-08

- `servediff.__version__` was a literal `"0.1.0"`, unchanged since the first release,
  while `servediff --version` read the installed distribution and printed `0.4.9`. Any
  script or test that checked `__version__` was reading a version nine releases old. It
  is now read from the installed package, and a test pins both paths to one number.

## 0.4.9 — 2026-11-08

- Eight more facts, chosen by auditing what the parser still ignores rather than by
  guessing: `gpu.free_gib` / `gpu.total_gib` / `gpu.utilization` / `gpu.budget_gib`
  (startup memory headroom), `jit.warmup_s` and `jit.monitor_mode`, the
  `compile.autotune_gemm` gate, `sampling.overridden` (the model's
  `generation_config.json` quietly replacing the sampler flags),
  `backend.candidates`, and SGLang's `server.command`.
- Kernel selection is read one fact per op (`kernel.ir_op.rms_norm=native`,
  `kernel.ir_op.gelu_and_mul_sparse=triton|native`) instead of one long
  `IrOpPriorityConfig(...)` string. The long form is wider than the table, so two
  different configs truncated to the same visible text while the row said "changed" —
  a claim the output did not support.
- `tools/audit_unread_lines.py` reports, per fixture, the lines no rule matches. That
  is how the list above was found, and it is the first place to look before adding a
  rule: a tool that shows "what the server actually ran with" is defined as much by
  what it ignores. On the real 0.33.0 log this set went from 87 lines / 80 distinct
  shapes to 80 lines / 73 shapes; the remainder is route tables, the ASCII banner,
  torch deprecation warnings, checkpoint-shard progress bars and `[backbone]`
  compile-timing detail, none of which should be diffed.
- README examples regenerated from the build: 0.28.0 vs 0.30.0 now reads
  `29 difference(s); 13 fact(s) identical`, and the V1-runner pair
  `16 difference(s); 25 fact(s) identical`.

## 0.4.8 — 2026-11-08

- The first command in the README was wrong on the page people actually run. After 0.4.7
  taught the parser to read the concurrency line, that 47047 pair prints
  `2 difference(s); 0 fact(s) identical` — a `kv_cache.max_concurrency` row appears
  because only one of the two excerpts quoted it — but the README still showed
  `1 difference(s)`. Corrected, and the new row is explained as a presence difference
  rather than a changed value.
- Packaging metadata: the keyword list now carries the terms a person would actually
  search for (startup-log, log-diff, kv-cache, cuda-graph, moe, tensor-parallel,
  inference-server), and the package declares `Topic :: System :: Logging`. Discovery on
  PyPI and GitHub is a packaging problem before it is a content problem.

## 0.4.7 — 2026-11-08

- Seven more facts are read from a vLLM startup log, all of them found missing while
  diffing the real 0.33.0 log from the previous entry: `model.architecture`
  (`Resolved architecture`), `model.max_len` (`Using max model len`),
  `kv_cache.memory_gib`, `kv_cache.tokens`, `kv_cache.max_concurrency`,
  `engine.init_s` and `server.bind`. These are the numbers that move when a flag change
  costs you concurrency; before this, a diff could show the args changing and the
  capacity silently not at all.
- `model.max_len` is the value the engine settled on, not the requested one, which is
  why the same `--max-model-len` can still differ between runs.
- README examples regenerated from the build: 0.28.0 vs 0.30.0 now reads
  `20 difference(s); 11 fact(s) identical`, and the V1-runner pair
  `14 difference(s); 16 fact(s) identical`. The vllm#47047 fixture diff now also
  reports a concurrency line that only one side quoted — a presence difference, not a
  backend change.

## 0.4.6 — 2026-11-08

- `capture` verified against a live server, not a fake process: vLLM 0.33.0 on one RTX 4060
  Laptop (8 GB, WSL2), Qwen2.5-0.5B. The server reached `/health` and served a completion,
  the log was written while it ran, the env snapshot landed before launch, and Ctrl-C exited
  `130` with the log intact. A failed run still passed the server's exit code (`1`) through.
- New fixture `vllm-0.33.0-real-rtx4060.log` — a full 170-line log from that run rather than
  an issue excerpt — pinned by `test_real_server_log_from_033_is_parsed`, which guards
  `engine.version`, `model_runner V2`, the attention backend, the KV-cache layout, UVA
  fallback, world size, the CUDA-graph capture list and the model-loading size.
- `Limits` states this run as evidence: the two env snapshots of the same command diff down
  to the two variables that separated a crash from a start.

## 0.4.5 — 2026-11-07

- `series` text output keeps both ends of a long value instead of cutting it at a fixed
  column width. Two backends that differed only in their last characters
  (`…NoDPEPMonolithic` vs `…NoDPEPModular`) rendered as identical text in two columns while
  the row was labelled "changed" — a report that hides the difference it reports. Markdown
  output was never truncated and is unchanged.
- `Limits` now lists what the fixtures actually contain (including the TP4+EP H200 excerpt
  from vllm#45604) instead of claiming a single-GPU SGLang log that is not in the repo;
  `tests/fixtures/README.md` documents all seven fixtures rather than five.

## 0.4.4 — 2026-11-07

- `capture` survives Ctrl-C. That is how a server run actually ends: interrupt it once the
  startup lines are printed. It used to leave a traceback and exit 1, which this tool
  reserves for "the inputs differ"; it now stops the child, keeps the log, says where it
  is, and exits 130. The server's own exit code is still passed through on a normal exit.
- README documents `capture` as the way to produce both logs, since most people arrive
  with one log and a question rather than two ready files.

## 0.4.3 — 2026-11-07

- `capture` now snapshots the interpreter the command itself names. `capture -- python -m
  vllm...` used to snapshot the interpreter servediff runs on, which is a different venv
  often enough that a later snapshot diff compared two copies of servediff and reported
  no difference — the failure mode this tool exists to name. When the command does not
  name a python launcher, the snapshot line now says which interpreter it describes and
  points at `--python`.
- The snapshot line prints the interpreter path it used, so the JSON and the log agree.

## 0.4.2 — 2026-11-07

- An unreadable path (`servediff show typo.log`, `diff` with a missing side) no longer
  raises a traceback and exits 1. Exit 1 means "the two sides differ", so a mistyped
  path used to read as a detected difference; it now prints one line and exits 2.
- `diff` on two `servediff env` snapshots marks truncated cells with `...`, like the
  log diff already did.
- README gains a commands-and-exit-codes table, including which env vars `env` records
  and the fact that `1` never means "something went wrong".

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
