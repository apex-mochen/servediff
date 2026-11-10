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

## The log from a stack nobody had seen

Everything above was measured on one machine's WSL2 runs — NVIDIA, CUDA, one GPU. So the
first honest test of the claim came from a log nobody here produced: the `collect_env` block
pasted into [vllm#65034](https://github.com/vllm-project/vllm/issues/65034), a ROCm box
(AMD Radeon AI PRO R9700, torch 2.13.0+rocm10.1.0, vLLM 0.29.1.dev, Python 3.14). 176 lines.

`audit` said **175 lines worth a look, 112 shapes** — essentially the whole block. Not a
single rule in the tool matched a single line of it. That is the failure mode the section
above says to expect, and it is also the strongest argument for the audit existing: a tool
whose noise table only describes its author's stack is one stack wide.

The fixes were both mechanical:

- **What the block states as fact, read as fact.** `engine.version` + `engine.git_sha`
  (`vLLM Version : 0.29.1.dev0+g98dff2a81.d20261002 (git sha: 98dff2a81, date: 20261002)`),
  `env.torch`, `env.torch_rocm`, `env.rocm`, `env.hip`, `env.os`, `env.python`,
  `gpu.model`, `env.pip.{torch,triton,transformers,numpy}`, and
  `env.var.{TORCHINDUCTOR_CACHE_DIR,PYTORCH_ROCM_ARCH,MAX_JOBS,LD_LIBRARY_PATH,…}`. These
  are the facts that decide which kernels a run can even choose: a torch built against a
  different ROCm is a different server, whatever the flags say.
- **What the block states as machine inventory, counted as noise.** The section rules
  (`====`), the CPU model and mitigation list, the wrapped CPU flag dump (including lines
  that start at column 0 with `3dnowprefetch` and carry digits), one
  `[pip3] amd-torch-device-gfx*` wheel per GPU architecture (~40 identical lines), the
  wrapped tail of `LD_LIBRARY_PATH`, `CMake version :`.

Result: **158 unread lines, 95 shapes, nothing worth a look.** Two regressions the same run
caught, both now pinned by tests:

- the generic `version X.Y.Z` banner rule took `CMake version : version 3.31.10` as the
  engine version, so `engine.version` held two values from two different tools;
- a launch command pasted into a report (`vllm serve … \` + `--flag value \`) is the user's
  shell history, not output — the server prints its args itself as "non-default args".

## Eleven more logs, and a second engine

One unfamiliar log proves the failure mode; it does not show its size. So the next pass was
twelve more report pastes, picked from issue searches rather than from anything this machine
ran: six vLLM and six SGLang.

| log | what it is | worth a look, before | after |
|---|---|---|---|
| vllm#65611 | eight MI355X cards, vLLM 0.33.0rc1 | 61 | 0 |
| vllm#65250 | gVisor, vLLM 0.30.0 | 51 | 0 |
| vllm#63505 | production image, vLLM 0.32.0, CUDA 13.0 | 45 | 0 |
| vllm#63536 | RTX PRO 5000, prose pasted between the logs | 44 | 0 |
| sglang#41654 | SGLang 0.5.20 environment dump | 39 | 0 |
| sglang#42684 | NIXL backend, scheduler dies at startup | 12 | 0 |
| sglang#43741 / #49206 / #48502 / #46756 | four shorter SGLang reports | 11 / 3 / 2 / 0 | 0 |
| vllm#66192 | GB200 NVL node, aarch64, SM100 | 9 | 0 |
| vllm#66193 | RTX PRO 6000 Blackwell, aggregate line repeats the card prefix | 13 | 0 |

Three classes of miss, none of which is a single bad regex:

- **An inventory read as a setting.** A CUDA container prints ~35 `[pip3] nvidia-*==` lines;
  only a few wheels decide which kernels load. The rule now reads `flashinfer-python` too and
  the rest of the inventory is noise. The same problem one level up: the block ends with the
  environment, and the tool had whitelisted six variables out of it, so `CUDA_VERSION`,
  `TORCH_CUDA_ARCH_LIST`, `VLLM_BUILD_COMMIT`, `VLLM_IMAGE_TAG` all read as "unread". It is a
  general `NAME=value` rule now — minus `CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES` and
  `VLLM_CACHE_ROOT`, which differ between two boxes started with identical flags and would
  otherwise read as differences.
- **An absence read as a value.** `ROCM Version : Could not collect` matched a rule that
  captured one token, so a CUDA box reported `env.rocm = "Could"`. Worse than noise: two
  CUDA boxes agreed on a value neither had, and `diff` could never show it. `Could not
  collect`, `N/A`, `Not Set` and `unknown` now produce no fact at all.
- **Another engine's prefix.** SGLang prints `[2026-10-06 00:15:19] NIXL KVManager
  initialized with backend: UCX`, and its own `name: value` dump. The brackets were not
  stripped, so every message behind them was content, and the dump was package inventory.
  Both now go through the same normalisation `parse` uses — the audit had been re-implementing
  the prefix handling on its own, which is how the two drifted apart.

The machine-inventory half is the same idea as the ROCm block: an eight-card `rocm-smi`
weight / hops / link-type matrix, the GPU topology grid with its `Legend:`, `GPU N:` headers,
Hypervisor lines and the InfiniBand `NIC0: mlx5_0` grid describe the box, not the run.

- **A dependency dump read as settings.** A GB10 box reporting a native-MTP failure
  (vllm#65873) printed every `libcudnn*.so` by absolute path — `ldd` output. Nine lines,
  nine "settings". A bare shared-library path is now noise; an `ImportError: libcudart.so.13:
  cannot open shared object file` is the opposite case and stays a fact.
- **An empty value that looks like a fact.** A multi-node image prints
  `GPU models and configuration :` and then one `GPU 0: NVIDIA GB200` per card, so the
  aggregate yielded `gpu.model = ""` — a fact that matched another log's empty string instead
  of being absent. Empty is now an absence, and when the aggregate is silent the per-card
  line supplies the model (with its own `GPU 0:` prefix removed, since some images paste the
  whole card line after the colon).

Two useful side effects on the way:

- `diff vllm-65250.log vllm-63505.log` — a gVisor run against a production image — reports
  **23 differences / 5 identical**, and the differences are the ones that decide what the
  server can do: `TORCH_CUDA_ARCH_LIST`, `VLLM_BUILD_COMMIT`, `VLLM_IMAGE_TAG`, `CUDA_HOME`,
  `NVIDIA_REQUIRE_CUDA`, plus `OMP_NUM_THREADS` / `MKL_NUM_THREADS` /
  `VLLM_USE_FLASHINFER_SAMPLER` that only one side printed.
- `diff sg-41654.log sg-42684.log` reports **16 differences / 0 identical**, including
  `startup: failed (scheduler)` and `TypeError: NixlKVManager.set_kv_buffer_tensors() got an
  unexpected keyword argument 'slot_layer_ids'` — a startup failure that no `server_args`
  records.

## What zero does and does not mean

Zero here is a claim about **what these twenty-one logs print**, not about log space. The four
full logs above, two fixtures (a crashed startup and a clean 170-line run) and the fourteen
issue pastes all report `nothing worth a look`; a log from a stack none of them covers may
well print shapes no rule knows — and one did: the ROCm block started at 175 and needed two
rounds of work before it reached zero, and the pastes after it added four more classes of
miss before they did. That is precisely why the audit prints the second group at all, and why
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

## A reproducer script is not a log

The paste that finally showed where the report was wrong was not a log at all.
sglang#49123 pastes a 195-line benchmark harness — `def run_marlin(...)`,
`handles[f] = safe_open(...)`, a module docstring — and the audit called 38 of its lines
"worth a look", every one of them the script itself. A reader who opens that report learns
the tool cannot read a script: true, and useless.

The rule that fixes it has to know that a log line *states* something — `name: value`, or a
sentence. It never starts with `def`, never opens with a bare `"""`, never ends in a trailing
comma. Seven conservative patterns on those shapes, plus one stateful pass over the prose
between two `"""`, took that paste from 38 lines to 1; the one left is a lambda's
continuation, which genuinely is a line no rule can read.

The counter-example had to stay: a shell-style `MAX_MODEL_LEN=4096` inside the same paste is
still read as `env.var.MAX_MODEL_LEN`. A script's code is noise; a script's *call* is not.

One bug surfaced while measuring it. The keyword pattern was written
`^\s*(?:if |elif |else |…)\b` — the space inside the alternative and `\b` after it, so
`if "W4A4" in name:` failed the boundary test (a space followed by a quote is not a word
boundary) and stayed in the report, while `if x:` did not. A pattern that is almost right is
the quietest kind of wrong: it looks like it covers the class and covers all but one member.

## A held-out corpus found a rule that only matched the reformatted copy

Everything above was measured on the fixtures this repo already carried, which is a closed
loop: they were hand-copied from issue bodies, and a copy can carry the copyist's formatting.
So eleven pastes were pulled fresh from vLLM and SGLang issues created after 2026-10-20, and
audited blind.

Eight of eleven came back with lines worth a look. Two of those were script pastes covered
above; the rest were real. The biggest one was not noise at all:

```
PyTorch version: 2.10.0+cpu          <- worth a look, and it is a fact
OS: Debian GNU/Linux 12 (bookworm) (x86_64)
Python version: 3.12.15 (...) [GCC 12.2.0] (64-bit runtime)
HIP runtime version: N/A
```

Every `collect_env` rule was written `^PyTorch version\s+:\s*(\S+)` — requiring whitespace
before the colon. ROCm's own dump aligns its colons with spaces, so the rule worked on that,
and vLLM's real `collect_env` prints `PyTorch version: 2.10.0+cpu` with none, so the rule was
dead on the output every bug report actually pastes. The audit reporting "worth a look" was
correct: the tool could not read the torch version on a CPU-only box.

That is the failure mode this whole writeup is about, one level up: a rule pinned to the
one copy of the wording its author happened to see. `\s*` on eight rules fixed it, and the
same eleven-log corpus dropped from eight files worth a look to two, both script pastes,
four lines.

The rest were inventory shapes the table had never seen: SGLang's macOS report (`MPS
available`, `Apple Silicon`, `Metal Support`, `Unified Memory`, `CPU Cores (…)`), a
multi-GPU row written `GPU 0,1,2,3: NVIDIA H20`, the container runtime's `brand=…driver>=…`
capability list, a topology header row, a reporter's parenthetical note, and a plugin source
paste (`WTYPE = …`, `tensor.offset`, `(Path(…) / "config.json").write_text(`). None of those
can be a log line, so each got one narrow pattern.

The last four lines were the same script in a harder form: an argument list standing alone
(`8, 96, 128, conv_kernel_size=4, num_spec=5`, `groups, KVCacheLayout.BLHNC`) and a
continuation that closes parentheses without starting a statement
(`…routing_input_mode=…), wp))(ml)`). A log line is a statement — it starts with a word and
says something about it — so it never consists of arguments alone. Those two patterns require
exactly that and nothing else: the first token must be a bare name, which is what keeps
`NVIDIA_DRIVER_CAPABILITIES=compute,utility` readable as a fact. Eleven pastes, then: **none
worth a look.**

Both corpora ship in the repo, so this section is rerunnable instead of believable:
`servediff audit tests/fixtures` (twenty-one logs) and
`servediff audit tests/fixtures/held-out-2026-10` (the eleven pastes, with
`PROVENANCE.md` naming each issue). Check out `v0.4.32` and run the second command to watch
it report eight of eleven.

## A second month, to see whether the rules learned logs or learned October

Eleven pastes from two weeks of one month is one point. The obvious doubt is that the noise
table was bent until that particular set passed — each rule here was written in front of a
paste that broke it, which is a good way to write a rule and an excellent way to overfit.

So the same collection was run again against issues created after 2026-11-01, a month later.
Twelve blocks qualified; five were already in the October batch, so seven were new.

Six of the seven were clean on the first try, which is the useful half of the result: the
table did generalize past the month it was written in. The seventh had one line:

```
    run(a) if a.cmd == "run" else compare(a)
```

A conditional expression used as a statement, lifted out of a reproducer script pasted as the
issue body. It picks between two values, so unlike a log line it has no subject to say
anything about — the same principle as the argument lists above, one form further out.

The first pattern written for it was the obvious shape — contains ` if `, contains ` else `,
ends in `)` — and it had to be thrown away, because it also swallowed a warning that merely
uses both words in a sentence:

```
Using FlashInfer all-reduce if available else falling back to the custom kernel
(set VLLM_ALLREDUCE_USE_FLASHINFER=0 to disable)
```

That line is a statement about a decision, and it is exactly what this tool exists to find.
The adopted pattern requires the line to *begin* with a call
(`^[A-Za-z_][\w.]*\([^()]*\)\s+if\s+…\s+else\s+…\)$`), which the sentence does not. It
matches one line across all four corpora — the one it was written for — and the guard test
pins the difference between the two.

Seven pastes, then: **none worth a look.** At 0.4.36 that directory reports one of seven.

## Limits of the claim

These four logs are from one machine's WSL2 runs plus one posted TP=8 log, plus eleven pastes
taken fresh from vLLM and SGLang issues in October 2026 and seven more in November.
`nvcc`, other engines, and other parallelism layouts are unmeasured, and a log from a
different stack may well print shapes none of these rules know. That is what the audit is
for: run it on your own log, and paste the report into
[the issue form](https://github.com/apex-mochen/servediff/issues/new?template=missed-log-line.yml)
— the second group is the useful half.
