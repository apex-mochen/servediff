# vLLM: "No available shared memory broadcast block found in 60 seconds" — what the line means, and how to tell which failure you have

This string shows up in dozens of threads, and the two answers that dominate search results are
"increase `/dev/shm`" and "it is still compiling". Both are sometimes right. Both are wrong often
enough that people keep re-posting the same log.

Here is the same question from the other end: instead of guessing the cause, read what the runs
actually resolved and let the log separate one waiting line from sixteen stalls. Every command
output below was produced by `servediff 0.4.61` — except the `repeats` demo near the top, which is
labelled as synthetic.

## Three readings of this line, and how to tell them apart

The line is emitted from the writer side of `shm_broadcast.py`, once every
`VLLM_RINGBUFFER_WARNING_INTERVAL` (60 s) while it waits. That much is fixed. What people disagree
about is *why* the writer is waiting, and the published answers fall into three groups:

| reading | who publishes it | what it predicts in the log |
|---|---|---|
| `/dev/shm` is too small | most guides, e.g. [kubesimplify](https://blog.kubesimplify.com/running-a-big-llm-across-multiple-gpus-with-vllm) ("you probably forgot `--ipc=host`") | a *different* error, raised at buffer creation, before any 60 s wait exists |
| workers are still compiling / capturing graphs | same guide ("shows up while vLLM is busy compiling or capturing CUDA graphs") | the lines stop once `Graph capturing finished` appears |
| one reader stopped draining, writer pinched | [LMCache's MI300X notes](https://blog.lmcache.ai/en/benchmarking-lmcache-for-multi-turn-agentic-workloads-on-amd-mi300x/) ("Both TP workers alive, no preemptions"), [GB10 UVM livelock notes](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/main/docs/uvm-livelock-gb10.md) | the wait outlives capture, and the cadence stays a metronome |

Three cheap checks decide between them, and all three are arithmetic on the log:

**Check 1 — does `/dev/shm` capacity explain it?** It cannot, if the capacity check passed. That
check runs at buffer creation and fails loudly, with the fix already in its message:

```
RuntimeError: Insufficient space in /dev/shm: N MiB required, M MiB free.
Increase /dev/shm (e.g. --shm-size or --ipc=host).
```

You cannot reach the 60-second watchdog with a full `/dev/shm`; you get this instead. So a run
that logs the 60 s line had enough shm **at creation**. (`--ipc=host` remains the right answer to
a *different* error.)

**Check 2 — did the wait outlive graph capture?** Count the lines after
`Graph capturing finished`. In the log analysed below, `Graph capturing finished` is at
`11:37:55` and **14 of the 16 lines come after it**, through `11:50:31`. "Still compiling" cannot
explain a tail that starts 13 minutes after compiling ended.

**Check 3 — is the cadence a metronome?** If the gaps hold at exactly one interval, it is one
continuous wait re-logging itself, not sixteen independent stalls. Sixteen real stalls are
irregular; this is not.

Two of the three checks are pure counting, which is why `servediff` ships them as a subcommand:

```
$ servediff repeats repro.log --top 3
/tmp/repro.log: 56 line(s), 2 shape(s) repeated 2+ time(s), 55 line(s) in them
    39x burst    11:33:26 -> 11:33:26.380  span 0.38 s
        gaps 0.010 s median, 0.010 s max
        [rank0]:[W113 11:33:26.000 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 20971520 bytes
    16x regular  11:34:32 -> 11:49:32  span 900 s
        gap 60.000 s median, 60.000 s max
        [W113 11:34:32 shm_broadcast.py:801] No available shared memory broadcast block found in 60 seconds.

  A regular gap is one wait being re-logged, not N separate failures: compare that span with
  anything that could have blocked a reader.
```

### The timestamp is the end of the wait, not its start

Every one of these lines is printed *after* the watchdog has already been waiting for the
interval it names. So `09:03:37` means the wait began at `09:02:37`, and comparing that with
anything else in the log is the check that actually separates the readings. `servediff repeats`
does the subtraction — it reads the period out of the message itself ("found in 60 seconds"):

```
6x regular  09:03:37 -> 09:08:37  span 300 s
   gap 60.000 s median, 60.000 s max
   No available shared memory broadcast block found in 60 seconds …
   this line is the end of a 60 s wait: the first one began at 09:02:37
     0 other line(s) inside that span: nothing moved during the wait
```

Two numbers, one diagnosis each:

| log | wait began at | what printed at that second | reading |
|---|---|---|---|
| A2, TP2 (`vllm-forum-2026-07`) | `09:02:37` | nothing — graph capture was still running (it ended `09:02:40`) | the wait outlived capture by 5 min with nothing inside: a reader stopped draining |
| RTX 6000 Ada, TP2 (`vllm#45198`) | `22:01:40` | `Triton kernel JIT compilation during inference: _zero_kv_blocks_kernel` | the wait began the same second a kernel started compiling: this one *is* compilation |

`vllm#45198` is the case the guides are right about — and it still is not a `/dev/shm` problem.
The reporter `--ipc=host` with `/dev/shm` bind-mounted, and his own workaround was
`--disable-custom-all-reduce`, not a bigger shm. What made it look like a shm hang was that the
only lines after the warning are the waits, 60 s apart, with nothing in between.

`0 other line(s) inside that span` is not decoration: it is the difference between "compilation
was still working through the window" (something would be printing) and "nothing at all was
working" (a stuck reader). Both cases in the table read zero — the thing that separates them is
*what preceded the wait*, which is the line `repeats` now puts under the wait.

(This demo runs on a 56-line synthetic log carrying the two shapes; on the real log below the same
two shapes read 2.318 s and 959 s.)

## A second worked example: TP2 on two A2 cards, six waits after capture

This is the smallest complete instance in the corpus — 24 lines, from
[forum topic 2828](https://discuss.vllm.ai/t/2828), a poster who said "unsure how to debug":

```
09:02:36  Available KV cache memory: 4.91 GiB
09:02:36  GPU KV cache size: 407,080 tokens
09:02:40  Graph capturing finished in 4 secs, took 0.11 GiB
09:03:37  No available shared memory broadcast block found in 60 seconds.  (×6, to 09:08:37)
```

Run the three checks on it:

1. **`/dev/shm` capacity** — no `Insufficient space in /dev/shm` line, so the buffer was created
   fine. This is not a shm-size problem.
2. **Did the wait outlive capture?** The first wait line is 57 s after `Graph capturing finished`,
   i.e. the wait began at `09:02:37` — **3 seconds before capture ended**. Capture overlapped the
   start of the wait, and then capture finished and the waiting continued for 5 more minutes.
   "Still compiling" predicts the opposite: the lines stop when capture ends. They do not.
3. **Cadence** — `servediff repeats` on the file:

```
6x regular  09:03:37 -> 09:08:37  span 300 s
   gap 60.000 s median, 60.000 s max
   No available shared memory broadcast block found in 60 seconds …
```

Exactly one interval, six times: **one** wait being re-logged, not six failures.

What is left is a reader that stopped draining. The run's own numbers bound it: 8.29 GiB of
weights per rank on 16 GB cards, 4.91 GiB of KV left, capture 0.11 GiB, `torch.compile` 43.57 s
total. There is no allocator OOM in this log, so the pool is not the suspect — the consumer is.
The next command is the one the thread lacked: `py-spy dump` on each worker pid while the wait
line is being printed, to see which rank is sitting in a broadcast read.

## The rest: what to compare between two runs

The question behind most of these posts is "what changed", and a benchmark number cannot answer it.
The startup log records what the server *resolved* on the way there — which attention and all-reduce
backends it picked, whether CUDA graphs were captured, which fusions ran, what the KV-cache pool
ended up being, and whether it finished with `ready` or a traceback. Two logs from two runs
therefore contain the answer, in plain text, without a profiler.

```bash
pip install servediff
servediff diff a.log b.log
```

## One flag: did it take effect?

Same vLLM version, same NVFP4 checkpoint, second launch taking the Marlin kernel path instead of
CUTLASS. The diff is two rows:

```
$ servediff diff vllm-47047-v0.23-default.log vllm-47047-v0.23-marlin.log   # excerpt
fact                             baseline               candidate               status
backend.moe                      VLLM_CUTLASS (NvFp4)   MARLIN (NvFp4)        changed
kv_cache.max_concurrency         <absent>               54.14                 only_candidate

2 difference(s); 0 fact(s) identical. Absent means the log did not print it, not that it was off.
```

Both of these logs are single-rank, so the live command prints two `note:` lines before the table
(it cannot tell whether an absent all-reduce path means "single rank" or "not used"); the fact rows
above are verbatim. The `2 difference(s)` counter and exit code `1` are what the run returns.

That is what "the flag did take effect" looks like. The expensive case is the mirror image: same
command line, `backend.moe` **identical** in both logs. A flag that silently did nothing produces
a diff of zero server-side facts, and no amount of benchmarking will distinguish "the flag is
neutral" from "the flag never reached the engine". The log diff distinguishes them in one line.

## Version bump, same flags

v0.23 → v0.26 on the same configuration. Nothing on the command line changed:

```
$ servediff diff vllm-53472-v0.23.log vllm-53472-v0.26.log   # excerpt
fact                             baseline                              candidate                              status
backend.candidates               ['FlashInfer TRTLLM', 'FlashInfer CUT... <absent>                               only_baseline
backend.moe                      FlashInfer CUTLASS (Unquantized)        TrtLlmBf16ExpertsMonolithic            changed
backend.moe_prepare_finalize     <absent>                                MoEPrepareAndFinalizeNoDPEPMonolithic  only_candidate

3 difference(s); 0 fact(s) identical.
```

The MoE backend moved from a CUTLASS path to a TRT-LLM monolithic expert kernel, and a
prepare/finalize path appeared that the older release never printed. Those are the facts a
throughput comparison is actually comparing.

## `absent` is a claim with a meaning

The distinction that matters most in practice:

- `<absent>` means **this run never printed the line**. It is not the same as "off", and not the
  same as "false".
- `--skip-absent` then reports only facts both sides printed, and counts what it left out:

```
$ servediff diff a028.log a030.log --skip-absent
3 difference(s); 14 fact(s) identical. 50 difference(s) skipped because one side never printed those lines (--skip-absent). Absent means the log did not print it, not that it was off.
```

The unfiltered run of the same pair reports:

```
53 difference(s); 14 fact(s) identical. Absent means the log did not print it, not that it was off.
```

Fifty of those 53 are one-sided, so a naive diff of 53 rows would overstate the change by a
factor of about 18. The 3 that survive are the ones both engines actually resolved, and only those
can be compared as values. The full 53-row table for this pair, with the command, exit code and the
reason the baseline died early, is kept in the repository so it has a permanent home:
[`writeups/wsl-uva-028-vs-030.md`](https://github.com/apex-mochen/servediff/blob/main/writeups/wsl-uva-028-vs-030.md).

## One log can contain two different failures

From [vLLM #54559](https://github.com/vllm-project/vllm/issues/54559) — 102 lines, TP2, single
GPU, 79.14 GiB:

```
$ servediff show 54559.log        # excerpt
gpu.total_gib           79.14          line 87
gpu.free_gib            78.46          line 87
gpu.consumed_gib        [64.46, 64.39] line 87
gpu.budget_gib          75.18          line 87
model_loading.gib       63.31          line 40
cuda_graph.capture_gib  3.26           line 87
cuda_graph.capture_s    85             line 86
torch.oom_requested_mib [20.0, 14.0]   line 1
torch.oom_free_mib      [9.19, 10.75]  line 1
```

Two things fall out of those numbers, neither of which the issue text says:

**39 OOM lines are two allocator bursts, not 39 failures.** They run from `11:33:26.640` to
`11:33:28.958` — 2.318 s — alternating `[rank0]`/`[rank1]`, 7–16 ms apart with one ~1 s gap, the
request dropping from 20 MiB to 14 MiB. And the card was not full: `gpu.free_gib 78.46` while the
allocator saw 9.19 MiB free, because the pool is set to `0.95 × 79.14 = 75.18 GiB`. The failure
is inside the allocator's own budget, not on the device.

**16 broadcast-wait lines are one 959 s wait, not sixteen stalls.** All from `EngineCore
pid=260`, `11:34:32` → `11:50:31`, one line per 60 s interval, a single +118 s step, and 14 of the
16 arrive *after* `Graph capturing finished` at `11:37:55`. Checks 2 and 3 from the top of the
post are answered by that sentence alone: the wait outlived capture, and the cadence held.

## The diff bounds the problem; it does not name the cause

From [vLLM #63705](https://github.com/vllm-project/vllm/issues/63705) — MiniMax-M2.7, TP4+EP,
4×H100. The reporter's own table puts the regression entirely in decode: ttft −0.9 / −1.8 / −0.8 %
on 0.29.0 / 0.31.0 / 0.32.0, tpot +22.7 / +27.9 / +22.1 %, throughput −17.9 / −21.1 / −17.5 %. A
benchmark table like that cannot say *which* server-side component moved. The startup logs can:

| fact | v0.26.0 | v0.29.0 |
|---|---|---|
| TP group all-reduce chain | `['PYNCCL']` | `['FLASHINFER', 'PYNCCL']` |
| EP group all-reduce chain | `['PYNCCL']` | `['PYNCCL']` |
| model runner | V1 (line absent) | V2 (`Using V2 Model Runner`) |

FlashInfer all-reduce entered the **TP** dispatch chain and EP stayed on PYNCCL, so the candidate
change is TP-side; Model Runner V2 became active on this configuration. Two named variables to
A/B, from lines that were already being written to disk.

And the honest part: an external contributor later re-ran that benchmark on 4×H100 with the same
serve flags and **did not reproduce the regression** — v0.29 came out about 8 % faster. So the
diff did not solve the issue; it told both sides what to test. A log says what is configured, not
what it costs. Reading the dispatch list as "every collective used FlashInfer" is specifically
wrong: it is a candidate set tried in order, falling through when a backend rejects the input.

## If the log says `CUDA out of memory` instead

That message means something else and has its own two wordings — the device's free memory vs the
allocator pool's — with the arithmetic and the command in
[`CUDA out of memory` on a card that looks free](cuda-oom-holder.html).

## What servediff is, and what it refuses to do

It reads startup logs from vLLM and SGLang and answers "what did each server actually run with":

```bash
pip install servediff
servediff diff a.log b.log              # facts, side by side, each with its source line
servediff diff a.log b.log --markdown   # same as a table
servediff show a.log                    # one log's facts with line numbers
servediff repeats a.log                 # one re-logged wait vs N real failures
servediff capture -o run.log -- <command>   # run a server and keep its startup log
servediff demo                          # two bundled logs, if you have none of your own yet
```

It does not profile, does not judge which change caused the slowdown, and does not guess when a
line is missing. Its job is to make the 53-row diff collapse into the 3 rows that can actually be
compared, and to keep `absent` from being read as `off`.

Source, changelog and the two full case analyses:
<https://github.com/apex-mochen/servediff>

---

*Prepared 2026-11-17; every command re-run 2026-11-22 in a fresh virtualenv with
`pip install servediff==0.4.61` (`servediff --version` → `servediff 0.4.61`). The two diff blocks,
the `--skip-absent` pair, `show`, `demo` and the link set were reproduced on that install; two
exceptions are labelled where they appear — the `repeats` demo is a synthetic 56-line sample (its
counters and layout are the tool's real output), and the 0.28/0.30 53-row pair is documented in
`writeups/wsl-uva-028-vs-030.md` instead of being re-run here. The numbers in the #54559 section
come from `servediff show` on the reporter's log and match `ISSUE-54559-analysis.md`. Both analyses
live in the repository, and both threads remain open.*
