# Analysis: vLLM #54559 TP2 startup — two failure families in one log

Source: https://github.com/vllm-project/vllm/issues/54559
Analyzed: 2026-11-13. One log, 102 lines / 20,533 bytes, vLLM 0.28 (`(EngineCore pid=260)`,
`Worker_TP0_EP0 pid=356`, `Worker_TP1_EP1 pid=408`), single GPU, 79.14 GiB total.
Numbers below are the output of `servediff show` on that log, not estimates.

```
$ servediff show 54559.log
gpu.total_gib          79.14          line 87
gpu.free_gib           78.46          line 87
gpu.consumed_gib       [64.46, 64.39] line 87
gpu.budget_gib         75.18          line 87
gpu.utilization        0.95           line 87
model_loading.gib      63.31          line 40
cuda_graph.capture_gib 3.26           line 87
cuda_graph.capture_s   85             line 86
torch.oom_device       [0, 1]         line 1
torch.oom_requested_mib[20.0, 14.0]   line 1
torch.oom_free_mib     [9.19, 10.75]  line 1
```

## Family 1 — 39 allocator OOM lines in 2.3 seconds

All 39 lines are `CUDACachingAllocator.cpp:3933 memory allocation failed with OOM`, alternating
between `[rank0]` (device 0) and `[rank1]` (device 1), from `11:33:26.640` to `11:33:28.958` —
**2.318 s total**.

| phase | interval | request |
|---|---|---|
| burst A (lines 1–5) | 16, 4, 15, 0 ms | 20 MiB |
| gap | ~1.018 s | — |
| burst B (lines 6–39) | 7–16 ms | 20 MiB → 14 MiB |

Two things follow from the numbers alone:

1. **It is not "the GPU is full".** The device summary line right below reports
   `gpu.free_gib 78.46` and `gpu.consumed_gib 64.46`. What failed was the *allocator's own* pool:
   `torch.oom_free_mib` says 9.19 MiB / 10.75 MiB free when 20 MiB was requested. vLLM sets that
   pool from `gpu_memory_utilization × total` = `0.95 × 79.14 = 75.18 GiB`, so the allocator ran
   out of *its* budget while the card itself still had room. Anything phrased as "OOM = card full"
   is wrong here.
2. **It is a retry loop, not 39 decisions.** 10–16 ms between attempts with one ~1 s gap is one
   allocator falling back (20 MiB, then 14 MiB), logged every attempt. Counting the lines as 39
   incidents overstates it by a factor of ~39; counting it as one event understates the
   repeating pattern. The honest count is **two bursts**.

## Family 2 — 16 broadcast-wait lines = one 959 s wait

All 16 lines are `shm_broadcast.py:801 No available shared memory broadcast block found in 60
seconds`, all from `(EngineCore pid=260)`:

```
1  11:34:32         6  11:40:31  +60    11  11:45:31  +60
2  11:35:32  +60    7  11:41:31  +60    12  11:46:31  +60
3  11:37:30  +118   8  11:42:31  +60    13  11:47:31  +60
4  11:38:30  +60    9  11:43:31  +60    14  11:48:31  +60
5  11:39:30  +60   10  11:44:31  +60    15  11:49:31  +60
                                        16  11:50:31  +60
```

959 s first to last, one line per 60 s interval. In `shm_broadcast.py` this text is emitted from
the `elapsed > VLLM_RINGBUFFER_WARNING_INTERVAL * n_warning` branch (`n_warning += 1`), so a fixed
cadence means **one continuous wait that kept re-logging itself**, not sixteen stalls. Sixteen real
stalls would be irregular; this is a metronome.

What the writer waits for (v0.28.0, same file):

```python
# L659
return not (written_flag and read_count != self.buffer.n_reader)
# L665-668
# this block is written and not read by all readers
# we need to wait until it is read by all readers
# L1029
n_reader = group_world_size - 1
```

A block becomes writable again only once **every** reader has marked it read. So one reader that
stops draining pinches the writer shut while the other readers stay healthy. With TP=2 and
EngineCore as the writer, the readers are `356` and `408`.

And the timeline rules out "busy compiling": `Graph capturing finished` is at `11:37:55`, and **14
of the 16 lines come after it**, through `11:50:31`. The wait outlived capture by more than 13
minutes, so "workers are still compiling" cannot be the explanation for the tail.

### The `/dev/shm` reading is wrong — capacity has its own error

`check_shm_free_space` (L227-247) runs when the buffer is created and raises, with the fix already
in the message:

```
RuntimeError: Insufficient space in /dev/shm: N MiB required, M MiB free.
Increase /dev/shm (e.g. --shm-size or --ipc=host).
```

A small `/dev/shm` therefore fails startup at that line and never reaches the 60-second watchdog.
Checked against the `v0.28.0` tag and `main`; a code search for the claim returns nothing.

## Prior art: this exact line has been diagnosed elsewhere as a UVM livelock

Two public notes, written by teams who hit the same string, are worth reading before blaming
Python:

- [`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`, `docs/uvm-livelock-gb10.md`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/main/docs/uvm-livelock-gb10.md)
  reports that when the kernel pages vLLM memory out under pressure on GB10 (unified memory),
  the UVM driver enters an unrecoverable spin: every engine step stalls and `shm_broadcast.py`
  logs this line every 60 s. Their incident is dated **2026-08-31 — the same day as this log**.
- [`steveseguin/b70-optimization-lab`, `2026-05-10-minimax-xpusmi-live-trace.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/notes/2026-05-10-minimax-xpusmi-live-trace.md)
  hit the same line mid-benchmark and had to kill the run, i.e. it is not a startup-only artefact.

The fingerprint they give is much cheaper to read than a stack:

| signal | healthy | UVM spin |
|---|---|---|
| `nvidia-smi` utilization | tracks the workload | **~96 %** while **power draw is idle (~12–20 W)** |
| `/health` | probes the engine | stays **200** (vLLM V1 only reports an errored flag) |
| `/metrics` tokens | advance | `prompt_tokens_total + generation_tokens_total` **frozen** while requests are queued |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | fires on a hang | **does not fire** |

Reading utilization as "the workers are busy, so they are computing" is exactly the trap this
fingerprint breaks: 96 % util at 12 W means the driver is spinning, not the SM doing work.

**This is a candidate, not this log's conclusion.** The notes are about GB10 unified memory; this
log's device reports `gpu.total_gib 79.14`, and the reporter never states the hardware. A single
`nvidia-smi dmon` line sampled during the stall separates the two readings:

- util ≈ 96 % with power ≈ 12–20 W → the workers are not reading because the driver is stuck
  (UVM/livelock reading), and `py-spy` will show them parked outside the compute path;
- util low with power normal → the ring is blocked by scheduling, and the `py-spy` comparison
  below is the answer.

## What to check next

`py-spy dump` on **both** workers, `pid 356` and `pid 408`. EngineCore (260) is the writer, so a
stack that is *not* inside the broadcast read path belongs to the worker holding the ring; the
other one is the innocent reader still letting the writer advance.

For family 1, the question a stack would answer is where the allocator was pinned: ask for
`torch.cuda.memory_summary()` at the moment of the first 20 MiB failure, or re-run with a lower
`gpu_memory_utilization` and see whether burst A and the 1 s gap disappear.

## Reproducing these numbers

```bash
servediff show 54559.log        # the table above, with line numbers
servediff diff a.log b.log      # same flags, two runs: which of these move
```
