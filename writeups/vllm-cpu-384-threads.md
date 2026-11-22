# vLLM CPU backend on a 384-thread box: "only two cores are busy" — what the startup log already answers

A June 2026 report on a 2× EPYC 9655 (384 threads, 1.5 TB RAM, CPU-only, Qwen3-8B) says the
server comes up but `htop` shows about two threads pegged, and the thread settled it with
"downgrading to 0.19.1 solved it". Nothing in the paste was ever checked against the flags
that were actually applied, so the thread ends with a version rollback and no cause.

The paste is 22 lines and it is enough to rule several explanations in or out. Everything below
was read from that paste with `servediff 0.4.64` — no synthetic input, no inferred flags. The
fixture is in the repo at `tests/fixtures/vllm-forum-2026-06/`, with `PROVENANCE.md` saying
which lines came from where.

```
$ servediff show vllm-cpu-2x-epyc-9655.log
cpu.core_ids                     [192, 193, …, 286]                                 line 8
cpu.numa_membind                 MEMBIND nodes [0]                                  line 12
cpu.reserved_cpus                [287]                                              line 9
cpu.threads_bind_mode            auto (auto_setup=True, skip_setup=False)           line 6
engine                           vllm                                               line -
kernel.ir_op.fused_add_rms_norm  native                                             line 3
kernel.ir_op.rms_norm            native                                             line 3
kv_cache.cpu_node                675.87/755.19 GiB (node 0, auto)                   line 19
kv_cache.max_concurrency         150.19                                             line 21
kv_cache.non_kv_gib              18.9                                               line 19
kv_cache.requested_gib           694.77                                             line 19
kv_cache.tokens                  4921472                                            line 20
model_runner                     v1 (v2 requires Triton)                            line 13
parallel.dp_leader               node 0 / dp 0                                       line 4
parallel.roles                   DP rank 0, PP rank 0, PCP rank 0, TP rank 0, …     line 15
parallel.world_size              1                                                  line 14
scheduler.async                  enabled                                            line 2
triton.drivers                   0 active (Triton disabled)                         line 10
weights.available_ram_gib        1508.73                                            line 16
weights.checkpoint_gib           15.26                                              line 16
```

21 facts out of 22 lines, and `audit` on the same file returns `0 shape(s) worth a look`.

## The binding did not fail

The claim "vLLM only used two cores" is contradicted by four lines of the paste itself:

```
INFO [ompmultiprocessing.py] VLLM_CPU_OMP_THREADS_BIND='auto', auto_setup=True, skip_setup=False
INFO [ompmultiprocessing.py] reserved_cpus=[287]
INFO [cpu_worker.py] local_rank=0, core ids=[192, 193, …, 286]
INFO [cpu_worker.py] NUMA binding: Using MEMBIND policy, NUMA nodes (0)
```

95 cores of one NUMA node were handed to this rank, one core was kept for the control plane, and
memory was bound to node 0. `servediff` counts the ids rather than trusting the sentence: 95
integers from 192 to 286, plus `[287]` reserved. That is a wide bind. If `htop` shows two busy
cores, the question is not "did vLLM bind" but "where did those 95 threads actually run".

## Three checks that separate the causes, all outside the flags

`htop` colours per core; a bound thread is only *allowed* on a core, it does not have to be there.
The cheap way to see the real placement:

```
grep Cpus_allowed_list /proc/<pid>/status           # what the kernel permits, per process
ps -L -o tid,psr,pcpu -p <pid> | sort -k3 -nr | head
```

- If `Cpus_allowed_list` is `192-286` and `ps -L` shows the busy threads on those cores, binding
  worked and the bottleneck is elsewhere (one request in flight, a serialized Python path, the
  sampling loop) — no flag will fix it.
- If `Cpus_allowed_list` is a small range such as `0-7`, the container's cpuset does not include
  192–286. Linux clamps `sched_setaffinity` to the cpuset, so the 95 threads all land in the few
  allowed cores: a couple of cores look pegged, the rest idle, and every binding variable in the
  log is technically correct. `cat /sys/fs/cgroup/cpuset.cpus.effective` says what the box may
  use at all, and the bound range has to sit inside it.

The second case is the one that a version rollback can "solve" by accident: an older build may
have picked a different core range that happened to fall inside the cpuset.

## The `nobind` result is the same arithmetic

The thread also reports that `VLLM_CPU_OMP_THREADS_BIND=nobind` gave "about half of the threads".
That is the OpenMP default, not a bug: with no binding, OpenMP spawns one thread per *physical*
core, and 384 threads on this box means 192 physical cores. "Half" is exactly the expected count.
It also means the `nobind` run removes the pinning that would have kept work on the node that
holds the KV cache.

## Two things the paste records that no flag states

```
INFO [importing.py] Triton not installed or not compatible; certain GPU-related functions will not be available.
INFO [cpu_worker.py] Model runner v2 requires Triton; using the v1 model runner instead
INFO [cpu_worker.py] Auto set (675.87/755.19) GiB for KV cache on node 0, with 694.77 GiB requested memory for the worker. 18.9 GiB memory was consumed by non-kv usages.
```

Triton is installed but reports zero active drivers, so the run silently drops to the v1 model
runner — a fact about what executed, not about what was configured. And the KV cache was sized by
the auto path to 675.87 of 755.19 GiB with 18.9 GiB lost to non-KV allocations, giving 4 921 472
tokens and a concurrency ceiling of 150.19 at 32 768 tokens per request. On a 384-thread box
serving one 8B model, that ceiling is far above what one request stream needs, which makes
"threads are idle because there is nothing to do" a live hypothesis, not a contradiction.

## What this does not claim

Not that 0.20.0 regressed CPU binding. The paste is a single run at unknown flags; the two logs
(0.19.1 and 0.20.0, identical command line) would settle it, and `servediff diff old.log new.log`
prints exactly the facts that changed between them. Until that diff exists, the honest statement
is: binding was applied to 95 cores, and the observed utilisation is consistent with a cpuset
clamp or with too little work, but not with a failed bind.

Source log: `https://discuss.vllm.ai/t/the-current-vllm-cpu-backend-is-not-working-properly/2708`.
