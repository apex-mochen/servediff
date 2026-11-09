# Same flags, 0.28 dies on WSL and 0.30 starts — what the startup log reveals

A write-up of one real upgrade, kept in the repo so the diff below has a permanent home
instead of living only inside a release note.

## The situation

- Laptop GPU: one RTX 4060 Laptop, 8 GB, under WSL2.
- The two `vllm serve` commands differed in exactly one thing: the version
  (0.28.0 vs 0.30.0). Flags, model, `--gpu-memory-utilization`, everything else
  identical.
- 0.30.0 answered `/health` and a completion. 0.28.0 died during engine init with:

  ```
  RuntimeError: UVA is not available
  ```

Benchmark result files would not have explained this: they record the client side, and
half of the run never produced a result. The explanation is in the startup log.

## What the log says

Both versions pick the V2 model runner (`Using V2 Model Runner`). What differs is how
they react to WSL2 having no pinned memory:

| | 0.28.0 | 0.30.0 |
|---|---|---|
| startup | `failed` | `ready` |
| startup.error | `RuntimeError: UVA is not available` | `<absent>` |
| uva_buffers | `<absent>` | `fallback: device memory` |
| backend.attention | `<absent>` (init stopped) | `FLASH_ATTN` |
| engine.version | `0.28.0` | `0.30.0` |

0.30.0 prints one warning and carries on:

```
Pinned memory is not available on this platform; falling back to device memory for UVA buffers
```

That fallback is vllm#56908, first released in 0.30.0. So the behaviour difference is not
a misconfiguration on the user's side, and not a benchmark-side effect — it is a
deliberate change in the engine, visible in one log line.

## How the diff was produced

```console
$ servediff diff vllm-0.28.0.log vllm-0.30.0.log
warning: baseline server failed to start: RuntimeError: UVA is not available; facts after the failure are absent because startup stopped, not because they were off

note: baseline: no all-reduce facts: world_size=1, so there is no all-reduce group and no backend is chosen; a TP>1 run is what makes this path observable
note: candidate: no all-reduce facts: world_size=1, so there is no all-reduce group and no backend is chosen; a TP>1 run is what makes this path observable

fact                             baseline                                 candidate                                status
backend.attention                <absent>                                 FLASH_ATTN                               only_candidate
backend.flash_attn_version       <absent>                                 2                                        only_candidate
backend.sampling_flashinfer      <absent>                                 disabled                                 only_candidate
compile.artifacts                <absent>                                 25 entries, 3 artifacts, 4996920 bytes   only_candidate
compile.autotune_gemm            <absent>                                 disabled (not enough SMs)                only_candidate
compile.dynamo_s                 <absent>                                 9.43                                     only_candidate
compile.graph_s                  <absent>                                 13.12                                    only_candidate
compile.range                    <absent>                                 1,2048                                   only_candidate
compile.total_s                  <absent>                                 31.08                                    only_candidate
compile.warmup_s                 <absent>                                 1.02                                     only_candidate
cpu.torch_threads                <absent>                                 16 -> 1                                  only_candidate
cuda_graph.capture_gib           <absent>                                 [0.25, 0.17]                             only_candidate
cuda_graph.capture_s             <absent>                                 [9, 11]                                  only_candidate
cuda_graph.captures              <absent>                                 ["PIECEWISE x51", "FULL x2", "FULL x35"] only_candidate
cuda_graph.pool_gib              <absent>                                 0.17                                     only_candidate
cuda_graph.profiling             <absent>                                 enabled (default since v0.21.0)          only_candidate
engine.init_s                    <absent>                                 109.17                                   only_candidate
engine.version                   0.28.0                                   0.30.0                                   changed
gpu.budget_gib                   <absent>                                 3.6                                      only_candidate
gpu.free_gib                     <absent>                                 6.93                                     only_candidate
gpu.total_gib                    <absent>                                 8.0                                      only_candidate
gpu.utilization                  <absent>                                 0.45                                     only_candidate
gpu.utilization_effective        <absent>                                 0.4151                                   only_candidate
gpu.utilization_to_match_kv_cache <absent>                                 0.4849                                   only_candidate
jit.monitor_mode                 <absent>                                 warn                                     only_candidate
jit.warmup_s                     <absent>                                 0.0                                      only_candidate
kernel.ir_op.gelu_and_mul_sparse <absent>                                 triton|native                            only_candidate
kv_cache.layout                  <absent>                                 LBNHC                                    only_candidate
kv_cache.max_concurrency         <absent>                                 69.3                                     only_candidate
kv_cache.memory_gib              <absent>                                 1.62                                     only_candidate
kv_cache.tokens                  <absent>                                 141936                                   only_candidate
model_loading.gib                <absent>                                 0.93                                     only_candidate
model_loading.seconds            <absent>                                 19.122619                                only_candidate
parallel.roles                   DP rank 0, PP rank 0, PCP rank 0, TP ... DP rank 0, PP rank 0, PCP rank 0, TP ... changed
sampling.overridden              <absent>                                 {'repetition_penalty': 1.1, 'temperat... only_candidate
server.bind                      <absent>                                 http://0.0.0.0:8031                      only_candidate
server.scale_out                 <absent>                                 disabled                                 only_candidate
server.tasks                     <absent>                                 ['generate']                             only_candidate
serving.chat_template_content_format <absent>                                 string                                   only_candidate
startup                          failed                                   ready                                    changed
startup.error                    ["RuntimeError: UVA is not available"... <absent>                                 only_baseline
uva_buffers                      <absent>                                 fallback: device memory                  only_candidate
weights.available_ram_gib        <absent>                                 4.23                                     only_candidate
weights.cache                    <absent>                                 miss (from scratch)                      only_candidate
weights.checkpoint_gib           <absent>                                 0.92                                     only_candidate
weights.filesystem               <absent>                                 EXT4                                     only_candidate
weights.load_s                   <absent>                                 10.49                                    only_candidate
weights.prefetch                 <absent>                                 disabled (EXT4 is not a network FS)      only_candidate

48 difference(s); 13 fact(s) identical. Absent means the log did not print it, not that it was off.
```

Exit code `1` here means "the two logs differ", not "the tool failed". `2` would mean it
could not read one of them.

Most of those `only_candidate` rows are honest consequences of the baseline dying early:
init stopped, so it printed nothing after the failure. Reading them as "0.28 had these
turned off" would be exactly the mistake servediff refuses to make — that is why the
warning line is printed above the table, and why `<absent>` is a value rather than a
blank.

## What this write-up does not claim

- It does not say UVA caused a slowdown. Here it caused a crash. The tool reports what
  the log printed; whether a difference costs throughput is a benchmark question.
- It does not generalise to other GPUs. The fallback is about *pinned memory being
  unavailable*, which on a normal Linux host with a real NVIDIA driver is not the case.

## Reproducing

```console
$ pip install servediff
$ servediff demo                       # two excerpt logs shipped with the package
$ servediff capture -o before.log -- python -m vllm.entrypoints.openai.api_server --model <model> ...
$ servediff capture -o after.log  -- <the same command>
$ servediff diff before.log after.log
```

`capture` runs the server, tees its log, and snapshots the environment *before* startup,
so the older version's log does not have to be hunted down after the fact.
