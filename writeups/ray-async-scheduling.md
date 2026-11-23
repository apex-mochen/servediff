# `Async scheduling will be disabled because it is not supported with the ray distributed executor backend`

The line is a WARNING on a 4-node vLLM run, and the flag it contradicts was accepted:

```
(APIServer pid=2483545) WARNING 04-05 16:37:36 [vllm.py:780] Async scheduling will be disabled
  because it is not supported with the `ray` distributed executor backend.
(APIServer pid=2483545) INFO 04-05 16:37:36 [vllm.py:790] Asynchronous scheduling is disabled.
```

Two lines, one decision. The first says *why*, the second says *what*. Neither is in `--help`,
and `--async-scheduling` stays in the printed args as if it applied.

## What it changes

Async scheduling overlaps the scheduler step with the model step. When the executor is Ray,
the engine refuses it and runs the fully synchronous path — a different token loop, a different
step latency, and a KV-cache profile that no longer matches a single-node run with identical
flags.

`servediff` therefore keeps the pair apart:

```
scheduler.async         disabled
scheduler.async_reason  ray executor
```

A diff against a single-node baseline then reads "async on, then off", instead of two logs that
look identical because both print the same flags. That distinction is the point: one value
answers *is it on*, the other answers *why not*.

## The same box, one line apart

The run also prints, before the refusal:

```
Tensor parallel size (16) exceeds available GPUs (4). This may result in Ray placement group …
```

which becomes `parallel.tp_exceeds_gpus=tp 16 > gpus 4`. A TP16 request over four nodes of four
GPUs each is legal on paper, and the warning is the only place the mismatch is stated. Together
with `parallel.world_size=16` and the rank line from each node, it is enough to tell a
mis-sized request from a cluster that lost a node.

## Why this is a corpus problem, not a flag problem

The three facts above exist only in the warning block. `args` records what was asked;
`--async-scheduling` and `--tensor-parallel-size` both appear there unchanged. Everything that
decides whether the run is comparable to the last one is printed once, in a WARNING, and then
the log moves on. That is the whole reason this tool reads warnings as configuration.

The fixture (`tests/fixtures/vllm-forum-2026-04/ray-4-node-random-fail.log`, topic 2526) also
ends in `startup.error=ray.exceptions.RayTaskError(KeyError)` — the failure that made the
poster ask — so the log can be read as "these settings, then this crash" instead of two
unrelated screenshots.
