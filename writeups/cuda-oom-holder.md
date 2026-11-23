# `CUDA out of memory` on a card that looks free: read which "free" the line means

You sized the model against `nvidia-smi`, the card showed plenty free, and vLLM died with
`torch.OutOfMemoryError`. A minute later `nvidia-smi` looks free again, so the obvious conclusion is
that vLLM is misreading the card. It is not. The failure line carries three numbers, and the
question is which pool "free" belongs to. There are two wordings and they mean different things.

## Wording 1 — free is the *device*, and someone else is on it

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB.
GPU 3 has a total capacity of 95.00 GiB of which 600.06 MiB is free.
Process 3617065 has 94.41 GiB memory in use.
```

| number | value | what it means |
|---|---|---|
| total capacity | 95.00 GiB | the card |
| another process holds | 94.41 GiB | not yours — a different pid |
| free at that instant | 600.06 MiB | what was left |
| vLLM asked for | 768.00 MiB | more than what was left |

`94.41 + 0.59 ≈ 95.00`: the card was fully occupied by another job when vLLM asked for 768 MiB.
No flag fixes this — `--gpu-memory-utilization` divides memory that must be free first, and there
was none. This is [forum topic 1234](https://discuss.vllm.ai/t/1234) exactly: vLLM 0.9.0, TP 4,
weights fit, one of the four cards already held by another job.

A free-looking card one minute later is not a contradiction: the message is a snapshot, and the
other job may have exited in between.

## Wording 2 — free is the *allocator pool*, and the card is fine

```
[rank0]: ... CUDACachingAllocator.cpp:3933 memory allocation failed with OOM on device 0
while trying to allocate 20971520 bytes (free: ... , total: ...)
```

Same words, different pool. Here `free` is what PyTorch's caching allocator had left **in its own
reserved segment**, not what the device had. In our own corpus (39 of these lines in 2.318 s, on
two ranks) 20 MiB was requested with 9.19 MiB and 10.75 MiB free, while the device summary line
two lines later reports `gpu.free_gib 78.46` — the card was three-quarters empty. The pool had
been sized by `gpu_memory_utilization × total = 0.95 × 79.14 = 75.18 GiB` and was full.

So "the card has free memory" and "the allocator can allocate" are different statements, and only
the second one decides whether vLLM starts. Raising `--gpu-memory-utilization` cannot help a card
that is already shared, and it is the only thing that helps a pool that is full.

## Telling them apart in one command

```
$ servediff show serve.log | grep torch.oom
torch.oom_device           3
torch.oom_free_mib         600.06
torch.oom_holder_gib       94.41
torch.oom_requested_mib    768.0
torch.oom_total_gib        95.0
```

`torch.oom_holder_gib` only exists for wording 1 — it is the size held by that other process. When
it is present, the card is shared; when it is absent and `gpu.free_gib` is large, you are looking at
the allocator pool. The pid from the line is deliberately **not** a fact: it changes every run and
would make every diff say "different" for a reason that means nothing. The holder's *size* is stable
enough to be one.

```
$ servediff diff run1.log run2.log
~ torch.oom_device     3 -> 2
= torch.oom_holder_gib 94.41
```

Same shortage, different card — a moved `CUDA_VISIBLE_DEVICES`, not a new bug.

## Finding the holder

The older wording gives you the pid (`Process 3617065`); the newer one does not. Either way, ask
the driver while the failure is happening rather than afterwards:

```
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

Two things worth ruling out before blaming a colleague:

- **It can be your own earlier run.** A killed `vllm serve` whose workers never exited leaves
  `VLLM::EngineCore` entries holding their slice. Re-check after stopping it, instead of
  reproducing and reading the same number again.
- **It can be a different physical card than you think.** `CUDA_VISIBLE_DEVICES` renumbers devices,
  so GPU 3 in the message is the fourth *visible* card — often physical 7 on an 8-GPU node.

## What the line cannot tell you

Not how long the other process will stay, and not whether the sharing is by design. If `total`
comes out well under the card you bought — 79.14 GiB on a 96 GiB part — the driver has already
partitioned it; check `nvidia-smi -L` for `MIG` entries before blaming vLLM.
