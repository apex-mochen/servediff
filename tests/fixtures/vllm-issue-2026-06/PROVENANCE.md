# Where this log came from

- **Source**: https://github.com/vllm-project/vllm/issues/45198 — "[Bug]: vllm hangs at Triton
  kernel JIT compilation during inference: _zero_kv_blocks_kernel with TP=2", opened 2026-06-10
  against vLLM `0.20.2` (Qwen3.5-35B-A3B, TP=2, 2× RTX 6000 Ada, `NCCL_P2P_DISABLE=1`).
- **Not verbatim.** The issue body mixes a full environment dump, the command line, and six
  quoted log lines. This fixture keeps only the four that carry a timestamp and a
  `[module.py:NN]` tag: one `jit_monitor.py:103` warning and three `shm_broadcast.py:698` waits.
  The reporter's own `[triton] load_binary START: _zero_kv_blocks_kernel` lines are left out
  because they carry no module tag and are instrumentation he added, not vLLM output.
- **Why it is here**: it is the second worked shape of the 60-second wait — the one where the
  wait is *caused by compilation*. The warning is at `22:01:40`; the first wait line is at
  `22:02:40`, and a wait line marks the **end** of a 60 s wait, so the wait began at `22:01:40`
  exactly. Nothing at all is printed inside the window. Both numbers come out of the log itself:
  `repeats` reads the period from the message ("found in 60 seconds") and subtracts it.
- **Expected facts**: `kernel.jit_during_inference=_zero_kv_blocks_kernel`. No `startup` fact —
  the run never reached a ready or a traceback line in the quoted excerpt.
- **Expected `repeats`**: one regular group, `3x regular 22:02:40 -> 22:04:40 span 120 s`,
  `gap 60.000 s median`, "the first one began at 22:01:40", "0 other line(s) inside that span".
- The two threads this distinguishes are worth naming: here the fix is compilation-related
  (the reporter's own workaround was `--disable-custom-all-reduce`), while
  `vllm-forum-2026-07/a2-stuck-on-shm-broadcast.log` has no compile event at all and no
  allocator OOM — a reader that stopped draining.
