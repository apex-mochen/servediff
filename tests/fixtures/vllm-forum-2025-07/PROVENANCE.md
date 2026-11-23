# Provenance: vllm-forum-2025-07

One post from `discuss.vllm.ai`, read through the forum's public JSON API (`/t/1234.json`),
not by copying a browser page.

The post has no `<pre><code>` block: the forum split the pasted log into paragraphs and
terminated each line with `<br>`. The file is therefore extracted, not verbatim: `<br>` became
a newline, HTML tags were stripped, and only lines carrying `[module.py:NN]` were kept. The
author wrote in Chinese between the log blocks; that prose was dropped.

| file | topic | posted | why it is here |
|---|---|---|---|
| `qwen-vram-enough-but-oom.log` | [1234 — 为什么显存足够却启动不了 serve](https://discuss.vllm.ai/t/1234) | 2025-07-30 | vLLM 0.9.0, TP4, a Qwen model, and an `OutOfMemoryError` on GPU 3 while the summary says there is free VRAM — the canonical "显存足够" thread. It is the oldest build in the corpus that loads a plugin group, so it is the only log that prints `plugins.loaded`; it also carries the multi-task default (`generate` out of five declared tasks) and a FlashInfer fallback to the torch sampler. The crash is a worker OOM wrapped in `Exception: WorkerProc initialization failed`. |

The post is public and the topic page is still up.

Facts this file is expected to produce (asserted in `tests/test_cli.py`):

- `engine.version=0.9.0`, `engine.platform=cuda`
- `plugins.loaded=lora_filesystem_resolver -> vllm.plugins.lora_resolvers.filesystem_resolver:register_filesystem_resolver`
- `model.tasks=generate (default of 'generate', 'classify', 'reward', 'embed', 'score')`
- `sampling.topk_topp_fallback=torch-native (FlashInfer unavailable)`
- `torch.oom_device=3`, `torch.oom_requested_mib=768.0`, `torch.oom_free_mib=600.06`,
  `torch.oom_total_gib=95.0`, `torch.oom_holder_gib=94.41` — the card is 95 GiB and another
  process holds 94.41 of it, which is the whole answer to the thread's question
- `startup.error` carries the `torch.OutOfMemoryError` text; the engine core's own wrapper
  (`Exception: WorkerProc initialization failed`) is recorded by its own rule.

It audits to zero shapes worth a look. The plugin group's header/footer, one
`Found nccl from library` per rank, one message-queue handle per rank, the legacy
block-manager warning and the three crash-marker lines are noise with reasons in `audit.NOISE`.
