# Provenance: vllm-forum-2026-07

One post from `discuss.vllm.ai`, read through the forum's public JSON API (`/t/2828.json`),
not by copying a browser page.

The post has no `<pre><code>` block: the forum split the pasted log into paragraphs and
terminated each line with `<br>`. This file is extracted, not verbatim: `<br>` became a
newline, HTML tags were stripped, and only lines carrying `[module.py:NN]` were kept. It is
24 lines — the whole startup of the run, ending where the poster gave up.

| file | topic | posted | why it is here |
|---|---|---|---|
| `a2-stuck-on-shm-broadcast.log` | [2828 — Support for Nvidia A2 Tensor Core GPU, stuck on shm_broadcast, unsure how to debug](https://discuss.vllm.ai/t/2828) | 2026-07-23 | The cleanest instance of the 60-second broadcast wait: six lines, **exactly** 60.000 s apart, 300 s total, on a TP2 A2 box. It is the worked example for `writeups/vllm-shm-broadcast-60-seconds.md`: graph capture finished at `09:02:40`, the first wait line lands at `09:03:37`, so the wait began ~3 s **before** capture ended and then outlived it — which rules out "still compiling" and leaves the reader-side stall. |

The post is public and the topic page is still up.

Facts this file is expected to produce (asserted in `tests/test_cli.py`):

- `kv_cache.memory_gib=4.91`, `kv_cache.tokens=407080`, `kv_cache.max_concurrency=24.85`
- `model_loading.gib=8.29`, `model_loading.seconds=3.000243`
- `compile.total_s=43.57`, `compile.graph_s=29.36`, `compile.warmup_s=1.71`
- `cuda_graph.capture_s=4`, `cuda_graph.capture_gib=0.11`

`repeats --top 3` on this file reports `6x regular 09:03:37 -> 09:08:37 span 300 s`,
`gap 60.000 s median` — one wait being re-logged, not six stalls.

It audits to zero shapes worth a look. The per-worker `Skipping CuTeDSL warmup` line is noise:
it appears on every run of a model without CuTeDSL kernels.
