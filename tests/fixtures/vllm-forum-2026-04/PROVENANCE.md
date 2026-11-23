# Provenance: vllm-forum-2026-04

One post from `discuss.vllm.ai`, read through the forum's public JSON API (`/t/2570.json`),
not by copying a browser page.

The forum does **not** render long logs as one `<pre><code>` block: it splits a pasted log
into paragraphs and terminates each line with `<br>`. This file is therefore extracted, not
verbatim: `<br>` became a newline, HTML tags were stripped, and only lines carrying a real
log preamble (`[module.py:NN]`) were kept. The author's closing paragraph of prose about the
warning was dropped (an earlier, looser filter kept it because the paragraph says "warning"),
and the en dashes in the model path were left as served.

| file | topic | posted | why it is here |
|---|---|---|---|
| `qwen36-35b-fp8-eager.log` | [2570 — Warning while serving Qwen/Qwen3.6-35B-A3B-FP8](https://discuss.vllm.ai/t/2570) | 2026-04-20 | vLLM 0.19.0, single GPU, `enforce_eager=True`, FP8 weights, a multimodal-capable Qwen3.5-MoE architecture run text-only. It is the cleanest "eager mode" log in the corpus — CUDA graphs off for two independent reasons (the flag and the mode) — and the only one that shows a default W8A8 block-FP8 kernel config being picked because no config file exists. The GDN prefill kernel line here has no `(requested=…, head_k_dim=…)` suffix at all, which is why that rule now treats the parenthetical as optional. |

The post is public and the topic page is still up.

Facts this file is expected to produce (asserted in `tests/test_cli.py`):

- `engine.version=0.19.0`, `model.architecture=Qwen3_5MoeForConditionalGeneration`,
  `model.max_len=32768`, `quant=fp8` via `backend.moe=TRITON (Fp8)`
- `cuda_graph=disabled (enforce_eager)`, `cuda_graph.mode=none`, `compile=disabled (enforce_eager)`
- `gdn.prefill_kernel=Triton/FLA` (no parenthetical in this build),
  `quant.kernel.config=default W8A8 (no config file)`,
  `quant.kernel.selected=Fp8LinearMethod: CutlassFP8ScaledMMLinearKernel`
- `arg.model` keeps the two hyphens of the Hugging Face cache directory
  (`…/models--Qwen--Qwen3.6-35B-A3B-FP8/…`) because the en dashes the forum printed are
  folded back; without that fold the path would read `models–Qwen–…` and could never equal
  the server's own path.

It audits to zero shapes worth a look. The ASCII banner and the `non-default args` dump are
noise; the eager-mode cudagraph line is noise because `cuda_graph` already records it.
