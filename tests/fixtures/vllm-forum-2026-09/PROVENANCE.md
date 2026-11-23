# Provenance: vllm-forum-2026-09

One post from `discuss.vllm.ai`, read through the forum's public JSON API (`/t/<id>.json`),
not by copying a browser page.

This forum does **not** render long logs as one `<pre><code>` block: it splits a pasted log
into paragraphs, keeps each line and terminates it with `<br>`. The file here is therefore
extracted, not verbatim: `<br>` became a newline, HTML tags were stripped, and only lines that
carry a real log preamble (`[module.py:NN]`, an ASCII banner line, `Traceback`, `File "…"`)
were kept. The prose paragraphs the author wrote between log lines were dropped, nothing was
re-wrapped, and the typographic quotes the forum returned (`‘align’`, `won’t`) were left as
the forum served them — that is what a user copying from the page actually pastes, and
`cli.parse` now curls them back to ASCII before matching.

| file | topic | posted | why it is here |
|---|---|---|---|
| `qwen35-nvfp4-mtp.log` | [2946 — is this normal for vllm serve output, waiting 1 hour before startup](https://discuss.vllm.ai/t/2946) | 2026-09-08 | vLLM 0.28 on Qwen3.5-27B-NVFP4, run once plain and once with `--speculative-config method=mtp num_speculative_tokens=2`. It is the only log we have that shows the speculative path from the inside: `max_num_scheduled_tokens` reset to 2048 by the spec settings, `min_p` and `logit_bias` declared non-working, the MTP layer re-forwarded per step, and MTP weight sharing. Plus the quantised kernels (NVFP4 FlashInfer Cutlass, FP8 Cutlass) and the DeepGEMM fallback to the vendored copy, none of which any flag records. |

The post is public and the topic page is still up. If it is edited later, this snapshot stays
as it was read, which is what a corpus needs.

Facts this file is expected to produce (asserted in `tests/test_cli.py`):

- `engine.version=0.28.0`, `model.architecture=["Qwen3_5ForConditionalGeneration", "Qwen3_5MTP"]`,
  `kv_cache.mamba_mode`, `quant.kernel.nvfp4=FlashInferCutlassNvFp4LinearKernel`,
  `quant.kernel.selected=CompressedTensorsW8A8Fp8: CutlassFP8ScaledMMLinearKernel`,
  `gdn.prefill_kernel`, `gdn.decode_kernel=cuda`, `compile.deep_gemm=vendored (vllm.third_party)`,
  `compile.deep_gemm_feature=["PDL enabled", "E8M0 enabled"]`,
  `sched.max_num_scheduled_tokens=2048`, `sampling.ignored_with_spec=min_p, logit_bias`,
  `spec.mtp_forward_repeat`, `spec.mtp_weights_shared=["embedding", "lm_head"]`,
  `mm.prefix_attention=disabled (multimodal config off)`
- `kv_cache.attention_block_size` and `kv_cache.mamba_page_pad` come out as two values each
  (1568 / 1600 tokens, 0.13% / 0.88%) because the post contains both the plain run and the
  speculative run; that is intended — a diff against another log picks the lowest rank.

It audits to zero shapes worth a look. The ASCII banner, the `non-default args` dump and the
torch.compile progress lines are noise with reasons in `audit.NOISE`.
