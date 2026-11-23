# Provenance: vllm-forum-2025-09

One post from `discuss.vllm.ai`, read through the forum's public JSON API (`/t/1602.json`),
not by copying a browser page.

This post has no `<pre><code>` block: the forum split the pasted log into paragraphs and
terminated each line with `<br>`. The file is therefore extracted, not verbatim: `<br>` became
a newline, HTML tags were stripped, and only lines carrying `[module.py:NN]` were kept. The
typographic quotes the forum served (`‘*.safetensors’`, `‘mistralai/Mistral-7B-…’`) were left
as served, which is why `weights.format` and `arg.model` are only readable after the quote
fold.

| file | topic | posted | why it is here |
|---|---|---|---|
| `mistral-7b-huge-batched-tokens.log` | [1602 — The vLLM server fails on large max-num-batched-tokens](https://discuss.vllm.ai/t/1602) | 2025-09-13 | The oldest log in the corpus (vLLM 0.10.1.dev1), and the only one from before PCP existed: its rank line has no `PCP rank` at all, which is why `parallel.roles` now takes whatever rank list follows instead of pinning the set. It is also the only log that names the weight file format (`['*.safetensors']`) and the download time, and one of two that print the attention backend for the V1 engine. |

The post is public and the topic page is still up.

Facts this file is expected to produce (asserted in `tests/test_cli.py`):

- `engine.version=0.10.1.dev1+gbcc0a3cbe`, `engine.platform=cuda`,
  `backend.attention=Flash Attention (V1 engine)`,
  `parallel.roles=DP rank 0, PP rank 0, TP rank 0, EP rank 0`,
  `weights.format='*.safetensors'`, `weights.download_s=17.724022`

It audits to zero shapes worth a look. The platform line's second copy, the init wait and the
`non-default args` dump are noise with reasons in `audit.NOISE`.
