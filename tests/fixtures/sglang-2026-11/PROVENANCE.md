# Provenance: sglang-2026-11

| file | source | date | why it is here |
|---|---|---|---|
| `sglang-52318-assertion.log` | [sglang#52318](https://github.com/sgl-project/sglang/issues/52318) | 2026-11-22 | A pure traceback (`assert input_size_per_partition % group_size == 0` in compressed-tensors weight creation) with **no** startup lines. It is the negative case: `servediff show` must not pretend to read it, and must say what it is instead. |

The block is copied verbatim from the issue body (public GitHub HTML, `<pre>` extracted); the
trailing `<frozen runpy>` warning in the source page is not part of the paste and is not here.

Expected: `show` exits 2 with `no vLLM or SGLang configuration lines recognised` plus a hint
that the file is a traceback; `audit` reports `nothing worth a look` (31 lines of noise).
