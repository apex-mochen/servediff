# Held-out corpus, October 2026

Eleven pastes taken from vLLM and SGLang issues created after 2026-10-20, used to test the
audit against output it had never seen. Every log in `../` was hand-copied from an issue body,
and a copy can carry the copyist's formatting; this directory is what a fresh paste actually
looks like.

Collected with the GitHub search API (`repo:vllm-project/vllm is:issue created:>2026-10-20`,
the same for `sgl-project/sglang`), taking the longest fenced block in each issue body and
keeping only blocks of 40 lines or more. Bodies were not edited.

| file | issue |
|---|---|
| `vllm-66392.log` | vllm-project/vllm#66392 |
| `vllm-66402.log` | vllm-project/vllm#66402 |
| `vllm-66422.log` | vllm-project/vllm#66422 |
| `vllm-66437.log` | vllm-project/vllm#66437 |
| `vllm-66451.log` | vllm-project/vllm#66451 |
| `vllm-66525.log` | vllm-project/vllm#66525 — a CPU-only `collect_env`; shipped as `vllm-66525-cpu-collectenv.log` in the main corpus as well, because it is the paste that exposed the padded-colon rules |
| `sglang-49123.log` | sgl-project/sglang#49123 — a benchmark script, not a log |
| `sglang-49129.log` | sgl-project/sglang#49129 |
| `sglang-49132.log` | sgl-project/sglang#49132 |
| `sglang-49133.log` | sgl-project/sglang#49133 |
| `sglang-49221.log` | sgl-project/sglang#49221 |

## What it proves

`servediff audit tests/fixtures/held-out-2026-10` reports `11 log(s): 0 with shapes worth a
look` at 0.4.35. At v0.4.32 the same command reported **8 of 11** files with lines worth a
look — check that out and rerun it to see the rules that this corpus broke.

The one thing it does **not** prove: that the noise table covers log space. Eleven pastes
from two projects over two weeks is a sample, not a census.
