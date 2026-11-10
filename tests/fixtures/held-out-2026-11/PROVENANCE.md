# Held-out corpus, November 2026

A second batch, collected a month after the first, to test whether the noise table learned the
October pastes or learned logs in general. Same method: GitHub search API
(`repo:vllm-project/vllm is:issue created:>2026-11-01`, the same for `sgl-project/sglang`),
the longest fenced block in the body, kept only at 40 lines or more, bodies unedited.

The query returned twelve blocks. Five of them (`vllm-66422`, `vllm-66437`, `vllm-66451`,
`vllm-66525`, `sglang-49221`) were created after 2026-11-01 and are already in
`../held-out-2026-10/`, so they are not duplicated here. The seven below are new.

| file | issue |
|---|---|
| `vllm-66610.log` | vllm-project/vllm#66610 |
| `vllm-66627.log` | vllm-project/vllm#66627 |
| `vllm-66629.log` | vllm-project/vllm#66629 |
| `sglang-49295.log` | sgl-project/sglang#49295 |
| `sglang-49334.log` | sgl-project/sglang#49334 |
| `sglang-49378.log` | sgl-project/sglang#49378 |
| `sglang-49403.log` | sgl-project/sglang#49403 — a reproducer script pasted as a body |

## What it proves

`servediff audit tests/fixtures/held-out-2026-11` reports `7 log(s): 0 with shapes worth a look`
at 0.4.37. At 0.4.36 the same command reported **one file with a line worth a look** — the
conditional expression `run(a) if a.cmd == "run" else compare(a)` in `sglang-49403.log`, a
shape the October batch never produced. The fix was one line in the noise table.

Six of the seven were clean before the fix, so this batch mostly measures the rules holding
across a month boundary rather than a new escape.

## What it does not prove

Same caveat as the October corpus: a sample, not a census. The absence of a false positive here
is not the absence of one in the wild.
