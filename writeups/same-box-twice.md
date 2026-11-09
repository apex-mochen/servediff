# The same box, measured twice: why a diff has to agree with itself

A diff tool earns trust the first time it tells you something you did not know, and loses
it the first time it tells you that two identical runs differ. This note is about the second
case, because it is the one that never shows up in a demo on one machine.

## Eight numbers where there is one measurement

An 8-way tensor-parallel vLLM start logs this once per worker:

```
(Worker_TP2 pid=2717794) INFO [model_runner.py:302] Model loading took 1.14 GiB and 36.339381 seconds
(Worker_TP3 pid=2717795) INFO [model_runner.py:302] Model loading took 1.14 GiB and 36.384071 seconds
(Worker_TP7 pid=2717799) INFO [model_runner.py:302] Model loading took 1.14 GiB and 36.395200 seconds
(Worker_TP0 pid=2717792) INFO [model_runner.py:302] Model loading took 1.14 GiB and 36.312437 seconds
(Worker_TP1 pid=2717793) INFO [model_runner.py:302] Model loading took 1.14 GiB and 36.446548 seconds
```

Physically there is one number here — roughly 36.3 seconds — plus 0.13 s of scheduling
jitter spread across the ranks. Read literally, there are eight, and the tool used to store
all eight, in the order they happened to land in the file.

That has a consequence which is easy to miss until it bites: compare two runs of *the same*
box, same image, same flags, and the two value lists almost never match, because worker 3
was 0.4% slower this time and worker 5 was 0.3% faster. The row says `changed`. Nothing
changed. The reader now has to decide whether to distrust the tool or the machine, and the
answer is: the tool.

## What it reads now

The four facts that every worker measures for itself — `model_loading.seconds`,
`model_loading.gib`, `weights.load_s`, `weights.available_ram_gib` — are read from the
lowest rank present in the log, and the other ranks' numbers are kept in the evidence line
rather than dropped:

```
model_loading.seconds            36.312437                                            line 94
  evidence: (rank 0 of 5 workers read; other workers: [36.339381, 36.384071, ...])
```

The distinction that keeps this honest is *which rank*, not *how many lines*. TP0 in the
same log also prints two genuinely different weight loads — a backbone and a draft head:

```
(Worker_TP0 ...) Loading weights took 28.64 seconds
(Worker_TP0 ...) Loading weights took 5.66 seconds
```

Same rank, two measurements, so both stay. Collapsing by count alone would have deleted the
second model's load time, which is exactly the kind of silent loss this project refuses.

## A magnitude, not just a word

The other half of the same problem is the word itself. `changed` on `compile.total_s` reads
identically at 32.58 s → 31.08 s (−4.6%) and at 32.58 s → 130 s (+299%). Only one of those
is worth a paragraph in a bug report. Numeric rows now carry the relative move:

```
compile.total_s                  32.58                           31.08                 changed  -4.6%
engine.init_s                    74.0                            109.17                changed  +47.5%
model_loading.seconds            13.191946                       19.122619             changed  +45.0%
```

Text rows keep the bare status — `VLLM_CUTLASS (NvFp4)` → `MARLIN (NvFp4)` has no magnitude,
and inventing one would be a lie.

Both conventions are additive: the counts do not move. The two comparisons documented in
the README still report `48 difference(s); 14 fact(s) identical` and `30 / 33` after this
change, which is how you can tell the presentation changed and the measurement did not.

## Why this is worth a version

The failure mode is invisible on one machine and guaranteed on eight. Anyone running TP8
across four nodes — the reason they are reading startup logs in the first place — hits it on
their first comparison of two runs. "Two identical runs differ" is a worse bug than "the
tool misses a line", because missing a line is visible (`servediff audit` prints it) and a
false difference is not: it looks like a finding.
