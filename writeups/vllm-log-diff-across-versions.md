# Diffing two vLLM startup logs from different versions: 59 of 75 rows mean "that build printed a line this one did not"

The usual reason to diff two startup logs is an upgrade: the old run worked, the new run is
slower, and the question is "did the server actually run the same thing?". Put a 0.28.0 log
and a 0.33.0 log side by side and you get a table — but most of that table is not about your
configuration at all. It is about which lines each release happened to print.

Measured on two real logs kept in this repo (`vllm-forum-2026-09/qwen35-nvfp4-mtp.log`,
0.28.0, and `vllm-0.33.0-real-rtx4060.log`, 0.33.0), with `servediff 0.4.104`:

```console
$ servediff diff qwen35-nvfp4-mtp.log vllm-0.33.0-real-rtx4060.log
note: versions differ: 0.28.0 vs 0.33.0: a fact that appears on one side only may be a line
this release started or stopped printing, not a setting you changed
...
fact                             baseline                       candidate                        status
arg.model                        unsloth/Qwen3.8-27B-NVFP4       <absent>                         only_baseline
args.source                      vllm non-default args          <absent>                         only_baseline
backend.attention                FLASHINFER                     FLASH_ATTN                       changed
backend.attention_mm             FLASH_ATTN                     <absent>                         only_baseline
backend.candidates               ['FLASHINFER', 'TRITON_ATTN']  ['FLASH_ATTN', 'FLASHINFER', 'TRITON_...  changed
backend.flash_attn_version       <absent>                       2                                only_candidate
...
75 difference(s); 7 fact(s) identical. Absent means the log did not print it, not that it was off.
```

Split those 75 rows by what they can mean:

| status | rows | what it says across two releases |
|---|---|---|
| `changed` | 16 | both logs printed the fact and the values differ — the only rows that describe a setting |
| `only_baseline` | 20 | the older release printed it, the newer one never wrote that line |
| `only_candidate` | 39 | the newer release started printing something the older one never had |

59 of 75 rows are one-sided. Read as configuration, they say "twenty settings were turned
off and thirty-nine appeared" — for two runs where nobody rewrote a launch script. Read as
release drift, they say what happened: 0.33 prints `backend.flash_attn_version`, 0.28 does
not; 0.28 printed `args.source`, 0.33 stopped. None of that is a flag anyone changed.

The trap is the direction of the inference. `<absent>` in a `diff` between two runs of the
*same* build really does mean "this run did not resolve it". Between two builds it can also
mean "this build no longer prints it", and the log itself bears no evidence either way.

## What the tool now says

Since `0.4.102` the first line of the report is the note shown above; `0.4.104` gave `series`
the same treatment. It is not a guess about the cause — it names the two versions and the
one reading that a one-sided row cannot support.

## Comparing the configuration anyway

If the question is "which facts did both runs resolve, and do they match", drop the rows one
side never printed:

```console
$ servediff diff qwen35-nvfp4-mtp.log vllm-0.33.0-real-rtx4060.log --skip-absent
...
weights.checkpoint_gib           21.81                0.92       changed  -95.8%
weights.load_s                   [5.46, 13.04, 1.84]  3.13       changed

16 difference(s); 7 fact(s) identical. 59 difference(s) skipped because one side never
printed those lines (--skip-absent). Absent means the log did not print it, not that it was off.
```

Those 16 rows are facts both releases printed, so both numbers exist — that is what makes
them comparable at all. They are not automatically attributions, though: these two runs are
also different models (`Qwen3.8-27B-NVFP4` in the 0.28.0 log, and a 0.92 GiB checkpoint in
the 0.33.0 one), so a 21.81 → 0.92 GiB movement is a different model, not a different
setting. Narrowing the row set is the first step; keeping the model and the flags fixed is
still the condition for reading the remaining rows as a config change.

## Three versions at once

`series` exists for the regression thread that has old, middle and new runs. The same
one-sided reading applies to every column:

```console
$ servediff series vllm-0.29.1-rocm-collectenv.log vllm-0.30.0-gvisor-cuda-collectenv.log vllm-0.32.0-cuda-collectenv.log
note: versions differ: 0.29.1.dev0+g98dff2a81.d20261002, 0.30.0, 0.32.0: a fact that only
some logs carry may be a line a release started or stopped printing, not a configuration change

fact                             vllm-0.29.1-rocm-collectenv...  vllm-0.30.0-gvisor-cuda-col...  vllm-0.32.0-cuda-collectenv...
engine.git_sha                   98dff2a81                      <absent>                        <absent>
engine.version                   0.29.1.dev0+g98dff2...20261002  0.30.0                          0.32.0
...
32 fact(s) differ across 3 logs; 2 identical. <absent> means the log did not print it, not that it was off.
```

Here the three logs are also different stacks (ROCm, CUDA under gVisor, CUDA), so the
columns cannot answer "what changed in 0.31" either. The note does not claim they can; it
only stops the one-sided rows from being read as settings.

## What this does not tell you

- **Whether a vanished fact was removed or merely turned off.** `args.source` printed in
  one build and not the next may be a log change or a config change. The startup log has no
  field that distinguishes them; the release notes and the code do.
- **That equal versions mean equal behaviour.** Two 0.33.0 runs can differ in every fact
  that matters; the note is silent there, and `diff` is back to its ordinary reading.
- **That a dev build equals the tag it derives from.** `0.33.0rc1.dev910+g4fab5a074` and
  `0.33.0` compare as different versions, because they are different builds.

The commands above are reproducible from the logs in
[`tests/fixtures/`](https://github.com/apex-mochen/servediff/tree/main/tests/fixtures); each
page on this site names the version that produced it. Vocabulary for every key is in
[`facts.html`](facts.html).
