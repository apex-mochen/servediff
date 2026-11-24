# Changelog

## 0.4.105

`servediff run.log` — the first line a newcomer types once they have a log — answered with
argparse's usage dump and nine choices, none of which says which command reads a log.

The question: bare `servediff` has pointed at the shortest path since the first release, but
the moment a person has an actual log they type the log. That went to argparse, which
refused the whole line, printed `invalid choice: 'run.log' (choose from show, explain, diff,
series, env, capture, demo, audit, repeats)` and exited 2. A mistyped command got the same
wall: `servediff dif` printed the usage dump, and the one word that would have fixed it —
`diff` — was in a list of nine.

Now the first word is answered before argparse refuses the line:

```
$ servediff run.log
servediff: run.log looks like a log, and a log needs a command:
  servediff show run.log        what the log actually resolved to
  servediff audit run.log       the lines no rule reads
  servediff diff run.log other.log    what changed between two runs

$ servediff dif
servediff: unknown command 'dif'; did you mean 'diff'?
```

Both exit 2, as before. A word that is neither close to a command nor a file still reaches
argparse, so a real usage error keeps its usage text. 216 tests, one new and one retargeted
(the old typo test asserted the usage dump; it now asserts the suggestion, and keeps the
`__main__.py` guard on the argparse path).

## 0.4.104

`series` compared three releases without saying they were three releases.

The question: 0.4.102 added `note: versions differ` to `diff`, because a one-sided row
between two releases usually means "that build prints this line and the other does not". The
same reading applies to `series`, which exists for exactly that input — three versions at
once — and it printed the table with no version anywhere except one row inside it. A row
carried by a single column then reads as a setting that changed between runs, which is the
one thing `series` is used to decide.

Now the note comes first, before the table:

```
note: versions differ: 0.29.1.dev0+g98dff2a81.d20261002, 0.30.0, 0.32.0: a fact that only
some logs carry may be a line a release started or stopped printing, not a configuration
change
```

Same sentence in `--json` (`notes`) and `--markdown` (a quote line above the table). Silent
when every log carries one version, and silent when only one version was printed at all —
three logs from one build are not a version comparison. 215 tests, one new (fails when this
change is reverted).

## 0.4.103

`audit` graded the JSON files servediff itself writes as if they were server logs.

The question: `servediff env > env.json` then `servediff audit env.json` reported 395 unread
lines and 394 shapes — for one JSON document — and then said every unread line was noise.
Both halves were wrong in the same way: a snapshot has no log lines, so nothing in it can be
"unread", and nothing in it can be "noise" either. `show env.json` was gentler but still
pointed at `audit`, the one command that could not help. The gate exists to name a file the
parser cannot read; a file that was never a log is a different answer, and the 0.4.101
distinction (empty vs unread) had already started saying so.

Now both commands name the file instead of grading it:

```
show  env.json: this is a servediff env snapshot, not a server log: compare two snapshots
               with `servediff diff a.json b.json`                    exit 2
show  facts.json: this is a servediff facts file, not a server log: compare it with a log
               using `servediff diff <log> --against <this file>`     exit 2
audit env.json: is a servediff env snapshot, not a server log: there are no log lines in
               it to grade                                            exit 2
```

In a mixed corpus the logs are still graded, the JSON files stay out of the count, and the
summary says `1 servediff file(s) skipped as not logs`; `--json` carries `skipped_not_logs` /
`skipped_not_logs_paths` only when something was skipped. A run where every file is
servediff output exits 2 with `nothing to grade` — zero logs graded is not a pass. 214 tests,
two new (each fails when this change is reverted).

## 0.4.102

Comparing two releases printed one-sided rows with no hint that the releases were different.

The question: `diff` says out loud when the engines differ (`only shared facts line up`),
because a vLLM fact and an SGLang fact cannot be compared. Nothing said it when the engines
matched but the versions did not. A 0.28.0 log against a 0.33.0 log gives 75 rows, of which
59 are one-sided — 20 `only_baseline`, 39 `only_candidate` — and those rows mean "that build
prints a line this one never printed" rather than "someone turned a setting off". Reading a
one-sided row as a setting turned off sends a person looking for a flag that was never there,
and the whole point of the tool is not to leave that reading open. (This entry first said
"ten rows before the first real change"; the measured split is above, corrected in the
writeup that accompanies this release.)

Now `diff` starts its notes with:

```
note: versions differ: 0.28.0 vs 0.33.0: a fact that appears on one side only may be a line
this release started or stopped printing, not a setting you changed
```

Same text in `--json` (`notes`) and `--markdown`. The note stays silent when both logs carry
the same `engine.version`, and silent when either log never printed one — an unprinted
version is not evidence of a different version. 212 tests, one new: 0.28.0 against 0.33.0
shows the note, the same file against itself does not, and the JSON carries it.

## 0.4.101

`audit` called an empty file "an unread file", which blames the parser for a log that does
not exist.

The question: a file that yields no fact is named as unread on purpose (0.4.96), but zero
bytes and a paste full of unknown lines fail for opposite reasons. On an empty file `audit`
printed `0 facts: not one configuration line in this file is recognised — an unread file is
not a clean one` and then `nothing unread: every non-empty line matched a rule`, the second
sentence being vacuously true over zero lines.

An empty file is now named as empty, and the two cases stay separate in `--json`:

```text
never-wrote.log: 0 unread line(s), 0 shape(s)
  0 line(s): this file is empty — the command wrote nothing, or its output went to another file
  nothing to read: there is no line in this file
```

`logs_read_nothing` still counts every no-fact file, so `--max-shapes 0` still exits `1`;
`logs_empty` / `logs_empty_paths` name the subset with no line at all, and a corpus row says
`0 line(s): empty file` instead of `0 facts: nothing recognised`. The repair differs: a crashed
redirect needs `2>&1` and a flush, an unread paste needs a rule.

## 0.4.100

One file holding two runs made `repeats` report a span that measured the gap between them.

The question: when a failing run and a working run are pasted into one file, does the span of
a repeated shape still mean "how long this lasted"? On `nemotron3-nano-v100-tp4.log` it did
not: six broadcast warnings spread over three runs printed `6x burst 12:02:47 -> 12:58:20
span 3333 s`, and nine other shapes printed spans of 3000 s or more for bursts that lasted a
second — the number was the distance between runs, not a duration.

A gap now counts as a run boundary when it is both at least 300 s and at least five times the
group's median gap, and such groups print `split` with the run count and the longest single
run:

```text
6x split    12:02:47 -> 12:58:20  span 3333 s in 3 runs, longest 180 s
    gaps 60.000 s median, 1626.000 s max
```

The watchdog reading is unchanged where it was true: the 49-stall CPU log from vllm#67598
still prints `49x regular ... span 2916 s`, and `--json` keeps its keys, adding `segments` and
`longest_segment_s`.

Corpus check: of the 71 logs, exactly one contains segmented groups (the two-run paste, 9
shapes there), so the 300 s / 5x rule does not fire on ordinary watchdogs. Regression test
builds a two-run file by hand and asserts `segments == 2`, `longest_segment_s == 120`, the
`in 2 runs, longest 120 s` wording, and that the "one wait being re-logged" footer is not
printed for a split group.

## 0.4.99

`show` said "this startup waited 360 s before `startup` reported ready" about waits that were
logged after ready.

The question: a file with two runs pasted together — the one that failed and the one that
worked — how many of its broadcast waits belong to the startup the note describes? One of
the 71 corpus logs answered "not all of them": `nemotron3-nano-v100-tp4.log` holds a `ready`
line at line 111 and waits at lines 112, 214, 215, 216 and 217. The note multiplied 6
intervals × 60 s into "this startup waited 360 s (6 min) ... before `startup` reported
ready", which is a claim about the second run's shutdown attributed to the first run's
startup.

Now `parse()` records where each wait sits relative to the first `ready` line
(`stall_split`), and the note branches on it:

- all waits before ready: unchanged — "this startup waited N s (M min) ... before
  `startup` reported ready".
- some waits after ready: "6 interval(s) with no free broadcast block, 360 s (6 min) in
  total; 5 of them came after `startup` reported ready, so they belong to another run or a
  shutdown in the same file, not to this startup".

The fact rows are untouched: `shm.broadcast_stalls` and `shm.broadcast_wait_s` still
describe the whole file, and the new sentence says so instead of implying a single startup.

Corpus check: 75 broadcast-wait lines across 71 logs, exactly one log has waits on both
sides of a ready line, and every log's note now matches its own split (no log prints both
forms). The regression test builds the two-run file by hand, asserts
`stall_split == {"total": 3, "after_ready": 2}`, asserts the new sentence and the absence of
"this startup waited", and asserts a single-run file still gets the old sentence.

## 0.4.98

`series` counted a shutdown traceback as the one thing that differs between identical runs.

The question: three runs of the same configuration, one of which logged a traceback after
reporting `ready`. `diff` of any pair says nothing changed — it drops `after_ready.*`
because a post-ready error says nothing about how the server started. `series` of the three
said **1 fact(s) differ** and exited `1`, and there was no flag to stop it: rows that came
after `ready` were always counted, while `--all` meant something else in this command ("also
print identical rows").

Now `series` applies the same rule as `diff`: the comparable rows decide the count and the
exit code, `after_ready.*` rows are left out, and the number left out is printed under the
summary instead of vanishing. `--all` lists them for detail without changing either the
count or the verdict, so a gate that adds `--all` cannot flip its own result. `--json`
carries `not_compared_after_ready`.

Checked on the fixture corpus: series keys now equal the union of pairwise `diff` keys
across 8 triplets (0 violations, previously 1), and over 12 triplets the exit code matches
`differences > 0` in text, markdown, `--json` and `--all` forms. The regression test builds
three logs by hand, asserts `0 fact(s) differ` with the count printed, and that `--all`
shows the row while still exiting `0`.

## 0.4.97

`--json` had two counts with one meaning each, and the meaning depended on a flag.

The question: for a pair of logs, does `compared_both_sides` answer the same number twice?
It did not. `servediff diff a.log b.log --json` on a corpus pair reported
`differences: 27, compared_both_sides: 27, hidden_absent: 0`; the same pair with
`--skip-absent` reported `27, 0, 27`. Both runs are correct about the rows they printed —
the flag only changes which rows are listed — but the counts moved, and the field a CI job
reads to decide "did anything actually change on the same code path?" was counting the
27 settings neither log printed on the other side. A table of `<absent>` rows and a real
comparison produced the same `compared_both_sides`.

Now the three fields are defined once and always add up:

- `differences`: every row that is not identical `= compared_both_sides + hidden_absent`
- `compared_both_sides`: rows both inputs printed **and** that disagree
- `hidden_absent`: the rest — one side never printed the line

The `--skip-absent` flag now changes only the `rows` array, never the numbers. Text output
also stopped printing "N difference(s) skipped because one side never printed those lines
(--skip-absent)" when the flag was not passed.

Checked over 60 random fixture pairs in both modes: 0 violations, and the accounting holds
in every case. The regression test asserts equal counts across the two modes on the
existing failed-startup pair, and fails if `compared_both_sides` goes back to counting
one-sided rows.

## 0.4.96

A file that reads nothing was the one file `audit` was quiet about.

The corpus question that had never been asked: how many of these files produce a single fact?
Sixteen of the 87 did not. They are reproducer scripts and docstrings pasted into issue bodies
beside the log, and every line of them matched the traceback-frame patterns in the noise table
— `try:`, `flush()`, `time.sleep(N)` — so `audit tests/fixtures --max-shapes 0` answered
**87 log(s), 0 with shapes worth a look** about a corpus it could not read 18% of.

The split is noise versus shapes, and both are statements about unread lines. A file that yields
no fact at all needs a third one:

```
0 facts: not one configuration line in this file is recognised — an unread file is not a clean one
```

Such a file now counts once toward `--max-shapes`, so a baseline cannot accept it away: a
baseline records shapes the parser was allowed to leave unread, not files it never read. In
`--json` the count is `logs_read_nothing` with `logs_read_nothing_paths`, and every log carries
its own `facts` count.

The sixteen pastes moved to `samples/not-startup-logs/`, which ships in the sdist and carries a
README on why. The startup-log corpus is `tests/fixtures/` at **71 logs**, still 0 shapes worth
a look; the new expected-failure set is that folder, and two tests hold both ends: a script
pasted as a log exits `1` under `--max-shapes 0` and names itself, and the corpus contains no
such file. Three older tests gained one fact line in their fixtures so they keep testing shape
gates rather than the new file rule. Tests 204 → **206**.

## 0.4.95

README only; no code change.

The PyPI page is the README, and it argued the case without showing it: five sections of
`## Logs it has read for people` now name public startup logs, each with the write-up that
shows the reading and its line numbers — the card that looked free while `CUDA out of
memory` fired, the same flags where 0.28 dies on WSL and 0.30 does not, "only two cores
are busy" on a 384-thread box, speculative decoding quietly ignoring `min_p` and
`logit_bias`, and a startup reported as 50 minutes whose 49 broadcast waits sum to 2 940 s
while the weights took 3.99 s ([vllm#67598](https://github.com/vllm-project/vllm/issues/67598)).

`README.zh-CN.md` carries the same list. All five links were fetched and answered `200`
before the entry was written.

## 0.4.94 — 2026-11-24

0.4.93's own fix, caught by walking the corpus it changed. The new attention-candidate rule
matched `out of potential backends: `, and the MoE selection line uses the same words:

```
Using FlashInfer CUTLASS Unquantized MoE backend out of potential backends: ['AITER', ...]
```

So one log carried two values under `backend.candidates` -- an attention list and an MoE
list -- and a row with two values reads as a configuration change that never happened. The
rule now requires `attention backend` before `out of potential backends`, which is the part
that distinguishes the two lines. The corpus goes from 6 logs carrying this key to 4, and
none of them carries two values.

The regression test states both halves: a log with both selection lines must yield one
value, the attention one; and the log that exposed it must stay single-valued. 204 tests.

## 0.4.93 — 2026-11-24

`backend.candidates` was written for a line vLLM does not print. The rule matched a bare
`backends: [...]` line; every real log puts the list on the selection line itself:

```
Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', ...]
```

So every log had the chosen backend as a fact and the set it was chosen from invisible. A
build that stops offering FLASHINFER diffed as "FLASH_ATTN, same as before" -- the exact case
the rule's own comment says it exists to catch. Both shapes are matched now.

The second defect was found by testing the first. With the list finally read, `diff` between
a log and the same log with one candidate appended printed two identical-looking cells:

```
backend.candidates   ['FLASH_ATTN', 'FLASHINFER', 'TRITON_... ['FLASH_ATTN', 'FLASHINFER', 'TRITON_... changed
```

and the summary said **0 difference(s), 61 identical**. The 40-column cell cut the value
before the change at column 65. Two values that differ past the column now widen just past
the character where they diverge, so the row stays narrow and readable:

```
backend.candidates   ['FLASH_ATTN', ..., 'FLEX_ATTENTION']  ['FLASH_ATTN', ..., 'FLEX_ATTENTION', 'ZZZ']  changed
```

Two regressions guard it: the list must be read from the selection line of a real log, and
a change at column 65 must be visible in the printed row.

## 0.4.92 — 2026-11-24

`sampling.overridden` read to end of line, so the value it shipped was not the dict the
warning quotes but the dict plus the log's own advice about it:

```
{'repetition_penalty': 1.1, ... 'top_p': 0.8}`. If this is not intended, please relaunch ...
```

Two defects in one string. The fact is not the sampling parameters any more, and the stray
backtick it carries is unbalanced, which in a markdown table cell is a code span that runs
into the next column. The rule now ends at the dict's closing brace; the value is the dict.

Found by scanning every value in the 87-log corpus for characters that break a markdown
table, which turned up exactly two: this one, and `kernel.ir_op.gelu_and_mul_sparse =
triton|native`. The second is not a defect: `|` is how the log lists two candidate kernels,
and `_md_cell` has escaped pipes since the first report, so it renders as `triton\|native`.

The corpus invariant left behind is general: every `sampling.overridden` value in
`tests/fixtures` must be a dict `ast.literal_eval` can read and must not contain the log's
follow-up sentence. 201 tests.

## 0.4.91 — 2026-11-24

The two numbers 0.4.90 added sit eleven seconds apart in a table: `shm.broadcast_stalls 49`
and `shm.broadcast_wait_s 2940`. The question they answer is "why did this startup take
fifty minutes", and nobody multiplying 49 by 60 to get 49 minutes out of two strings is a
reader we can spare. `show` now states the product in the units a person uses, with the
startup outcome attached so a 5-minute wait and a 49-minute one do not read alike:

```
note: this startup waited 2940 s (49 min) for a free broadcast block, across 49 interval(s)
      before `startup` reported ready; the sum is a floor, the first wait began before line 1
```

The same sentence is in `show --markdown` as `> ℹ️`, which is the artifact that gets pasted
into an issue, and it stays out of the report for logs that never waited. The fact rows and
the key set are unchanged: no new fact, no glossary entry, `facts.html` untouched.

This exists because a vLLM issue reports "50m for vLLM startup" and the log it pasted
accounted for that in one line shape: 49 waits, 2 940 s, weights 3.99 s, compile warmup
16.39 s. The counters made it visible; the sentence makes it quotable.

## 0.4.90 — 2026-11-23

`No available shared memory broadcast block found in 60 seconds` was classified as noise.
That was defensible: it is a symptom, and the facts were the allocator lines underneath it.
But a real v0.34 log has 49 of them and then starts fine, so `servediff show` on a startup
that spent 49 minutes waiting printed the same table as a startup that took 90 seconds, and
`audit` offered the most repeated line in the file back as an unread shape.

Two facts now come out of those lines, counted rather than emitted per line (`Fact.add`
collapses identical values, so 49 lines each saying "60 seconds" would arrive as one):

```
shm.broadcast_stalls             49      line 64
shm.broadcast_wait_s             2940    line 64
```

`startup` still reads `ready` — waiting is not failing — and the evidence points at the first
interval, so `--evidence` quotes the line that started the count. `shm.broadcast_wait_s` is a
floor, not the whole delay: the first block of waiting happens before the first line, and the
glossary says so.

The corpus gained two keys (276) and a family (44); `facts.html` was regenerated.## 0.4.89 — 2026-11-23

Two entries in the glossary named an issue. One is a citation (`vllm#54559 another process
held 94.41 of 95.00 GiB`, the log line that explains the OOM); the other was not:

```
  what it does not tell you: ... vllm#69280 read the 350% as actual waste and compared it
  with a measured 1.7-2.7x, which are two different quantities
```

That sentence shipped to PyPI and to `facts.html`, and it is the one a reader sees when they
run `explain kv_cache.padding_waste` — a public page telling a named issue that its author
misread a number. The gloss was right and the form was wrong: documentation should describe
the pattern that misleads, not prosecute the report that taught us the pattern, and the
moment we answer in that thread the text would read as a jab at the person we are answering.

It now says the same thing without the name:

```
  A ratio measured from allocated pools and this printed ceiling are different
  quantities; comparing them reads as a contradiction that is not there
```

`facts.html` was regenerated and redeployed. No parser change; no fact value changed.## 0.4.88 — 2026-11-23

Found while looking at the first `show --markdown` report of 0.4.87, an hour after it shipped.

`compile.range` and `compile.cached_range` rendered as `1,3248`. The log prints a pair —
`compile range (1, 3248)` — and the two captures were joined with a bare comma, so a
`(min_batch, max_batch)` range came out as one number with a misplaced thousands separator.
Everyone who read a table took `1,3248` for a broken value, which is the worst thing a fact
can look like in a report that is pasted into an issue.

Both now read `(1, 3248)`, the way the server prints it. A test had asserted the broken
string (`== "1,2048"`), so the defect was preserved as expected output; it now asserts the
pair and says why.

Nothing else changed: the diff still compares these as strings, so a range change still
shows as one changed row.
## 0.4.87 — 2026-11-23

`diff` had a pasteable report and `show` did not. That left the most common situation — one
log, one question, no second log to compare — with nothing to hand to anybody: the output was
fixed-width terminal text, and a code block in an issue reads as a screenshot of somebody's
terminal, not as a table a maintainer can scan.

`servediff show serve.log --markdown` renders the same facts as a GitHub-flavored report:
one heading, the log path and its engine, a `| Fact | Value | Line |` table, the open notes,
and the two rows every such report needs — the line each value came from, and the sentence
"Absent means the log did not print it, not that it was off."

Two details that decided the shape:

- Values are **not** trimmed to the 60-character terminal column. `_fmt` shortens with "..." so
  rows line up; in a pasted report a value cut mid-word (`cache.radix` in the test fixture
  ends in `hybrid_swa=False hy...`) is a fact the reader cannot use, so the cell carries the
  whole value and the table reflows instead.
- `--evidence` stays opt-in and, when asked for, goes into a `<details>` block — a 31-fact
  log would otherwise open with 31 quoted log lines before the answer.

`show --json --markdown` exits 2 instead of silently choosing one, matching `audit`.
## 0.4.86 — 2026-11-23

`facts.html` existed and nothing pointed at it. The page is the only place that lists all
274 keys with what each one counts, and it was reachable from the site index alone: the PyPI
page — where anyone installing the tool lands — mentioned `explain` but never the page, so an
external reader had no reason to believe there was more than the one key they asked about.

Two links, in the two places a stranger actually stands:

- `README.md`, next to the other note links: "Every fact key, with what it counts and what it
  does not" → the page, described as the same table `servediff explain` reads.
- `servediff explain` now ends a single-key answer, and the family list, with the page URL.
  Someone reading `explain kv_cache.tokens` learns that 273 other keys exist, without a flag.

No parser or glossary change: the output gains one trailing line, the JSON payloads are
untouched.
## 0.4.85 — 2026-11-23

A reading in 0.4.82 was wrong in a way a real issue shows. vllm#69280 (open, unanswered)
quotes `Add 7 padding layers, may waste at most 350.00% KV cache memory` and treats the
350% as the KV memory actually wasted, then compares it with a 1.7-2.7x ratio measured
from allocated pools. The log says "may waste at most": it is an upper bound the planner
prints when caches with different block sizes have to share one page layout, not the
measured waste of an allocated cache. Two different quantities.

`kv_cache.padding_waste` now says so, in both fields:

```
  unit: percent of KV memory, as an upper bound
  what it does not tell you: the log says 'may waste at most', so this is not the measured
  waste of an allocated cache: multiplying it into a memory estimate overstates the loss.
  vllm#69280 read the 350% as actual waste and compared it with a measured 1.7-2.7x, ...
```

The old text ("the share of KV memory lost to padding layers") read like a measurement,
which is precisely how it gets multiplied. The entry now names the issue it was corrected
on, so the next reader can check the reasoning instead of trusting the sentence.

Verified on the issue's own lines before changing anything: `show` reads
`kv_cache.padding_layers=7`, `kv_cache.padding_waste=350.00%`,
`kv_cache.group_sizes=[16, 16, 16, 16, 16, 16, 64, 64, 64, 64, 64, 64]`,
`kv_cache.tokens=349495` from four pasted lines, each with its evidence line.
`pytest` 196 passed; site `facts.html` regenerated from the same table.

## 0.4.84 — 2026-11-23

0.4.83 put the explanation in the terminal output. Two gaps were left, both on the wrong
side of where the text travels:

`--markdown` is the report people paste into an issue, and the reader of that pasted text
is usually not the person who ran it. `--why --markdown` now emits the same grouping as a
section of the report:

```
## What the changed keys mean

- **`compile`** (6 keys: `compile.cached_range`, `compile.dynamo_s`, ...): torch.compile /
  inductor on the backbone: backend, ranges compiled, fusions enabled, seconds spent, ...
  *Does not tell you:* compile.total_s hides which phase grew; compile.dynamo_s, graph_s
  and warmup_s split it
```

The omitted count is stated in the report too ("N more changed key(s) are not explained
here"), so a paste never implies the six shown are the whole story.

`series` compares three or more logs and raises exactly the same question; the flag
existed on `diff` and not there. `series --why` now works in text, markdown and JSON
(`why` + `why_omitted`), with rows untouched.

Both were found by exercising the flag on the paths it claimed to support rather than on
the one path it was written for. `pytest` 196 passed (three new tests: markdown section
present with `--why`, absent without it, series answers the same question); `cli_sweep`
also runs `diff --why --markdown` and `series --why`, no tracebacks.

## 0.4.83 — 2026-11-23

`diff` states a change and stops. 0.4.82 gave you one command to ask what a key means;
this puts the answer in the diff itself, where the question is actually asked:

```
$ servediff diff a2.log sglang-42684.log --why
...
what these changed keys measure (and what they do not):
  compile  (6 keys: compile.cached_range, compile.dynamo_s, compile.graph_s, compile.range,
            compile.total_s, compile.warmup_s)
    torch.compile / inductor on the backbone: backend, ranges compiled, fusions enabled,
    seconds spent, and why a recompile happened.
    [does not tell you: compile.total_s hides which phase grew; compile.dynamo_s,
    graph_s and warmup_s split it]
```

Grouping is the point, not the prose: six `compile.*` keys moved by one version bump are
one explanation, not six copies of it. Six entries is the cap (`--why` on a 46-row diff
would otherwise print a page), and the keys past it are counted, not dropped -- `(N more
changed key(s) not explained here; servediff explain <key> reads any of them)`. `--json`
carries the same grouping as a `why` object plus `why_omitted`, and the `rows` array is
untouched, so an existing consumer does not see a schema change. `--why` is text/JSON
only; against `env` snapshots it says why it does not apply.

One new gate: the glossary may only point at keys the corpus actually produces. Each
`trap` names other keys (`kv_cache.requested_gib`, `backend.attention_forced`,
`sched.max_running_requests`); a rename would leave those as dead references that still
read correct. The test resolves every dotted token in the prose against the key list,
fact-family by fact-family. `pytest` 193 passed; `cli_sweep` now also runs `diff --why`
and `--why --json`, no tracebacks.

## 0.4.82 — 2026-11-23

`show` prints keys (`kv_cache.max_concurrency`, `transfer.staging_total_mib`) and the
only place their meaning lived was the comment above the regex that produced them -- in
the source, where nobody reading the output looks. 0.4.81 made the raw line reachable;
this makes the number readable.

`servediff explain kv_cache.max_concurrency` answers three things: what it counts, what
it is measured in, and what it does **not** tell you -- the third being the one that
costs people a decision:

```
kv_cache.max_concurrency
  unit: sequences
  what it means: How many concurrent sequences the cache holds at the per-request token
                 count the same sentence names.
  what it does not tell you: it divides by that sentence's token count, not by your real
                             request size -- at half the length you get roughly twice the
                             concurrency
```

56 entries cover all 274 keys the bundled corpus produces: 43 family glosses plus 13
exact keys where the family reading is too coarse (`gpu.utilization_effective` is the
fraction reached, not the one asked for; `torch.oom_holder_gib` is the process holding
the card in vllm#54559). A family name prints the whole family. `explain` with no key
lists them all; `--json` for scripts.

Three gates keep this honest rather than decorative:

- every key in `tests/fixtures/expected_fact_keys.txt` must resolve (274 assertions in
  one test) -- a new fact without a reading fails CI;
- the key list shipped in the package must equal the corpus list, so the typo check
  cannot drift into "no entry" on a real log;
- a dotted key no log produces exits 2 and names the closest keys. Before this, a typo
  (`kv_cache.tokenz`) answered with the family gloss and exit 0, which reads exactly
  like documentation.

`tools/build_site.py` now renders the same table to `facts.html` (43 families, 13 exact
keys, 13 pages in the sitemap), so the vocabulary is a public page and not only a
`--help` string. `pytest` 189 passed; `tools/cli_sweep.py` 1 038 calls, no traceback;
`tools/check_wheel.py` also proves `explain` and `--evidence` survive packaging.

## 0.4.81 — 2026-11-23

`show` prints our reading of a log plus the line number it came from. Disagreeing with
the reading needs the server's own text, and finding line 3 000 of a 100 MB file by hand
is where that stops happening — which is exactly the point where someone would have
quoted the line into an issue.

`servediff show server.log --evidence` prints it under each fact:

```
compile.dynamo_s                 9.35      line 6
    line 6: (Worker_TP0 pid=137) INFO 07-19 09:01:58 [backends.py:1148] Dynamo bytecode transform time: 9.35 s
```

Every evidence entry is printed, not just the first, because a fact that two ranks
reported has two lines and the disagreement between them is the interesting part.
Default output is unchanged. `pytest` 183 passed.

## 0.4.80 — 2026-11-23

The first command after `pip install servediff` is `servediff`, and it answered
`error: the following arguments are required: command`. A newcomer has no log yet,
so the useful answer is the shortest path to output: bare `servediff` now prints the
five commands, the docs URL, and exits 0.

`tools/check_wheel.py` closes the gap that everything else in this repo leaves: the
tests run the source tree, but what ships is a wheel, and the `examples/` that
`demo` reads exist only if packaging includes them. It builds the wheel, installs it
into a fresh venv with `--no-index` (so a published copy cannot stand in), and from a
directory with no source tree checks `--version` and `__version__` against
pyproject, the first-run block, `demo` reading its packaged logs, `show`/`audit`/
`repeats`, and that `show --json` parses. It runs as its own CI job.

That check also explains a lie we had been telling ourselves: in a dev venv left
from an earlier install, `servediff --version` reports the *installed* metadata
(0.4.77 on a 0.4.80 checkout), because the version is read from the distribution
rather than a literal. That is correct for pip users and misleading for us, so the
version is now verified where it matters — on the artifact.

`pytest` 182 passed.

## 0.4.79 — 2026-11-23

The corpus tests pin what the commands *print*. Nothing pinned that they *return*.

`tools/cli_sweep.py` runs every command — `show`, `show --json`, `audit` (three flag
shapes), `repeats` (four), `diff`, plus the ways to mistype them — over all 87 logs
and seven hostile files (empty, binary, one 4 MB line, a UTF-16 PowerShell redirect,
CRLF, a `.log.gz`, a file of dashes): 1 009 invocations in 26 s, and it fails the
build on a traceback. Run against the 0.4.77 wheel it reports exactly the four
cases that release shipped broken (`repeats` on `ray-4-node-random-fail.log` and
`qwen-vram-enough-but-oom.log`, plain and `--json`), which is the point of keeping
it: the bug that motivated the harness was one nobody's test would have caught,
because a crash is not a different printed line. It is now a CI step.

A sweep-level check cannot see a wrong sentence, so one invariant moved into
`pytest`: `show` exits 2 *only* when the log carried no facts. That promise was
already load-bearing ("no configuration lines recognised") and was only tested on
the handful of logs each test happened to name.

`servediff env --json` no longer dies in argparse: `env` writes JSON already, and
`--json` is accepted so the habit works on every command instead of printing a
usage dump.

`pytest` 181 passed.

## 0.4.78 — 2026-11-23

`repeats` read the file twice. Two fixes came out of removing that, plus one crash.

**The second pass.** `report` walked the whole log a second time to answer the only
question the cadence can't: was anything else printed *during* the wait. That answer
only needs the timestamps, which the first pass had already collected and thrown
away. Counting them now costs a bisect per repeated shape instead of a full re-read.
96 MB log: `repeats` 22.80s -> 5.96s (it was 37.90s at 0.4.76).

**SGLang logs had no times at all.** SGLang stamps lines `[2026-10-06 00:15:19]`,
and the preamble stripper removed the bracketed timestamp *before* the clock search,
so the only clock on the line went with it: every SGLang wait printed "untimed,
gap unknown". Reading the clock off the raw line first fixes it —
`2x burst 00:15:19 -> 00:15:50 span 31.00 s` where there used to be nothing.

**A crash on real forum logs.** Five copies of the same traceback frame, all
stamped `16:38:14`: count 2+ with a single distinct clock reached a `None` median
and the formatter raised `TypeError`. `repeats` died on
`ray-4-node-random-fail.log` and `qwen-vram-enough-but-oom.log`; both now say
"N line(s) but one clock: no gap to read". All 87 corpus logs run through the CLI
with exit 0.

Output is unchanged on 83 of 87 logs (byte-identical report text); the two SGLang
logs change because they now carry times, and the two crash logs change because
they now print. `pytest` 180 passed.

## 0.4.77 — 2026-11-23

A 96 MB startup log used to take a minute just to read. It now takes eleven seconds.

`show`/`audit` ran every one of the 198 line regexes against every line. A log has
five shapes and a million repetitions of them, so 99% of that work was guaranteed
to fail. Each rule's longest literal (taken at bracket depth zero, so `|` inside a
group can't leak) is lifted out once; lines are then bucketed by their digit-free
shape and only the rules that matched that shape are retried. Shape caches in
`audit` and `repeats` reuse the same trick, and `audit`'s unread-line sampler no
longer walks the full rule list per line.

Measured on the 1 456 284-line / 96 MB log (same machine, warmed page cache):

| command   | before | after |
|-----------|--------|-------|
| `show`    |  59.0s | 11.1s |
| `audit`   |  90.2s | 14.8s |
| `repeats` |  37.9s | 22.8s |

No output changed: every command was compared byte-for-byte against the previous
build over all 87 corpus logs (`0 differing`), and `pytest` still reports 178
passed. `tools/bench_parse.py` regenerates the synthetic log and prints the three
timings, so the number stays checkable instead of remembered.

## 0.4.76 — 2026-11-23

The two files a stranger is most likely to hand it were the two it handled worst.

- **A rotated `.log.gz` is read directly.** It used to answer "no configuration lines
  recognised" about a perfectly good compressed log, which sends you to `zcat` and back.
  The gzip magic is checked on the bytes, the output says `decompressed N byte(s)` on stderr,
  and `audit <dir>` now counts `*.log.gz` alongside `*.log` so a rotated corpus is not a
  partial corpus. bzip2 / xz / zstd are named as containers with the command to unpack them,
  instead of being reported as logs with no configuration in them.
- **`show <directory>` points at `audit`.** `show: tests/fixtures: Is a directory` was a
  strerror; reading a directory as one corpus is a feature of this tool, so the message
  now names `servediff audit <dir>`.

## 0.4.75 — 2026-11-23

The code in 0.4.74 was right; the test guarding it was not, and the Windows job said so.

- **0.4.74's `ci` run on `windows-latest` failed** while uploading to PyPI succeeded, so the
  published wheel is the one you want — but a red check under a release tag is not something to
  leave standing. The defect was in the test: it decoded the child process's stderr as UTF-8,
  and a pipe on Windows uses the ANSI code page (`UnicodeDecodeError: 0x97 ... invalid start
  byte` — an em dash). The note the product prints is pure ASCII and was there all along.
  The assertion now compares the raw bytes, so neither side's locale can make it lie.
- No product change; `servediff` behaves exactly as 0.4.74.

## 0.4.74 — 2026-11-23

A log captured on Windows was not being read at all.

- **`_read_log` sniffs the bytes instead of assuming UTF-8.** PowerShell 5.1's `>` and `Out-File`
  write UTF-16LE with a BOM, so the most natural way to capture a server log on Windows produced
  a file that decoded to NULs and `?`: every `[module.py:851]` anchor broken, and
  `show` answering "no configuration lines recognised" about a perfectly good log. The BOM is
  honoured (UTF-16LE/BE, UTF-8 with BOM), and a NUL-density check covers UTF-16 written without
  one. The chosen encoding is stated once on stderr, so stdout stays pipeable. Applies to
  `show` / `diff` / `audit` / `repeats` and to stdin.
- **The short-paste hint counts its lines again.** `hint: only {} line(s) here` was a plain
  string, not an f-string, so a 2-line paste was scolded with a literal `{}`.
- `writeups/powershell-redirect-is-utf16.md` — the diagnosis (`FF FE`), the four fixes in order,
  and why a silently-empty log sends you after the wrong bug.

## 0.4.73 — 2026-11-23

SGLang's half of the promise gets the numbers it was hiding.

The richest SGLang log in the corpus read 19 facts; it reads 31 now, and the additions are the
ones a run-to-run difference actually turns on:

- `transfer.staging_buffer_mib`, `transfer.staging_buffers`, `transfer.staging_total_mib`,
  `transfer.staging_method` — PD disaggregation allocates KV staging buffers one line each.
  In this log that is **four allocations of 896 MiB: 3.5 GiB of the card** that no argument
  mentions, and the count had to be derived, because `Fact.add` collapses identical values on
  purpose.
- `transfer.custom_mem_pool` — "Staging buffer using cudaMalloc (no custom mem pool)": the
  fallback that costs NVLink / MNNVL transport, printed once, in no flag.
- `env.lib.flashinfer_cubin`, `env.lib.flashinfer_jit_cache`, `env.lib.sglang_kernel`,
  `env.lib.sglang_router`, `env.lib.torchao`, `env.lib.outlines`, `env.lib.interegular`,
  `env.lib.modelscope` — the library block SGLang prints before the model loads, including
  `torchao: Module Not Found`, which is a silent code-path downgrade rather than an error.
  The same block also prints packaging, psutil and aiohttp; those are left out — true, and never
  the explanation for anything.
- `gpu.compute_capability` — `GPU 0 Compute Capability: 9.0`, the number a kernel-selection
  diff has to agree on before anything else can be compared.

Fact keys 262 → 274; tests 170 → 171; `audit` over the corpus still 87 log(s), 0 shapes worth a
look.

## 0.4.72 — 2026-11-23

`show` now says when the problem is cadence rather than configuration.

A hang log answers `show` with a dozen ordinary startup facts — KV cache size, capture time,
compile totals — and then repeats one watchdog line. That is the wrong answer to the question the
run actually failed at, and nothing pointed at the command that reads it. Adding one line on
**stderr** (so `servediff show log | grep` stays pipeable):

```
note: this log repeats one message 6 times at a 60.000 s cadence — a watchdog re-logging one
wait, not 6 failures; `servediff repeats <log>` says when that wait began and whether anything
printed inside it
```

A log that repeats nothing gets no pointer; a log with several regular groups names the most
frequent one. Tests 169 → 170, one of them asserts the pointer is not on stdout.

Also: PyPI's sidebar now links the notes site (`Notes = https://apex-mochen.github.io/servediff/`),
so the install page and the nine articles are one click apart instead of two.

## 0.4.71 — 2026-11-23

`servediff demo` now shows both claims, not just the first one.

A new install has no logs, so the first command has to carry the argument by itself. It already
diffed two excerpt logs from vllm#47047 (same flags, `VLLM_CUTLASS` → `MARLIN`). It now also runs
`repeats` on two more logs that ship inside the package, because the `No available shared memory
broadcast block found in 60 seconds` warning reads identically in both and means opposite things:

- `vllm-2828-a2-two-waits.log` — A2, TP2: the wait began at `09:02:37`, graph capture was still
  running, and the wait outlived it by five minutes with nothing printed inside it
- `vllm-45198-jit-then-wait.log` — the wait began at `22:01:40`, the same second
  `Triton kernel JIT compilation during inference: _zero_kv_blocks_kernel` was printed

Both blocks end with the arithmetic that separates them: the warning's timestamp is the *end* of
the wait, so the start is that timestamp minus the period named in the message.

No rule or fact changes; two example logs added (~3 KB), README's demo paragraph explains the
second half.

## 0.4.70 — 2026-11-23

A second entry point, for the other error people paste: `CUDA out of memory`.

The line carries three numbers, and the whole question is which pool `free` belongs to:

- **wording 1** — `Process N has M GiB memory in use` — `free` is the *device*, and `free + holder
  ≈ total` means the card is fully occupied by another job. `--gpu-memory-utilization` divides
  memory that has to be free first, so no flag helps. This is forum topic 1234.
- **wording 2** — `memory allocation failed with OOM ... (free: ..., total: ...)` — `free` is
  *PyTorch's caching allocator*, not the card. Our corpus has 39 of these in 2.318 s with 20 MiB
  requested and 9.19 MiB free, while `gpu.free_gib` was 78.46: the pool was sized by
  `gpu_memory_utilization × total` and was full, with a three-quarters-empty card.

New writeup `writeups/cuda-oom-holder.md` (the arithmetic, both wordings, how to find the holder,
what `CUDA_VISIBLE_DEVICES` renumbering does to "GPU 3", and when the card has already been
partitioned). The 60-second broadcast article now links to it, since the two messages are forever
confused. Site 9 → 10 files, both pages in the sitemap.

No rule changes, no fact changes: this release is the article plus its cross-link.

## 0.4.69 — 2026-11-23

A wait line marks the *end* of the wait, and `repeats` now does that subtraction.

**The timestamp arithmetic, in the tool.** `No available shared memory broadcast block found in
60 seconds` is printed after the watchdog has already waited 60 s, so `09:03:37` means the wait
began at `09:02:37`. `repeats` reads the period out of the message and prints where the wait
actually started, then whether anything else was printed inside the window:

```
6x regular  09:03:37 -> 09:08:37  span 300 s
   this line is the end of a 60 s wait: the first one began at 09:02:37
     0 other line(s) inside that span: nothing moved during the wait
```

That is the check that separates the three published readings by arithmetic instead of opinion. On
a TP2 A2 box the wait begins 3 s before graph capture ends and then outlives it by five minutes
with nothing inside. On vllm#45198 it begins at `22:01:40` — the same second
`Triton kernel JIT compilation during inference: _zero_kv_blocks_kernel` is printed. One is a
stuck reader, the other *is* compilation; both look identical without the subtraction.

- new fact `kernel.jit_during_inference` — a Triton kernel that compiled during inference rather
  than warmup, which is what the request actually paid for (vllm#45198)
- `repeats` also reports how many other lines printed inside a regular wait's span, and how many
  distinct shapes they were: "something was moving" vs "nothing moved during the wait"
- new fixture `vllm-issue-2026-06/qwen3.5-tp2-jit-then-wait.log` (issue 45198), four lines,
  verbatim with the reporter's own instrumentation left out, PROVENANCE says why
- writeup `vllm-shm-broadcast-60-seconds.md`: new section "The timestamp is the end of the wait,
  not its start", with both cases side by side

Fact keys 261 → 262; tests 168 → 169; corpus 86 → 87 logs.

## 0.4.68 — 2026-11-23

The "显存足够却启动不了" thread answers itself once the holder is read, and the six-line
broadcast wait gets its arithmetic written down.

**The OOM line names the process holding the card.** The older wording
(`Tried to allocate 768.00 MiB. GPU 3 has a total capacity of 95.00 GiB of which 600.06 MiB is
free. Process 3617065 has 94.41 GiB memory in use`) is now read as
`torch.oom_device=3`, `torch.oom_requested_mib=768.0`, `torch.oom_free_mib=600.06`,
`torch.oom_total_gib=95.0` and `torch.oom_holder_gib=94.41`. The pid is deliberately dropped:
it differs every run and would appear in every diff.

Seven more keys from vLLM 0.9.0 and a TP2 A2 box:

- `plugins.loaded` — the plugin group this build still loads, the only record of code that runs
  before the model does
- `model.tasks` — a five-task checkpoint served with no task flag, defaulting to `generate`
- `sampling.topk_topp_fallback` — FlashInfer missing, sampling moved to the torch kernel
- `engine.platform` now also matches the `__init__.py` module, which is where 0.9.x prints it
- the engine core's worker wrapper (`Exception: WorkerProc initialization failed`) is recorded
  as a startup error instead of being lost, without turning the oneDNN fallback sentence into a
  fake failure (`Exception: could not create a primitive descriptor` stayed a warning)

**`audit` on a directory reads the whole directory.** It globbed one level, so
`servediff audit tests/fixtures` audited 21 of the 86 logs and printed "21 log(s)" as if that
were the corpus — the same silent-scope failure the tool is built to catch, in the tool itself.
It now uses `rglob`. Running it on the full corpus immediately exposed one unruled shape family
the later batches had: the page-cache prefetch trio (`started` / `percentage` / `finished`),
which a `RayWorkerWrapper` prefix had kept out of view. Both are now covered, and a test holds
the printed count equal to what an `rglob` finds.

Two new fixtures: `vllm-forum-2025-07/qwen-vram-enough-but-oom.log` (topic 1234) and
`vllm-forum-2026-07/a2-stuck-on-shm-broadcast.log` (topic 2828). The broadcast writeup gains a
second worked example computed from that 24-line log: the wait starts 3 s before capture ends
and outlives it by five minutes, six lines 60.000 s apart. Fact keys 253 → 261; tests 163 → 168.

## 0.4.67 — 2026-11-23

A multi-node run says, once, in a warning, that the async scheduler it was asked for is not
running. That and ten other facts from three builds (0.10.1.dev1, 0.21.0, a Ray cluster).

**`--async-scheduling` under Ray.** `Async scheduling will be disabled because it is not
supported with the `ray` distributed executor backend` is a WARNING; the flag stays in the
printed args. `scheduler.async` still records the state, and the reason is now its own key,
`scheduler.async_reason=ray executor`, so a diff can tell "the flag was off" from "the backend
refused". Same run: `parallel.tp_exceeds_gpus=tp 16 > gpus 4`, `ray.env_prefixes` (which
variables Ray carries to the workers), and `startup.error=ray.exceptions.RayTaskError(KeyError)`.

Ten new fact keys:

- **Ray / multi-node:** `scheduler.async_reason`, `parallel.tp_exceeds_gpus`, `ray.env_prefixes`
- **Heterogeneous attention heads (vLLM 0.21.0, Gemma-4-31B-IT-NVFP4, TP2 over 2×16 GB):**
  `backend.attention_forced` (`TRITON_ATTN (head_dim 256 vs 512)`), `attn.backend`,
  `quant.checkpoint_format` (ModelOpt NVFP4, experimental)
- **Multimodal model run text-only:** `mm.text_only`, `mm.chunked_input`
- **Older builds:** `engine.platform`, `weights.format`, `weights.download_s`

`parallel.roles` no longer requires a PCP rank: the oldest log in the corpus (vLLM
0.10.1.dev1) prints `DP, PP, TP, EP` and nothing else, and the rule now takes whatever rank
list follows. A pasted copy whose ANSI escape lost its ESC byte (`e[36m` left behind) is
stripped too, so those lines classify as what they are instead of as unread shapes.

Three new fixtures — `vllm-forum-2026-04/ray-4-node-random-fail.log`,
`vllm-forum-2026-05/gemma4-nvfp4-hetero-heads.log`,
`vllm-forum-2025-09/mistral-7b-huge-batched-tokens.log` — each with a `PROVENANCE.md`.
New writeup `writeups/ray-async-scheduling.md`. Fact keys 242 → 253; tests 160 → 163; every
fixture still audits to zero shapes worth a look.

## 0.4.66 — 2026-11-23

Two pasted logs, two silent losses: quotes last time, **dashes** this time. Plus eleven facts
from a pre-Ampere card and an eager-mode run.

**Hyphens become en dashes.** Hugging Face writes a cache directory as
`models--Qwen--Qwen3.6-35B-A3B-FP8`; a forum or word processor turns the `--` into an en dash,
so a pasted `arg.model` reads `models–Qwen–…`. The fact was still read, with a value that
could never equal the server's own path: a diff against the box reported a model change that
never happened. `parse` and `audit` now also fold U+2013 → `--` and U+2014 → `---`, in the
same table that folds the curly quotes. Before: `arg.model=…/models–Qwen–Qwen3.6-35B-A3B-FP8`;
after: `…/models--Qwen--Qwen3.6-35B-A3B-FP8`.

**A local rank is not a port.** `Starting vLLM API server 0 on http://0.0.0.0:8000` prints the
rank inside a data-parallel group first. Read as a port it produced `server.port: 0` on a TP4
box, differing from every other log for no reason; the rule now keeps only `server.url`.

Eleven new fact keys, all settings no flag records:

- **Hybrid cache rewriting (vLLM 0.14, Nemotron-3-Nano-30B-A3B, TP4 over V100):**
  `kv_cache.mamba_ssm_dtype` (the engine moves it to float16 itself), `kv_cache.padding_layers`
  and `kv_cache.padding_waste` (one padding layer, 4.35% of the cache), `moe.config`
  (default because no config file exists), `parallel.dp_sync_nccl` (off under async scheduling).
- **A card without bfloat16:** `gpu.bf16_support`, `model.cast=torch.bfloat16 -> torch.float16`,
  `backend.flash_attn_blocked=FA2 needs sm80+`, and `backend.allreduce_reason` — custom
  allreduce naming why it switched itself off (PCIe-only box).
- **Eager mode and kernel fallbacks (vLLM 0.19, Qwen3.6-35B-A3B-FP8):**
  `quant.kernel.config=default W8A8 (no config file)`, plus `chat.template_warmup` and
  `server.url`.

Two rules stopped requiring text newer builds no longer print: `parallel.roles` accepts any
rank list (this build has no EPLB rank), and `gdn.prefill_kernel`'s `(requested=…,
head_k_dim=…)` suffix is optional. Both were silently dropping facts.

New corpus `tests/fixtures/vllm-forum-2026-01/` and `…-2026-04/`, each with a `PROVENANCE.md`
stating what was extracted and left typographic. New writeup `writeups/paste-typography.md`.
Fact keys 222 → 242; tests 155 → 160; every fixture still audits to zero shapes worth a look.

## 0.4.65 — 2026-11-23

Speculative decoding overrides three settings you passed and says so only in warnings; a log
pasted through a forum lost two facts to typographic quotes.

Nine facts from one forum paste (vLLM 0.28.0, Qwen3.5-27B-NVFP4 served plain and again with
`--speculative-config method=mtp num_speculative_tokens=2`): `sched.max_num_scheduled_tokens`
(the budget reset to 2048 while `--max-num-batched-tokens` stays in the args),
`sampling.ignored_with_spec` (`min_p, logit_bias`), `spec.mtp_forward_repeat`,
`spec.mtp_weights_shared`, `quant.kernel.nvfp4`, `quant.kernel.selected`,
`compile.deep_gemm`, `compile.deep_gemm_feature`, `gdn.prefill_kernel`, `gdn.decode_kernel`,
`mm.prefix_attention`. A flag names a format (`--quantization compressed-tensors`), never the
kernel; the kernel is where a throughput delta between two runs on one checkpoint comes from.

**Pasted logs keep their quotes.** The forum serves `Mamba cache mode is set to ‘align’` and
`won’t`; the server writes ASCII and every rule is ASCII, so a copy from a browser, a chat
client or a word processor silently lost the mamba cache mode and the spec warning. `parse`
and `audit` now curl U+2018/U+2019/U+201C/U+201D back before matching. Before: 2 shapes
unread on that fixture; after: 0.

Also: new corpus `tests/fixtures/vllm-forum-2026-09/` with a `PROVENANCE.md` that states how
the forum's paragraph-per-line rendering became a file (it has no `<pre>` block, so this is
extraction, not a verbatim block), a writeup
`writeups/spec-decode-overrides.md`, and the fact-key inventory is now **222 keys**.

## 0.4.64 — 2026-11-22

CPU serving was only half readable: the log pins each worker to a core list and lets
`OMP_NUM_THREADS` from outside override the torch cut, and neither value is in the flags.

Fifteen facts read from a real 0.34 CPU log — `cpu.threads_bind_mode`, `cpu.core_ids`,
`cpu.omp_threads`, `cpu.set_num_threads`, `cpu.numa_auto`, `cpu.numa_membind`,
`cpu.numa_migrate`, `kv_cache.attention_block_size`, `kv_cache.mamba_page_pad`,
`kv_cache.cpu_node`, `kv_cache.lcm_block_sizes`, `mm.pixel_cap`, `mm.encoder_cache`,
`compile.inductor_disabled`, `api.chat_template_order`. The interesting ones are the changes
no flag records: the hybrid cache pads its mamba page by 20.75% so both pages come out equal
(which is why the KV numbers do not match what you configured), the CPU layer allocates
`4.0/30.69 GiB` per node, and the backend refuses to move torch threads once affinity is
bound.

**The audit was hiding four of them.** Its "prose" rule — a capitalised English sentence
ending in a period — was written for text written around a pasted log, and it matched engine
messages too: `Padding mamba page size by 20.75% …`, `Encoder cache will be initialized …`,
`Explicitly set (4.0/30.69) GiB …` and `kv lcm block sizes 640` were counted as noise. So the
tool reported `nothing worth a look` while settings sat unread underneath. The rule now only
applies to lines with no log preamble, and `test_a_sentence_a_log_actually_printed_is_never_prose_noise`
pins it. Any earlier "zero remainder" claim was made with that rule in force.

That makes the full CPU log the fifth corpus at zero: `audit … --max-shapes 0` exits 0, CI
gates it, and `PROVENANCE.md` lists what is left (121 noise lines, including the 49-occurrence
broadcast wait, kept as noise with its own reasoned rule).

Also: the README now points at the rendered notes site
(<https://apex-mochen.github.io/servediff/>) and lists `writeups/same-box-twice.md`.

On Windows, `servediff show server.log | head` still printed a traceback, because the closed
pipe does not arrive as `BrokenPipeError` there — the CRT reports `OSError: [Errno 22]
Invalid argument` (Winsock uses winerror 109/232). The guard now recognises that form as well,
so the same command is silent on both platforms: before, 12 lines of traceback and
`OSError: [Errno 22] Invalid argument`; after, zero lines on stderr. The regression test
that used `head` now uses a second Python as the reader (Windows ships no `head`) and a unit
test pins the Windows form directly.

## 0.4.63 — 2026-11-22

The hybrid cache on 0.34 was invisible, and a lost compile-cache hit had no line.

vLLM 0.34 builds with Mamba/GDN layers stop printing `Available KV cache memory` and print
`kv cache group sizes (640, 640, 640, 640)` instead, plus the mamba cache mode the model chose
by default. Neither matched a rule, so on Qwen3.5 the cache layout the engine settled on was
absent from `show` entirely. Two more startup steps only exist on these runs: the multimodal
warmup (`Multi-modal warmup completed in …s`) and a torch.compile cache artifact that failed
to load and forced the whole compile again (`[backbone] Compiling model again due to a load
failure from …`) — that one is measured startup time that no flag records.

Four facts (`kv_cache.group_sizes`, `kv_cache.mamba_mode`, `mm.warmup_s`,
`compile.recompile_reason`), and two progress-bar shapes (`Download complete:`,
`Reconstruction complete:`) reclassified as noise rather than as lines worth a look. The full
CPU log's unread remainder drops from 22 to 9, and the pinned count plus `PROVENANCE.md` say
which nine.

Also: the three `writeups/` links in the README were relative, so on the PyPI project page
they pointed at `pypi.org/...` and went nowhere. They are absolute now.

## 0.4.62 — 2026-11-22

A CPU log came out as `startup ready` with no backend facts, and a silent fallback stayed invisible.

On vLLM 0.34 with `device_config=cpu` the GPU-shaped `Using X attention backend` line is never
printed. The choices are printed one backend at a time instead — the vision tower, the multimodal
encoder, the CPU linear kernel, the GDN prefill kernel — and none of them matched a rule. A diff
of two CPU runs therefore reported "no backend change" whether or not one of them had asked for
something else: `<absent>` on both sides, which is the failure mode this tool exists to prevent.

Five rules now read those lines: `backend.attention_vit` (with `(default)` when no flag won),
`backend.attention_mm`, `backend.linear`, `backend.gdn_prefill`, and
`backend.linear_fallback`. The last one is the reason to look at CPU startup logs at all: oneDNN
refuses a primitive descriptor, torch takes over, and nothing on the command line records it.

The fixture is a full 173-line log from vllm#67598 (0.34.0rc1.dev398, one AVX2 CPU, Qwen3.5-0.8B
W8A8), kept in `tests/fixtures/full-cpu-2026-11/` because 22 shapes in it are still unread —
the first GPU-less log in the corpus, and the remainder is written down next to it instead of
being claimed away.

## 0.4.61 — 2026-11-13

Sixteen waiting lines were one wait, and nine characters of readme said "16 stalls".

`servediff show` compresses a log into facts, so the copies disappear: 39 allocator OOM
lines and 16 ring-buffer warnings each came out as one fact, and the only signal that
separated one wait from sixteen failures — the gap between them — was thrown away.
`servediff repeats` keeps it. It groups each message shape that appears more than once,
reads the clock on every copy, and reports the span, the median gap and whether the gaps
hold: `regular` (even gaps, at least a second apart — a watchdog re-logging one wait),
`burst` (millisecond gaps — a retry loop), `untimed` (no clock on the line).

Measured on the log from vllm#54559: 39 OOM lines are one burst inside 2.32 s (median
gap 0.010 s), and the 16 `No available shared memory broadcast block found` lines are one
959 s wait re-logged every 60 s. `--min`, `--top`, `--markdown` and `--json` behave like
`audit`'s, and both readings are spelled out in the README.

Two log shapes had to survive before the counts were honest. A tqdm progress bar carries
no newline of its own, so a captured log put the next process's warning on the same
physical line — that line's shape then held the bar's digits and the warning counted as
15, not 16. And two ranks interleave their timestamps, so one line arrives out of order:
read as a sequence it produced `max gap 86399.589 s` (a negative delta wrapped to a day),
read as a set of moments it is 0.9 s. Both are now the first two tests of the command.

## 0.4.60 — 2026-11-12

The other engine says the kv-cache dtype differently, and half of that was unread.

`kv_cache.dtype` was produced by one phrasing only: SGLang's `KV Cache is allocated.
dtype: fp8, size: …`. vLLM writes the same decision as `Using fp8 data type to store kv
cache.`, which no rule read, so it fell out of `show` entirely and showed up in `audit`
as a shape worth a look. Found by running `audit` over a log pasted into vllm#66423 — a
segfault report with no fixture of ours anywhere near it — not by reading the code.

The new rule feeds the same fact name, so a log carrying both phrasings still reads as
one value instead of two rows that disagree about nothing.

## 0.4.59 — 2026-11-12

A typo names the command you typed, not a file you never wrote.

Running from a wheel but inside a module (`python -m servediff dif`) answered the
mistyped subcommand with `usage: __main__.py [-h] [--version] ...`. The console script
said `servediff` all along, so the same mistake produced two different first lines, and
the one a reader hit named a file that only exists inside an installed package. The parser
now declares its own program name; both invocations print `usage: servediff`.

The Chinese README is brought forward to what the tool now does: `--only MATCH` (prefix or
substring, `_`/`-` ignored, values never searched), the dropped-flag note with a real
`note:` line captured from a published wheel on someone else's log, the corrected
`53 → 3 + 50 hidden` split, and that a narrowed comparison only explains its own facts.

`125 tests` (one new: a mistyped subcommand exits 2 and its first stderr line is
`usage: servediff`, with no `__main__.py` anywhere in it). `21/21`, `53 / 14`,
`35 / 33` and all five corpora at 0 shapes are unchanged.

## 0.4.58 — 2026-11-12

`--only` no longer explains the facts it excluded.

A narrowed comparison is a statement about one decision, and the absence notes kept
explaining the decisions next to it. With `--only attention` over the 0.28.0-vs-0.30.0
pair, two paragraphs about the all-reduce path the reader had just said they were not
comparing ran above the single-row table:

```text
note: baseline: no all-reduce facts: world_size=1, so there is no all-reduce group and no backend
note: candidate: no all-reduce facts: world_size=1, so there is no all-reduce group and no backend
```

The notes now name the fact they are about, and a note is kept only if `--only` kept that
fact. Asking for the same pair with `--only backend.` prints them unchanged — the point of
the note is still to keep "absent" from being read as "off", it just no longer volunteers
that about a fact outside the gate.

`124 tests` (one new: the all-reduce note disappears under `--only attention`, returns
under `--only backend.`, and the excluded-fact row never appears either way). `21/21`,
`53 / 14`, `35 / 33`, the demo's `2 difference(s)` and all five corpora at 0 shapes are
unchanged.

## 0.4.57 — 2026-11-12

The word in the log now finds the fact it means.

`diff --only` matched fact names by prefix only. The word a reader has is never the fact
name — it is what the log printed. `#63705` is the case in point: someone grepped two
startup logs, found the TP all-reduce chain go from `['PYNCCL']` to
`['FLASHINFER', 'PYNCCL']`, and that chain is `backend.allreduce_dispatch` here. Typing
`--only all_reduce` — the log's own spelling — matched nothing and exited 2:

```text
$ servediff diff a.log b.log --only all_reduce
diff: --only all_reduce matched no fact in either input; a gate on nothing is not a pass
```

A pattern now matches as a prefix or anywhere inside the fact name, with `_` and `-` ignored
on both sides, so `all_reduce`, `all-reduce` and `allreduce` reach `backend.allreduce_*`. The
loud exit stays where it was and the boundary stays where it was: a pattern that matches
nothing still exits 2, and values are still never searched — `--only FLASHINFER` matches
nothing, because a gate pinned to a value moves with whatever that week's version printed.

`123 tests` (one new: three spellings each find the dispatch row and leave
`parallel.world_size` out, plus the two boundaries above). `21/21`, `53 / 14`, `35 / 33`,
the demo's `2 difference(s)` and all five corpora at 0 shapes are unchanged.

## 0.4.56 — 2026-11-12

The dropped flag becomes a sentence, not a row.

0.4.55 read `argument 'x' currently has no effect in Rust frontend, ignoring` as
`frontend.ignored_arg`. Reading it is not the same as saying it: in a table of 18 facts a
flag the server accepted and then dropped looks like trivia, and a pair of runs that both
dropped the same flag compares as `same` with exit 0 — every fact identical, neither run
configured the way its command line reads.

`show` now ends with the consequence when the fact is there:

```text
note: the command line named 2 flag(s) this frontend accepted and never applied (structured_outputs_config, enable_auto_tool_choice): the run is not configured the way the command reads
```

and `diff` carries the same note per side, including when the baseline is a recorded facts
file — a dropped flag is the one thing a facts file can still report, because the frontend's
own line was written down. Two runs that both dropped it still exit 0 (nothing differs),
but the note says why that pair is not the comparison you meant to make.

No new fact names, no change to counts: `53 / 14`, `35 / 33`, the demo's `2 difference(s)`
and all five corpora at 0 shapes are unchanged, `tools/verify_readme_exit_codes.py` is 21/21,
and 122 tests (two new: the note appears on `show`, appears on both sides of `diff`, and a
log without a dropped flag gains no note so the documented output holds).

## 0.4.55 — 2026-11-11

Two facts for the flags a run did not get.

`vllm#58580` looked like noise when the sweep counted it: 143 lines, 55 facts, mostly a
`collect_env` block. What `audit` still flagged as unread, twice, was this:

```text
(RustFrontend pid=1185) WARNING [unsupported.rs:67] argument 'structured_outputs_config'
currently has no effect in Rust frontend, ignoring
(RustFrontend pid=1185) WARNING [unsupported.rs:67] argument 'enable_auto_tool_choice'
currently has no effect in Rust frontend, ignoring
```

The run passes both flags, the frontend accepts both, and neither reaches anything. Same
command line, different server — and no amount of log-diffing by hand makes that a row
instead of a footnote in 80 lines. It is now `frontend.ignored_arg`, one value per dropped
flag, so a pair of runs that differ only in which flags survived shows up as a list that
changed.

The same block carries `Overriding draft model max model len from 262144 to 131072`, i.e. the
speculative path halved the draft model's context with no flag recording it: `spec.draft_max_len`
keeps both ends (`262144 -> 131072`) so an upgrade that moves the bound is visible.

The audit of that block went from 89 unread lines / 63 worth a look to 86 / 60, and its facts
from 55 to 57.

120 tests (one new, pinning these lines verbatim, including the order of the two ignored
args). Unchanged and re-verified: 21/21 documented exit codes, five corpora at
`0 with shapes worth a look`, README's worked examples still 53 / 14 and 35 / 33, the same
numbers reproduced from the 0.4.54 wheel in a fresh venv.

## 0.4.54 — 2026-11-11

Docs correction, no code change.

0.4.53 added four facts from the allocator's failure line and then overstated what they meant.
Re-read against `vllm#54559`'s own timestamps, all 39 of those lines fall inside the
**model-loading window** (11:33:26-11:33:28) and stop the moment both ranks report
`Model loading took 63.31 GiB memory` at 11:33:29. On a 79.14 GiB card that is a near-miss
(20 MiB asked with 9.19 MiB free), not the cause. What keeps failing is the `shm_broadcast`
message: first at 11:34:32, then every 60 s through 11:50:31 — it outlives torch.compile
(40.21 s, finished 11:34:51) and graph capture (85 s, finished 11:37:55), so whatever hangs does
so after startup.

The 0.4.53 facts stay as they are; the CHANGELOG paragraph and the README paragraph that
pointed at this issue now say that instead.

## 0.4.53 — 2026-11-11

The allocator's own failure line, which no rule read.

`vllm#54559` dies with `No available shared memory broadcast block found in 60 seconds` and
carries **39** of these underneath, untouched:

```text
[rank0]:[W831 11:33:26.640394862 CUDACachingAllocator.cpp:3933] memory allocation failed with
OOM on device 0 while trying to allocate 20971520 bytes (free: 9633792, total: 84974239744).
```

Now four facts — `torch.oom_device`, `torch.oom_requested_mib`, `torch.oom_free_mib`,
`torch.oom_total_gib` — and where they sit becomes readable instead of guessed. All 39 fall
inside the **model-loading window**, 11:33:26 to 11:33:28, and stop the moment loading reports
`Model loading took 63.31 GiB memory` on both ranks: on a 79.14 GiB card the allocator came
within **9.19 MiB** of empty while asking for 20 MiB, then recovered. They are a near-miss, not
the cause — the `shm_broadcast` messages begin at 11:34:32 and keep coming every 60 s to
11:50:31, i.e. they outlive compile (11:34:51) and graph capture (11:37:55), so what hangs is
after startup, not memory. Facts that let a reader see that shape are worth having even when
they turn out to clear the run rather than convict it.

119 tests. Five corpora still `0 with shapes worth a look`; the README's worked examples are
unchanged (53 / 14 and 35 / 33), since no shipped log carries this line.

## 0.4.52 — 2026-11-11

The extensible KV cache's own accounting, and the number a "KV dropped 2.4x" claim is made of.

- **`kv_cache.suggested_gib` / `kv_cache.suggested_bytes`.** The engine prints
  ``Replace gpu_memory_utilization config with `--kv-cache-memory=3746943079` (3.49 GiB)``
  whenever the requested utilization cannot fit the context. That suggestion is the number the
  loss is measured against, and it was not read.
- **The after-warmup split.** `Memory after warmup: 8.66 GiB free, 0.78 GiB committed to the
  KV cache, 3.83 GiB kept for transient activations; 28 KV cache blocks (1.51 GiB) fit within
  the requested 26.75 GiB. Torch may reserve up to 28.65 GiB.` → `kv_cache.after_warmup_free_gib`,
  `kv_cache.committed_gib`, `kv_cache.transient_gib`, `kv_cache.blocks`, `kv_cache.blocks_gib`,
  `torch.may_reserve_gib`.
- **The shortfall that follows it.** `kv_cache.needed_gib`, `kv_cache.available_gib` and
  `model.max_len_fit` from the `ValueError: To serve at least one request with the model's max
  seq len …` line.
- Which makes a real report checkable: in
  [vllm#66193](https://github.com/vllm-project/vllm/issues/66193) the headline is "cutting KV
  2.4x", and that ratio is 3.49 GiB suggested / 1.46 GiB available = **2.39x**, with a 28.65 GiB
  reservation against a 26.75 GiB budget. All four numbers are now facts in a 4-line excerpt.

The same rule change is visible in the README's worked example, so its counts were re-measured
again rather than carried over: 0.28.0 vs 0.30.0 is now **53 differences / 14 identical**
(`--skip-absent`: 3 rows, 50 hidden) and the forced-V1 pair is **35 / 33**. 118 tests.

## 0.4.51 — 2026-11-11

The one line a real OOM report turns on was unreadable.

- **The post-capture breakdown is now a fact.** After the graph capture, vLLM prints
  `Actual usage is 45.05 GiB for consumed memory (weights + non-torch), 14.16 GiB for peak
  activation, and 10.32 GiB for CUDAGraph memory`. Only the head of that sentence was read
  (`gpu.free_gib` / `gpu.total_gib` / `gpu.utilization` / `gpu.budget_gib`); the triple that
  answers "where did the memory go" was dropped. It is now `gpu.consumed_gib`,
  `gpu.peak_activation_gib` and `gpu.cudagraph_gib`.
- **`gpu.utilization_effective` no longer needs the trailing clause.** The rule required
  `... without CUDA graph memory profiling`; real reports paste the same sentence as
  `--gpu-memory-utilization=0.9700 is equivalent to --gpu-memory-utilization=0.7715` and stop
  there, so the fact was absent exactly in the logs people hand over.
- Found by reading a real case, not by guessing: in
  [vllm#65907](https://github.com/vllm-project/vllm/issues/65907) the reporter's whole
  argument is that consumed memory stayed ~45 GiB while peak activation fell 14.16 → 8.97 GiB.
  `servediff diff` over his two verbatim excerpts now prints exactly that:

  ```text
  gpu.consumed_gib                 45.05   44.82   changed  -0.5%
  gpu.peak_activation_gib          14.16    8.97   changed  -36.7%
  gpu.cudagraph_gib                10.32     3.7   changed  -64.1%
  ```

- Numbers in the README's worked example moved with the parser, so they were re-measured and
  corrected rather than left standing: 0.28.0 vs 0.30.0 is now **51 differences / 14 identical**
  (`--skip-absent`: 3 rows, 48 hidden) and the forced-V1 pair is **33 / 33**. The unverifiable
  "19 facts / 5 190 B" claim in the Chinese README was replaced with a measured range.

117 tests (one new: both real line shapes from that report).

## 0.4.50 — 2026-11-11

Positioning only — no code change.

- **The first sentence now leads with one log.** It said *"Same flags, different server"* and
  put the two-log comparison at the door. Measured over 200 open issues (100 vLLM + 100
  SGLang, created 2026-10-20..11-05, all comments read): exactly **1** thread carried anything
  `servediff` can read, and **0** carried two. The old pitch was asking for raw material that
  issue threads do not contain. It now reads *"One startup log in — the settings the server
  actually ran with come out"*, and the two-log comparison is the second step.
- `demo` stays as the zero-input first command (it needs no log at all); a second block,
  **"Already have a log? `servediff show serve.log`"**, now sits right under it, because that
  is the path a reader with a crash behind them actually takes. `show` prints each fact with
  its line number and exits `0` as soon as it read something — verified against the shipped
  0.33.0 log.
- `README.zh-CN.md` carries the same change.

## 0.4.49 — 2026-11-11

Documentation only — no code change.

- **The exit-code table now matches what 0.4.46–0.4.48 actually do.** `diff` gained *one log
  with nothing to compare it to* as a documented `2`; `env` gained *or the path cannot be
  written*; `capture` now says `2` when the paths cannot be written or the command cannot
  start. Each of these was already true in the code, and the table was the part that lied.
- **`README.zh-CN.md` gets the one-log path.** Its "use your own log" section started with
  `capture`, which needs a server and a Ctrl-C, while most people already have a log. It now
  starts with `show serve.log` / `show serve.log --json > facts.json` /
  `diff today.log --against facts.json`, and leaves `capture` to *the failure is ahead of you*.
  Re-ran all three against the published 0.4.48 wheel: exit `0`, `0 difference(s); 56 fact(s)
  identical`.

## 0.4.48 — 2026-11-10

- **A single-byte console no longer turns a report into exit 1.** `audit` echoes log lines
  back, and a real log carries a box-drawing banner (`██▄█▀`). On a default Windows console
  (cp1252) that raised `UnicodeEncodeError`, so `audit` exited `1` — and `1` means *the inputs
  differ*, so a gate read a console limitation as a config change. `show --json` failed the
  same way; because it writes the payload in one `print()`, nothing was written and
  `--against` then read an empty facts file. Output streams now keep the console's own code
  page and let a character it cannot encode come out as `?`.
- Found by the step added in 0.4.47: CI's Windows job runs under that code page, and
  `python tools/verify_readme_exit_codes.py` reported `19/21` there while both Linux jobs
  reported `21/21`. The failure was real, not a test artifact.
- 2 new tests (116 total): `audit` and `show --json` under `PYTHONIOENCODING=cp1252` exit `0`
  with no `UnicodeEncodeError` and no `Traceback`.

## 0.4.47 — 2026-11-10

- **`env -o` and `capture -o` say so when the path cannot be written.** Pointing `-o` at a
  directory that does not exist raised `FileNotFoundError` and exited `1` — and `1` is documented
  as "the inputs differ", so a script that branches on it reads a typo as a finding. Both now
  exit `2` with `cannot write <path>`. For `capture` this happens before anything is launched,
  so it is a usage error, never the server's exit code.
- **`tools/verify_readme_exit_codes.py`: the command table is now a test.** It builds its own
  fixtures, runs every documented case (21 of them), and fails on a code that does not match or
  on a traceback. CI runs it on each matrix job, so a README promise cannot drift from the tool
  without going red.
- 2 new tests (114 total): `env` and `capture` to an unwritable path are exit `2`, with no
  `Traceback`.

## 0.4.46 — 2026-11-10

- **`diff` with one log now explains itself instead of raising.** `candidate` became optional
  in 0.4.42 only because `--against` supplies the other side; without either, the code reached
  `_read_log(None)` and a newcomer's very first command — `servediff diff startup.log` — died
  with `AttributeError: 'NoneType' object has no attribute 'read_text'`. It now exits `2` with
  `diff: one log is not a comparison -- pass the second log, or '--against facts.json'
  (record it from a log you kept: servediff show --json <log> > facts.json)`.
- Found by walking the README as a stranger would: fresh venv, `pip install servediff`,
  `servediff demo`, then "diff the one log I have". No code path was changed beyond that
  branch; every other exit code stays as it was.
- 1 new test (112 total): one log without `--against` is exit `2`, names both routes, and
  asserts `Traceback` is not in stderr.

## 0.4.45 — 2026-11-10

- **`diff <dir> --against facts.json`: gate a whole corpus in one command.** A CI matrix
  writes one startup log per row; six diffs bury the answer in six tables. Give `diff` the
  directory and it prints one line per log — `N difference(s); M fact(s) identical` plus the
  first changed keys — then one summary line and one exit code. `--only` / `--skip-absent` /
  `--all` narrow every row the same way they narrow a single diff, and `--json` returns
  `{against, facts, logs, logs_differing}` for a machine.
- **A blank or truncated artifact in that directory exits 2.** A log nothing can be read
  from is a broken gate input, not a run that happened to match; an empty directory is
  likewise "no *.log inside", not a clean pass.
- 4 new tests (111 total): per-log lines and one exit code, `--only` narrowing the corpus,
  the empty directory, and the blank artifact.

## 0.4.44 — 2026-11-10

- **`diff --only PREFIX` (repeatable): a gate on the decisions you own.** Package pins move
  every week; the startup decisions — attention/MoE backend, CUDA graph mode, parallelism —
  are what a version bump actually changes. `--only backend. --only cuda_graph.` compares
  those rows and nothing else, and the note says how many facts were left out, so a narrowed
  pass cannot be read as a clean one.
- **A prefix that matches nothing exits 2, not 0.** `--only cuda_graph` on two logs that
  never printed it is a green gate watching nothing — the same failure as an unreadable
  baseline. It says so and fails.
- `--only` is rejected on `servediff env` snapshots rather than silently ignored: those are
  compared whole today.
- 4 new tests (107 total): narrowing, two prefixes at once, the no-match exit 2, and the
  snapshot rejection.

## 0.4.43 — 2026-11-10

- **Corrected the size figures in the 0.4.42 example.** 0.4.42 said a facts file is 4 KB
  "versus 40 KB for the log"; measured on the fixtures it is **5 190 B for 19 facts**, the
  ROCm collect_env paste it came from is **9 904 B**, and a full startup log is 33–42 KB.
  The point still holds — you commit kilobytes instead of the log — but the numbers in
  the README and the release body now match what `Get-Item` reports.
- No code change: same behaviour, same 103 tests, same exit codes.

## 0.4.42 — 2026-11-10

- **`servediff diff <log> --against facts.json`: one log is enough to gate.** A CI job has
  today's startup log; nobody keeps the log from the run they accepted three months ago.
  `servediff show accepted.log --json > facts.json` records those facts in a few kilobytes
  (19 facts from a collect_env paste is 5.2 KB, against 9.7 KB for that excerpt and 33–42 KB
  for a full startup log), and `--against`
  diffs today's log against them with the same rows, the same `<absent>` semantics and the
  same exit codes as a two-log diff. Passing a second log alongside `--against` is a usage
  error rather than a guess.
- **`show --json` now also emits `values`, the lossless list.** `value` collapses to a
  scalar when a fact was printed once and to a list when it was printed several times, so
  reconstructing facts from it turns `cuda_graph.captures` (one list-valued fact) and
  `backend.attention` (two conflicting backends) into the same shape. `--against` reads
  `values`, so a round trip through the file cannot merge or split a fact; files written
  before 0.4.42 still load, with their `value` taken as one value.
- **Recorded facts cannot explain an absence.** In `--against` mode the baseline gets no
  absence note — a facts file has nothing to say about why a line was never written — and
  the output names how many facts the baseline holds, so a narrowed comparison is never
  read as a clean one. Unreadable or non-facts input exits `2`, not `1`: a broken baseline
  is a broken gate, not a finding.
- `--raw-args` is ignored when the baseline is a facts file, because a facts file records
  facts and not the resolved command line; the flag no longer invents `raw_arg.*` rows.
- 3 new tests (103 total) covering the round trip, the multi-valued fact, and the three
  new exit-2 cases.

## 0.4.41 — 2026-11-10

- **The second CI gate is now on the landing page.** `--baseline` shipped in 0.4.40 and was
  documented only inside the audit section; the README that PyPI renders now carries the
  two-step recipe near the top — write the baseline once on a log you have read, then gate
  with `--max-shapes 0` against it — with the exit codes spelled out: `1` a new shape, `2`
  the baseline or the log could not be read, which is a broken gate rather than a finding.
- No parser or CLI change: this release is the documentation the previous one needed, and
  the PyPI long description is this README.

## 0.4.40 — 2026-11-10

- **`audit --baseline` / `--write-baseline`: a gate that stays on.** `--write-baseline
  accepted.json` records the shapes a run saw; `--baseline accepted.json` reports and gates
  only on shapes that are not in it, printing how many were hidden (`N shape(s) in the
  baseline (M line(s)) were not counted`). Without this, `--max-shapes` goes red the first
  time the server prints a line no rule reads and stays red forever — a gate that is always
  red gets deleted, and then nothing is checked. A baseline turns it into the question that
  actually matters: did this upgrade make the server print configuration nothing reads.
- **A shape in the baseline is still reported as unread**, because it is: accepted once, not
  read. The single-log report says `nothing new: every shape worth a look is already in the
  baseline`, not `every unread line is noise`, which would misdescribe what happened — that
  wording is now pinned by a test.
- **`--json` gains two fields only when `--baseline` is given** (`baseline_shapes_not_counted`,
  `baseline_lines_not_counted`), so a pipeline reading the default document sees the same
  fields it saw in 0.4.38. `--max-shapes` and the exit code behave as before: `--json` still
  changes the format, not the code, and `--json` with `--markdown` is still exit 2.
- **A file that is not a baseline is exit 2**, not a traceback and not exit 1: an unreadable
  or malformed `--baseline` is an input error, the same rule the rest of `audit` follows.
- 100 tests.

## 0.4.39 — 2026-11-10

- **A third held-out batch, collected five days later, and what it found.** `created:>2026-11-05`
  across both repositories produced 28 usable pastes (3306 lines), now in
  `tests/fixtures/held-out-2026-11b/` with a `PROVENANCE.md`. The rules as of 0.4.38 reported
  17 of those 28 files: 191 shapes.
- **The `[conda]` package inventory.** 164 of the 191 shapes were one table — `collect_env`
  lists every installed wheel twice, once as `[pip3] name==version` and once as a `[conda]`
  row, and only the pip form was covered. Four reports carried ~41 rows each, and every row
  reads like a version fact. The `[conda]` table is the same inventory in another column
  format, so it gets the same policy; `[conda] Could not collect` was already there.
- **The rest of a pasted reproducer's expression lines**: an f-string (`f"…{…}"`), a tuple
  assignment with qualified names (`layer.output_size_per_partition, layer.input_size_per_partition
  = N, K`), the argument list that only closes the call above it, a one-line call chain
  (`tf.AutoModel…from_config(cfg).to(…).save_pretrained(path)`), a nested call closed twice
  (`asyncio.run(main(parser.parse_args()…))`), a bare name closing a parenthesis, a lone `[`,
  an annotation-only assignment, a `!= (…)` continuation, an `and … )` continuation, a tuple
  of layer specs ending the `for … in (` header, `async def`, and `finally:`.
- **One candidate measured and thrown away again.** `^\(.*\)$` — any parenthesized line —
  looked like it covered the reproducer's Path expression, and it ate 31 real vLLM log lines,
  every one a `(EngineCore pid=…)` entry. The adopted pattern requires the closing `)` to be
  followed by `.name`, which is how a value built in brackets gets used. The engine line is
  now a control in the tests.
- All five corpora report zero: 21 fixtures, 11 October pastes, 7 November pastes, these 28,
  and the 6 logs from a real machine. 96 tests.

## 0.4.38 — 2026-11-10

- **`servediff audit --json`.** The audit prints a table for a person; a CI gate should read
  fields. `--json` emits one object with, per log, `path`, `unread`, `shapes` and the
  `shapes_worth_a_look` entries (shape and count), plus `shapes_worth_a_look` (the number
  `--max-shapes` gates on) and `logs_with_shapes_worth_a_look`. The counts come from the
  same pass that renders the text, so the two cannot disagree.
- `--json` changes the format, not the exit code: `--max-shapes` still decides it. Passing
  `--json` with `--markdown` is an error (exit 2) rather than one silently winning.
- Internally the audit's summary now carries the full log label and shortens it only for the
  table, so the JSON path is the path that was read.
- 92 tests.

## 0.4.37 — 2026-11-10

- **A second held-out batch, a month later, to test whether the rules learned logs or learned
  October.** The same collection (`created:>2026-11-01`) produced twelve blocks; five were
  already in the October batch, so `tests/fixtures/held-out-2026-11/` holds the seven new
  ones (713 lines) with a `PROVENANCE.md`. Six were clean on the first try; the seventh had
  one line.
- **A conditional expression standing alone is not a log line.** `run(a) if a.cmd == "run"
  else compare(a)`, lifted out of a reproducer script, is the argument-list shape one form
  further out: it picks between two values, so it has no subject. The obvious pattern
  (contains `if` and `else`, ends in `)`) was measured and rejected — it also ate the warning
  *"Using FlashInfer all-reduce if available else falling back to the custom kernel (set
  VLLM_ALLREDUCE_USE_FLASHINFER=0 to disable)"*, which is a statement and belongs in the
  report. The adopted pattern requires the line to begin with a call; it matches one line
  across all four corpora.
- The old result is reproducible: `v0.4.36` reports one of seven on this directory.
- 89 tests.

## 0.4.36 — 2026-11-10

- **The held-out corpus ships, so the claim about it can be rerun instead of believed.**
  `tests/fixtures/held-out-2026-10/` holds the eleven pastes taken fresh from vLLM and SGLang
  issues in October 2026 (1291 lines), with a `PROVENANCE.md` naming each issue and stating
  what the sample does not prove. A test audits that directory and asserts
  `11 log(s): 0 with shapes worth a look`; checking out `v0.4.32` and rerunning the same
  command reports eight of eleven, which is the measurement 0.4.33 was about.
- Directory expansion stays non-recursive, so `audit tests/fixtures` still reports the
  twenty-one main-corpus logs and not thirty-two; the two corpora are audited separately.
- 87 tests.

## 0.4.35 — 2026-11-10

- **The last four "worth a look" lines in the held-out corpus are gone.** They were the same
  pasted script in a harder form: an argument list standing alone
  (`8, 96, 128, conv_kernel_size=4, num_spec=5`, `groups, KVCacheLayout.BLHNC`) and a
  continuation that closes parentheses without starting a statement
  (`…routing_input_mode=…), wp))(ml)`). A log line is a statement — it starts with a word and
  says something about it — so it never consists of arguments alone. Two patterns, both
  anchored on that: the line must be arguments only, and the first token must be a bare name.
  That anchor is what keeps `NVIDIA_DRIVER_CAPABILITIES=compute,utility` a fact.
- Eleven pastes taken fresh from vLLM and SGLang issues (October 2026): **0 with shapes worth
  a look.** 21 shipped fixtures: 0. Six real startup logs: 0.
- Measured before adopting: both patterns hit **zero** lines across the 21 fixtures and the six
  real startup logs, and the `NAME=value` counter-examples were checked explicitly. 86 tests.

## 0.4.34 — 2026-11-10

- **The log that caught the padded-colon bug is now fixture 21.** `vllm-66525` is a real
  `collect_env` pasted into a vLLM issue from a Debian 12 box with no GPU: unpadded version
  lines, `torch 2.10.0+cpu`, `HIP runtime version: N/A`, `GPU models and configuration: No
  CUDA`. Every fixture the repo carried was a hand copy that had kept the padded colons, so
  the 0.4.33 bug was invisible here; this one pins it down against the real form, and adds the
  first corpus entry from a machine with no accelerator at all.
- A test now asserts that log reads `env.torch`, `env.os`, `env.python` and `gpu.model`, and
  that it contributes nothing to "worth a look" — so a future rule regression fails in the
  suite instead of in someone's issue.
- Corpus count in README, README.zh-CN and the writeup: twenty → twenty-one. 85 tests.

## 0.4.33 — 2026-11-09

- **Every `collect_env` rule was pinned to the padded colon.** They were written
  `^PyTorch version\s+:\s*(\S+)`, which requires whitespace before the colon; ROCm's own dump
  aligns its colons with spaces, and vLLM's real `collect_env` prints
  `PyTorch version: 2.10.0+cpu` with none. So on the output every bug report actually pastes,
  `env.torch`, `env.torch_rocm`, `env.hip`, `env.os`, `env.python` and `engine.version` were
  never read. `\s*` on those rules; `env.allocator` reads `Caching allocator config`.
  Found by auditing eleven issue pastes taken fresh from vLLM and SGLang (created after
  2026-10-20) instead of the repo's own fixtures — 8 of 11 came back with lines worth a look.
  After this version: 2 of 11, both script pastes, four lines.
- Noise, same corpus: SGLang's macOS report (`MPS available`, `macOS Version`, `Apple
  Silicon`, `Metal Support`, `Unified Memory`, `CPU Cores (…)`, `GPU Cores`), a multi-GPU row
  written `GPU 0,1,2,3: NVIDIA H20`, a container runtime's `brand=…driver>=…` capability
  list, a topology-matrix header row, a reporter's parenthetical note, `Is XPU available`,
  `CPU:` / `Versions of relevant libraries:` headers with a colon, and a plugin source paste
  (`WTYPE = …`, `tensor.offset`, `{`, `sys.modules[…] = …`, `(Path(…) / "x").write_text(`).
- `MAX_MODEL_LEN=4096` inside a pasted script is still a fact (`env.var.MAX_MODEL_LEN`), and
  `GPU models and configuration: No CUDA` is still a fact (`gpu.model`): a script's code is
  noise, a script's call is not, and a CPU box saying it has no GPU is a statement.
- README numbers unchanged: 48/14, `--skip-absent` 3/14/45, 30/33, `--skip-absent` 22/33/8.
  The 20-log corpus and the six real startup logs still report `0 with shapes worth a look`.
  84 tests.

## 0.4.32 — 2026-11-09

- A reproducer script pasted into an issue body is no longer reported as lines worth a look.
  sglang#49123 pastes a 195-line benchmark harness, and the audit called 38 of its lines
  "worth a look" — every one of them the script: `def run_marlin(...)`,
  `handles[f] = safe_open(...)`, a module docstring. Those shapes (definition, tuple unpack,
  subscripted assignment, bare `print(...)`, a line ending in a comma, the prose between two
  `"""`) are noise now: that paste goes from 38 to 1, and the one left is a lambda's
  continuation. A shell-style `MAX_MODEL_LEN=4096` in the same paste is still read as
  `env.var.MAX_MODEL_LEN`: a script's code is noise, a script's call is not.
- Fixed a keyword pattern that was almost right: `^\s*(?:if |elif |else |…)\b` failed on
  `if "W4A4" in name:` because the space sat inside the alternative and `\b` tested a
  space→quote boundary. Loop and branch lines whose condition is a quoted string stayed in
  the report; now they do not.
- The 20-log corpus and the six real startup logs still report `0 with shapes worth a look`.
  82 tests.

## 0.4.31 — 2026-11-09

- A numeric difference now prints its magnitude as well as the word: `changed  +45.0%`,
  `changed  -4.6%`. `changed` alone reads the same whether torch.compile moved 4% or 400%,
  and the magnitude is what a reader decides on. Text values keep the bare status.
- A per-worker measurement is one fact again. On an 8-GPU box every worker logs
  `Model loading took 1.14 GiB and 36.3 seconds` with its own jitter, so one log held eight
  different values — and two runs of the same box could never agree. `model_loading.seconds`,
  `model_loading.gib`, `weights.load_s` and `weights.available_ram_gib` are now read from the
  lowest rank present, with the other ranks kept in the evidence line. Two *different*
  measurements from the same rank (TP0 loading a backbone and a draft head) are still kept
  apart.
- 80 tests. The two README comparisons still report 48 / 14 and 30 / 33 unchanged: the
  magnitude is added after the status, it does not move the counts.

## 0.4.30 — 2026-11-09

- `servediff audit` takes a directory (or several paths) and answers for the whole corpus:
  one line per log with its unread count, shape count and verdict, then only the rows that
  matter. `servediff audit tests/fixtures` audits all twenty logs in one command, which is
  the claim this project rests on, now checkable in one line instead of twenty runs.
- One more report from the field: vllm#65873, a GB10 (aarch64) box running
  `Nemotron-3.5-Super-VL-120B-A12B-NVFP4` with native MTP. Its nine shapes worth a look were
  all absolute paths to `libcudnn*.so` — an `ldd` dependency dump, which is machine
  inventory, not a setting. Now noise; the log reports nothing worth a look.
  A missing library in an `ImportError` stays a fact.
- Twenty fixtures, 76 tests.

## 0.4.29 — 2026-11-09

- Two more reports, both on hardware the earlier logs did not cover: a GB200 NVL node
  (vllm#66192, aarch64, SM100, torch 2.14.0) and an RTX PRO 6000 Blackwell box
  (vllm#66193). Each carried 9 and 13 "shapes worth a look" that were all machine
  topology — the InfiniBand NIC grid (`NIC0: mlx5_0`, one row per NIC, `NIC Legend:`) —
  and an empty `GPU models and configuration` fact that said nothing.
- An empty value is an absence now. Multi-node images print
  `GPU models and configuration :` with nothing after the colon and then one
  `GPU 0: NVIDIA GB200` per card; that produced `gpu.model = ""`, which compares equal to
  another log's empty string instead of being absent.
- The per-card line is read as the model when the aggregate is silent, and the aggregate's
  own per-card prefix is dropped: `GPU models and configuration : GPU 0: NVIDIA RTX PRO 6000
  Blackwell Server Edition` now yields `NVIDIA RTX PRO 6000 Blackwell Server Edition`, not
  `GPU 0: NVIDIA …`.
- Two new fixtures, two new tests, 72 tests.

## 0.4.28 — 2026-11-09

- The second class of misses, found the same way: twelve logs from issues the tool had never
  seen — six vLLM (a production image, an eight-card MI355X node, a gVisor run, three short
  excerpts) and six SGLang. Before: 61 / 45 / 44 / 51 "shapes worth a look" on the four full
  vLLM logs and 39 / 12 / 11 / 3 / 2 on SGLang. After: nothing worth a look on all twelve.
- Facts: the `[pip3]` rule now covers `flashinfer-python` (it decides kernels; the rest of
  the pip inventory does not). Every `NAME=value` the `collect_env` block ends with becomes
  `env.var.NAME` — the six-entry whitelist became a general rule — minus `CUDA_VISIBLE_DEVICES`,
  `NVIDIA_VISIBLE_DEVICES` and `VLLM_CACHE_ROOT`, which differ between two boxes started with
  the same flags.
- `Could not collect`, `N/A`, `Not Set` and `unknown` are absences, not values. A CUDA box
  was reporting `env.rocm = "Could"` because the rule captured one token of
  `ROCM Version : Could not collect`, so two CUDA boxes compared equal on a value neither had.
- SGLang: its own `name: value` environment dump (`sglang:`, `PyTorch:`, `sglang-kernel:`,
  `flashinfer_python:`, `triton:`, `CUDA_HOME:`, `NVCC:`, `CUDA Driver Version:`,
  `ulimit soft:`), the brackets around its timestamp (`[2026-10-06 00:15:19] …`, stripped as
  a prefix by `parse`, `audit.message` and the noise table alike), `cache.radix` from
  `Tree cache initialized: …`, `transfer.nixl_backend` from `NIXL KVManager initialized
  with backend: UCX`, `startup: failed (scheduler)`, and `IndexError` as a startup cause.
- Audit noise: the pip inventory, GPU topology grid / rocm-smi weight-hops-link matrices /
  `Legend:` / Hypervisor lines, `GPU N:` and `GPU[N] (Topology)` headers, variables printed
  with a name but no value, prose pasted between the logs (Markdown bullets, bold sentences,
  issue-template headings), and source echoed around a traceback (Triton patch diffs,
  frame assignments, bare closing brackets).
- Six new fixtures (`vllm-0.32.0-cuda-collectenv.log`, `vllm-0.33.0rc1-rocm-mi355x.log`,
  `vllm-0.30.0-gvisor-cuda-collectenv.log`, `sglang-0.5.20-env-dump.log`,
  `sglang-0.5-nixl-startup-failure.log`, `sglang-0.5-indexerror-per-request.log`), five new
  tests. 70 tests.
- `diff vllm-65250.log vllm-63505.log` now reports 23 differences / 5 identical and names
  the ones that matter between a gVisor run and a production image: `TORCH_CUDA_ARCH_LIST`,
  `VLLM_BUILD_COMMIT`, `VLLM_IMAGE_TAG`, `CUDA_HOME`, plus the threads and FlashInfer-sampler
  settings only one side printed.

## 0.4.27 — 2026-11-09

- Docs only: the audit writeup now carries the ROCm measurement (vllm#65034: 175 shapes
  worth a look → facts plus noise, nothing left over), and both READMEs name the covered
  stack. No behaviour change — 0.4.26 is the release that reads those lines.

## 0.4.26 — 2026-11-09

- ROCm support, found by auditing a log the tool had never seen: the `collect_env` block
  pasted into vllm#65034 is ~180 lines and each one reads like a setting, so `audit`
  reported 175 "shapes worth a look". It now yields facts — `engine.version` +
  `engine.git_sha`, `env.torch`, `env.torch_rocm`, `env.rocm`, `env.hip`, `env.os`,
  `env.python`, `gpu.model`, `env.pip.{torch,triton,transformers,numpy}` and
  `env.var.{TORCHINDUCTOR_CACHE_DIR,PYTORCH_ROCM_ARCH,MAX_JOBS,LD_LIBRARY_PATH,…}` — and
  the block's scaffolding (section rules, CPU inventory, the flag dump, per-arch
  `amd-torch-device-gfx*` wheels, the wrapped `LD_LIBRARY_PATH` tail) is counted as noise.
  That log now reports nothing worth a look.
- The generic `version X.Y.Z` banner rule no longer takes `CMake version : version 3.31.10`
  as the engine version.
- A launch command pasted into a report (`vllm serve …` plus `--flag value \` lines) is
  noise: the server prints its own args as "non-default args".
- New fixture `tests/fixtures/vllm-0.29.1-rocm-collectenv.log` (176 lines, vllm#65034).
  65 tests.

## 0.4.25 — 2026-11-09

- The last real shapes are facts now, so every full log in this repo audits clean: `Casting
  torch.bfloat16 to torch.float32` → `model.cast` (a coercion no flag records, and it changes
  every result), `AsyncLLM output_handler failed` → `startup: failed (output handler)`,
  `EngineDeadError` (fully qualified as `vllm.v1.engine.exceptions.EngineDeadError`) →
  `startup.error`, and an installed-but-unimportable optional kernel library (`Module …
  was found but failed to import`, `ImportError: … cannot open shared object file`) →
  `kernel.module_missing` / `kernel.missing_lib`.
- Audit: `outputs = await engine_core.get_output_async()` is a frame echo and counts as
  noise; a bare `AssertionError` under a failed import check does too, since the detail is
  the line above that is now read — counting it again would double-report one condition.
- All four full real logs (vLLM 0.28.0 crash, 0.28.0 V1 runner, 0.30.0 killed, 0.23.1 TP=8)
  and both real-log fixtures report `nothing worth a look`. That is a claim about what those
  logs print, not about every log: `writeups/audit-noise-real-logs.md` now says so.
- 63 tests.

## 0.4.24 — 2026-11-09

- New writeup [writeups/audit-noise-real-logs.md](writeups/audit-noise-real-logs.md): what
  `servediff audit` is for, and the before/after counts on all four full real logs
  (146 → 1, 66 → 5, 89 → 4, 190 → 0), with the two bugs that caused them — noise patterns
  matched against the prefixed line instead of the message, and rules pinned to the newest
  wording that silently dropped the older log's facts. Linked from both READMEs; `writeups/`
  now ships in the sdist so the links hold inside the tarball.
- No behaviour change.

## 0.4.23 — 2026-11-09

- Read the configuration a real TP=8 startup log prints. On the vLLM 0.23.1 log from
  vllm#48591, `servediff audit` went from **70 shapes worth a look to zero**: new facts for
  `Using AttentionBackendEnum.FLASHINFER backend.`, `Chunked prefill is enabled with
  max_num_batched_tokens=…`, `Asynchronous scheduling is enabled.`, the forced worker start
  method, `Reducing Torch parallelism from 96 threads to 1`, `Using Eagle3 auxiliary layers
  from config: (…)`, `Cache the graph of compile range (…)`, `Using FlashInfer for top-p &
  top-k sampling.`, `FlashInfer resolved query dtypes: …`, `vLLM is using nccl==…`, and the
  DP group leader's ranks.
- Two rules were quietly missing older logs rather than reporting them as absent:
  `Model loading took 1.14 GiB and 36.34 seconds` (0.23 prints no `memory`) and
  `Filesystem type for checkpoints: FUSE.OFS.` (`\w+` stopped at the dot, taking the
  checkpoint size and available RAM down with it). Both now read on 0.23 and 0.30.
- Recorded outcomes that degrade a run instead of stopping it: `CUDA error
  (…flash_api.cpp): invalid configuration argument` as `kernel.cuda_error`, and a worker
  that dies while the API server keeps printing as `startup: failed (worker died)`.
- Audit noise: a wrapper script's own output (`>>> …`, `[spec_bench] …`, `!!! …`) is noise —
  it is not a vLLM line and no rule could ever read it — along with annotated and
  tuple-assignment frame echoes (`compilation_times: list[…] = self.collective_rpc(`,
  `status, result = mq.dequeue(…)`), `[Autotuner]: Autotuning process starts`, and the
  `I want to run inference of a [specific model]` line from the issue template.
- 63 tests; the new test asserts every one of those lines becomes a fact **and** that the
  audit reports none of them as unread, so the two cannot drift apart again.

## 0.4.22 — 2026-11-09

- `servediff audit` now reads a crashed log. Every line of a real multi-process log carries
  a preamble (`(EngineCore pid=3811) ERROR 10-28 11:26:34 [core.py:1346] …`), and the noise
  patterns described the message, not the prefixed line — so a whole traceback came back as
  "shapes worth a look". The classifier now strips the preamble first (process, level,
  `MM-DD HH:MM:SS` timestamp, `- INFO -`, `module.py:line`), then matches the message, and
  also retries with a trailing `# comment` removed. On the four full real logs in this repo
  the count of shapes worth a look went 146 → 1 (vLLM 0.28.0 crash, UVA), 66 → 5 (0.28.0 with
  the V1 runner), 89 → 4 (0.30.0 killed after starting), 190 → 70 (the TP=8 0.23.1 report) —
  and what is left in the last one is configuration the parser genuinely does not read yet
  (`Using Eagle3 auxiliary layers from config`, `Model loading took … GiB and … seconds`).
- Traceback frames, `^^^` markers, `Process EngineCore:`, `--- Logging error ---` dumps,
  `[shutdown]` teardown lines, uvicorn's shutdown narration, `resource_tracker` and NCCL
  teardown warnings, `[Autotuner]: Autotuning process starts`, and text pasted from the issue
  template are all noise now. A real setting that carries the same preamble stays visible:
  the new test asserts three such lines are *not* noise, so widening a pattern has to break
  the suite.
- New fixture `tests/fixtures/vllm-crash-traceback.log`: a 43-line excerpt of a real crashed
  startup (engine and API-server tracebacks, a logging-error dump, shutdown chatter), with
  machine paths anonymised but line shapes untouched. `audit` on it reports `nothing worth a
  look`, while the `RuntimeError: UVA is not available` line stays consumed by the
  `startup.error` rule rather than appearing as unread.
- 62 tests.

## 0.4.21 — 2026-11-09

- Reads the last nine lines of the real vLLM 0.33.0 startup log that no rule matched. Four
  carry configuration and now have rules:
  `Reducing Torch threads from 16 to 1 for serving` → `cpu.torch_threads` (the pair, because
  `1` alone cannot say what it was cut from); `Dynamo bytecode transform time: 9.43 s` →
  `compile.dynamo_s`; `Compiling a graph for compile range (1, 2048) takes 13.12 s` →
  `compile.range` + `compile.graph_s`; `Initial profiling/warmup run took 1.02 s` →
  `compile.warmup_s`. `compile.total_s` is one number; an upgrade that doubles startup is
  usually one of these three moving, and the split is what says which.
- The other five are noise and are now classified as such instead of being offered back to
  the reader: `[backbone] Starting torch.compile` and `JIT kernel warmup starting.` (phase
  markers with no setting), `Using cache directory: …` and `saved AOT compiled function to …`
  (per-machine absolute paths that can never be equal across two boxes), and
  `Engine N: Avg prompt throughput …` (serving telemetry, not startup configuration).
- `servediff audit` on that log is now clean — 64 unread lines, every one of them noise,
  nothing worth a look — and a test holds the standard there: it fails, naming the shape, the
  day a version prints configuration this parser cannot read.
- README / write-up numbers recomputed from a fresh run, since the new facts change them:
  0.28.0 vs 0.30.0 is now **48 differences / 13 identical** (`--skip-absent`: 3 visible,
  45 hidden), and 0.28.0 with the V1 runner vs 0.30.0 is **27 / 33**. The fact table gained
  rows for the compile split and for CPU threads.
- 61 tests.

## 0.4.20 — 2026-11-08

- `servediff audit --max-shapes N` exits `1` when more than N shapes need a look, so the
  audit report can be a pipeline gate: run it against a log from your own upgrade and the
  build fails when the server starts printing configuration this parser cannot read yet.
  The count comes from the audit module, not from counting lines of the printed report
  (that report truncates shapes, so a text count would be wrong on a large log). `1` here
  means "the parser is behind this log", not "the inputs differ" — `diff`'s `1` keeps its
  own meaning, and `audit` without the flag still exits `0`.
- README: a `ci` badge (it existed and was not shown), and a short "why not just diff the
  two logs" — timestamps, pids and redrawn progress bars make a text diff of a startup log
  useless, and a value that moved is invisible inside a line that reworded itself.
- 60 tests.

## 0.4.19 — 2026-11-08

- Fixes the failed 0.4.18 release: the publish workflow runs the test suite from the
  unpacked sdist, and the new `tests/test_demo_svg.py` opens `tools/render_demo_svg.py` and
  `assets/demo.svg`, neither of which `MANIFEST.in` shipped. The sdist build itself was
  fine; only the tests-from-the-sdist step caught it, which is the point of that step.
  Both paths are now in `MANIFEST.in`, verified by building the sdist, unpacking it, and
  running `pytest` there before tagging.
- `CONTRIBUTING.md` rewritten to be usable by someone who has never seen the code: where a
  rule goes and what a handler returns, why one fact per independent thing, the
  trailing-period trap that has already broken this parser twice, asserting values *and*
  line numbers, and the sdist rule above. A visitor who can add a rule is worth more than
  a visitor who files an issue.
- Also recorded, because it was claimed and wrong: PyPI's project page ships a CSP of
  `img-src 'self' data:`, so the SVG cannot render there. It is a GitHub-surface asset
  only; the alt text and the text sections carry the PyPI page.
- 59 tests.

## 0.4.18 — 2026-11-08

- The README now opens on a picture instead of prose: `assets/demo.svg` is the real
  `servediff demo` output rendered as a terminal window — same flags, `VLLM_CUTLASS` →
  `MARLIN`, plus the one-sided KV cache row. Generated by `tools/render_demo_svg.py` from
  captured stdout (never hand-edited), so the image cannot drift from the command; a test
  regenerates it and fails if the committed copy loses the demo's rows.
  The renderer escapes the text (`<absent>` and `&` must go in escaped or the document
  breaks), wraps prose at 96 columns but never a table row, and lets a wrapped continuation
  inherit the line's colour.
- This release exists because the PyPI project page renders the README: the image and the
  `--skip-absent` section are what a first-time visitor now sees.
- 59 tests.

## 0.4.17 — 2026-11-08

- `servediff diff --skip-absent`: on the most common real pair — same flags, one run died —
  the crash truncates one log, so every later fact is a one-sided `<absent>` row. That was
  40 of the 43 rows on the 0.28.0-vs-0.30.0 pair and it buried the three that explain the
  outcome. The flag keeps only facts both logs printed: **43 → 3 rows**, with the failure
  warning kept and a line stating how many were hidden. Exit code stays 1 (the inputs do
  differ) and `--json` carries both `differences` and `compared_both_sides`, so a script
  cannot read a shorter table as a cleaner result.
- 57 tests (the new one also checks the hidden count is not silently folded into
  "identical").

## 0.4.16 — 2026-11-08

- The loading phase is now three facts instead of one: `weights.cache` (`miss (from
  scratch)` when the run had nothing to read from disk), `model_init.seconds` and
  `weights.load_s`. "Startup got slower" is not answerable until you can see whether the
  time went into constructing the module or reading shards. Audit on the real 170-line
  log: 71 unread / 12 worth a look → **68 unread / 9 worth a look**.
- `servediff audit --markdown`: wraps the report in a code fence sized around the longest
  run of backticks inside it (log messages quote flags with them, and a shorter fence ends
  the block early, silently cutting the lines that follow), so the third field of the
  issue form is one paste rather than a reformat.
- README: the English command table never listed `demo` or `audit` — the two commands a
  fresh install needs — and now does; a GitHub Actions snippet shows the `--markdown` diff
  going into a step summary with exit 1 failing the job. The extracts table gained rows
  for the split loading cost, the weight source/prefetch decision, GPU headroom including
  the CUDA-graph-profiling effective utilization, and the server surface facts.
- Both worked examples regenerated again, because three new facts move the counts:
  `43 difference(s); 13 fact(s) identical` (was 41/13) and `24 difference(s); 31 fact(s)
  identical` (was 22/31), plus the `writeups/wsl-uva-028-vs-030.md` table from the same run.
- 56 tests.

## 0.4.15 — 2026-11-08

- Eleven facts from lines the 0.4.14 audit still called worth a look, all of them things
  that change what a run costs or what it serves while every flag stays the same:
  - `gpu.utilization_effective` and `gpu.utilization_to_match_kv_cache`, from
    `CUDA graph memory profiling is enabled (default since v0.21.0)… equivalent to
    --gpu-memory-utilization=0.4141 without … increase … to 0.4859`. A version that
    changes only this default gives you less KV cache for the same flag, and the line
    says so in numbers.
  - `cuda_graph.profiling` — the mode itself, with the version it became the default.
  - `weights.filesystem`, `weights.checkpoint_gib`, `weights.available_ram_gib` and
    `weights.prefetch` — where the loader read from and whether it could prefetch
    (`Auto-prefetch is disabled because the filesystem (EXT4) is not a recognized
    network FS`). On a slow filesystem this is the whole startup-time difference.
  - `kernel.flashinfer_autotune` (`disabled` when autotune is skipped),
    `server.tasks`, `server.scale_out`, `serving.chat_template_content_format`.
- A trailing full stop bit the parser again: `increase --gpu-memory-utilization to
  ([\d.]+)` swallowed the sentence's period, so `float("0.4859.")` raised and
  `servediff show` died with `exit 1` on a real log. The rule matches the period
  explicitly now, and the test states the failure it prevents.
- Audit on the real 170-line log: 78 unread / 19 worth a look (0.4.14) → **71 unread /
  12 worth a look**, noise unchanged at 59.
- README's two diffs were regenerated with this parser, because eleven new facts move
  the counts: 0.28-vs-0.30 is now `41 difference(s); 13 fact(s) identical` (was 29/13),
  and the V1-runner pair `22 difference(s); 31 fact(s) identical` (was 16/25). The
  `writeups/wsl-uva-028-vs-030.md` table was regenerated from the same run.
- 54 tests.

## 0.4.14 — 2026-11-08

- `servediff audit` separates the gap into noise and shapes worth a look. On the real
  170-line vLLM 0.33.0 log it used to print all 80 unread lines, 59 of them route
  tables, banner art, deprecation warnings, progress bars and access logs — nobody
  pastes that into an issue. Now the noise is counted (grouped in full only under
  `--everything`) and the 19 remaining shapes are listed. Noise is matched against the
  raw line, not the digit-collapsed shape: collapsing turned `200` into `N`, so
  uvicorn access logs survived the first filter.
- Two facts the audit itself surfaced: `cuda_graph.capture_s` and
  `cuda_graph.capture_gib`, from `Graph capturing finished in 3 secs, took 0.25 GiB`.
  That line was previously unread even though a neighbouring line already gave
  `cuda_graph.pool_gib`; a capture pass that takes twice as long is startup-regression
  evidence on its own. Both passes are kept (`[3, 4]` / `[0.25, 0.18]` on the real log).
- A regression test now asserts that no line whose value `parse()` records is also
  reported as unread, so the audit report and the parser cannot drift apart again.
- 53 tests.

## 0.4.13 — 2026-11-08

- `servediff audit <log>` prints the lines no rule reads, grouped by shape, so a user
  whose fact came out `<absent>` can hand over the exact gap instead of describing it.
  The 0.4.9 release pointed at `tools/audit_unread_lines.py`, which only exists in a
  checkout — the capability was unusable for anyone on PyPI. The dev script remains as
  a wrapper over the same code.

## 0.4.12 — 2026-11-08

- `servediff demo` diffs two excerpt logs carried inside the package. Until now a fresh
  `pip install servediff` gave you a tool with nothing to run it on: every documented
  command needs two startup logs, which is exactly what someone who has not hit the
  problem yet does not have. The demo pair is public vllm#47047 text, and the command
  prints why its exit code is 1 rather than 0.
- The package now ships `servediff/examples/*.log` as package data.

## 0.4.11 — 2026-11-08

- The 0.4.9 note here claimed the unread set on the real log went "from 73 to 46
  shapes". It did not: the real movement was 87 lines / 80 shapes down to 80 lines /
  73 shapes, and 73 was the after number, not the before one. Corrected; the rest of
  that entry stands.
- `tools/audit_unread_lines.py` now applies the same ANSI normalisation the parser does
  before deciding a line is unread, so its report cannot count a line the parser reads.

## 0.4.10 — 2026-11-08

- `servediff.__version__` was a literal `"0.1.0"`, unchanged since the first release,
  while `servediff --version` read the installed distribution and printed `0.4.9`. Any
  script or test that checked `__version__` was reading a version nine releases old. It
  is now read from the installed package, and a test pins both paths to one number.

## 0.4.9 — 2026-11-08

- Eight more facts, chosen by auditing what the parser still ignores rather than by
  guessing: `gpu.free_gib` / `gpu.total_gib` / `gpu.utilization` / `gpu.budget_gib`
  (startup memory headroom), `jit.warmup_s` and `jit.monitor_mode`, the
  `compile.autotune_gemm` gate, `sampling.overridden` (the model's
  `generation_config.json` quietly replacing the sampler flags),
  `backend.candidates`, and SGLang's `server.command`.
- Kernel selection is read one fact per op (`kernel.ir_op.rms_norm=native`,
  `kernel.ir_op.gelu_and_mul_sparse=triton|native`) instead of one long
  `IrOpPriorityConfig(...)` string. The long form is wider than the table, so two
  different configs truncated to the same visible text while the row said "changed" —
  a claim the output did not support.
- `tools/audit_unread_lines.py` reports, per fixture, the lines no rule matches. That
  is how the list above was found, and it is the first place to look before adding a
  rule: a tool that shows "what the server actually ran with" is defined as much by
  what it ignores. On the real 0.33.0 log this set went from 87 lines / 80 distinct
  shapes to 80 lines / 73 shapes; the remainder is route tables, the ASCII banner,
  torch deprecation warnings, checkpoint-shard progress bars and `[backbone]`
  compile-timing detail, none of which should be diffed.
- README examples regenerated from the build: 0.28.0 vs 0.30.0 now reads
  `29 difference(s); 13 fact(s) identical`, and the V1-runner pair
  `16 difference(s); 25 fact(s) identical`.

## 0.4.8 — 2026-11-08

- The first command in the README was wrong on the page people actually run. After 0.4.7
  taught the parser to read the concurrency line, that 47047 pair prints
  `2 difference(s); 0 fact(s) identical` — a `kv_cache.max_concurrency` row appears
  because only one of the two excerpts quoted it — but the README still showed
  `1 difference(s)`. Corrected, and the new row is explained as a presence difference
  rather than a changed value.
- Packaging metadata: the keyword list now carries the terms a person would actually
  search for (startup-log, log-diff, kv-cache, cuda-graph, moe, tensor-parallel,
  inference-server), and the package declares `Topic :: System :: Logging`. Discovery on
  PyPI and GitHub is a packaging problem before it is a content problem.

## 0.4.7 — 2026-11-08

- Seven more facts are read from a vLLM startup log, all of them found missing while
  diffing the real 0.33.0 log from the previous entry: `model.architecture`
  (`Resolved architecture`), `model.max_len` (`Using max model len`),
  `kv_cache.memory_gib`, `kv_cache.tokens`, `kv_cache.max_concurrency`,
  `engine.init_s` and `server.bind`. These are the numbers that move when a flag change
  costs you concurrency; before this, a diff could show the args changing and the
  capacity silently not at all.
- `model.max_len` is the value the engine settled on, not the requested one, which is
  why the same `--max-model-len` can still differ between runs.
- README examples regenerated from the build: 0.28.0 vs 0.30.0 now reads
  `20 difference(s); 11 fact(s) identical`, and the V1-runner pair
  `14 difference(s); 16 fact(s) identical`. The vllm#47047 fixture diff now also
  reports a concurrency line that only one side quoted — a presence difference, not a
  backend change.

## 0.4.6 — 2026-11-08

- `capture` verified against a live server, not a fake process: vLLM 0.33.0 on one RTX 4060
  Laptop (8 GB, WSL2), Qwen2.5-0.5B. The server reached `/health` and served a completion,
  the log was written while it ran, the env snapshot landed before launch, and Ctrl-C exited
  `130` with the log intact. A failed run still passed the server's exit code (`1`) through.
- New fixture `vllm-0.33.0-real-rtx4060.log` — a full 170-line log from that run rather than
  an issue excerpt — pinned by `test_real_server_log_from_033_is_parsed`, which guards
  `engine.version`, `model_runner V2`, the attention backend, the KV-cache layout, UVA
  fallback, world size, the CUDA-graph capture list and the model-loading size.
- `Limits` states this run as evidence: the two env snapshots of the same command diff down
  to the two variables that separated a crash from a start.

## 0.4.5 — 2026-11-07

- `series` text output keeps both ends of a long value instead of cutting it at a fixed
  column width. Two backends that differed only in their last characters
  (`…NoDPEPMonolithic` vs `…NoDPEPModular`) rendered as identical text in two columns while
  the row was labelled "changed" — a report that hides the difference it reports. Markdown
  output was never truncated and is unchanged.
- `Limits` now lists what the fixtures actually contain (including the TP4+EP H200 excerpt
  from vllm#45604) instead of claiming a single-GPU SGLang log that is not in the repo;
  `tests/fixtures/README.md` documents all seven fixtures rather than five.

## 0.4.4 — 2026-11-07

- `capture` survives Ctrl-C. That is how a server run actually ends: interrupt it once the
  startup lines are printed. It used to leave a traceback and exit 1, which this tool
  reserves for "the inputs differ"; it now stops the child, keeps the log, says where it
  is, and exits 130. The server's own exit code is still passed through on a normal exit.
- README documents `capture` as the way to produce both logs, since most people arrive
  with one log and a question rather than two ready files.

## 0.4.3 — 2026-11-07

- `capture` now snapshots the interpreter the command itself names. `capture -- python -m
  vllm...` used to snapshot the interpreter servediff runs on, which is a different venv
  often enough that a later snapshot diff compared two copies of servediff and reported
  no difference — the failure mode this tool exists to name. When the command does not
  name a python launcher, the snapshot line now says which interpreter it describes and
  points at `--python`.
- The snapshot line prints the interpreter path it used, so the JSON and the log agree.

## 0.4.2 — 2026-11-07

- An unreadable path (`servediff show typo.log`, `diff` with a missing side) no longer
  raises a traceback and exits 1. Exit 1 means "the two sides differ", so a mistyped
  path used to read as a detected difference; it now prints one line and exits 2.
- `diff` on two `servediff env` snapshots marks truncated cells with `...`, like the
  log diff already did.
- README gains a commands-and-exit-codes table, including which env vars `env` records
  and the fact that `1` never means "something went wrong".

## 0.4.1 — 2026-11-07

- Table cells mark truncated values with `...` instead of cutting them mid-word. A
  cell that ended in `TP ran` read like the value itself rather than a clipped column.
- README examples regenerated from this build. The quickstart block now shows the two
  `note:` lines the tool actually prints, and the 0.28-vs-0.30 blocks carry the real
  `15 difference(s); 9 fact(s) identical` and `10 difference(s); 13 fact(s) identical`
  — the old numbers predated several fact rules.

## 0.4.0 — 2026-11-06

- `servediff series a.log b.log c.log` compares three or more runs in one table with
  one column per log, so a middle version is not lost between two pairwise diffs —
  the shape regression threads like vllm#53472 actually have. Supports `--markdown`,
  `--json`, `--all`, `--raw-args`, and `-` for stdin on at most one entry. Exit codes
  match `diff`.

## 0.3.10 — 2026-11-06

- `show` and `diff` accept `-` for stdin, so a log can be read straight off the
  machine that produced it: `ssh node 'cat serve.log' | servediff show -`. Only one
  side may be `-` (stdin is read once), and `--markdown` labels it `stdin`.

## 0.3.9 — 2026-11-06

- Keep the all-reduce dispatch chain when the log line was truncated. The statement
  carries a seven-item candidate list, and log sinks cut long lines; the previous
  rule required the trailing `out of potential backends: [...]` and produced nothing
  silently when it was missing. The fallback is anchored to a real log prefix
  (`[module.py:NN]`), so the same sentence quoted as prose in a pasted issue body
  does not turn into a fact.
- Pin multi-group parsing with a verbatim 4×H200 TP4+EP fixture from vllm#45604: the
  fact now carries one value per group (`tp:0` and `ep:0`) instead of collapsing to
  the last line.

## 0.3.8 — 2026-11-06

- `diff` now names why a fact could not be observed, for the two cases the tool can
  actually tell: a single-rank log (`world_size=1`, so no all-reduce group exists and
  no backend is chosen) and a v0.29.0+ log whose FlashInfer auto-selection fact is
  missing because that line is logged at DEBUG. Previously a fact absent from *both*
  logs produced no row at all and was silently counted as identical, which let
  "nothing could be seen" read as "nothing changed". Reported in text, JSON (`notes`)
  and `--markdown` output.

## 0.3.7 — 2026-11-05

- Extract the all-reduce backend selection from TP startup logs: the chosen
  dispatch list per TP group (`backend.allreduce_dispatch` /
  `backend.allreduce_potential`), FlashInfer's auto-selected allreduce backend
  (`backend.allreduce_flashinfer`) and its norm-fusion workspaces
  (`backend.allreduce_fusion`). Verified against a real vLLM 0.23.1 TP=8 log
  (vllm#48591). These lines are the only place the fused communication path shows
  up, so a silent switch from custom all-reduce to FlashInfer's path becomes
  visible in a diff instead of staying invisible.

## 0.3.6 — 2026-11-03

- Parse SGLang's `server_args=ServerArgs(...)` keyword-call form (not just the
  `{...}` dict literal): verified against a real SGLang startup log (sgl-project/sglang#4862,
  v0.4.4, TP=4) — model, TP size, memory fraction, port, attention/sampling backends
  and startup failure with root cause are now extracted instead of nothing.
- Record `startup=failed` for AttributeError/KeyError/TypeError as well as
  RuntimeError/ValueError/OOM/AssertionError; expose the SGLang `port` arg.
- Add `servediff --version` (prints the installed package version).

## 0.3.5 — 2026-11-02

- README: document the real TP>1 validation (vLLM 0.23.1 TP=8, issue #48591) and
  the #63705 decode-only regression analysis as usage examples.

## 0.3.4 — 2026-11-02

- Verified the parallel-config rules against a real TP=8 startup log (vLLM 0.23.1,
  issue #48591): every rank prints its own `world_size=N` line and the role
  assignment has no ETP field. `parallel.world_size`, `parallel.roles`,
  `compile.fusions` and `model_runner` all parse correctly; added a regression test
  distilled from that log.

## 0.3.3 — 2026-11-02

- Extract `model_loading.gib` / `model_loading.seconds` and the torch.compile
  outcome (`compile.artifacts`, `compile.total_s`); detect the V1 runner via the
  `gpu_model_runner.py` module when the "Using VN Model Runner" line is absent
  (it is V2-only). All found by continuing to dogfood the full 0.28-V1 / 0.30-V2
  startup logs: a V1-vs-V2 diff now shows the runner change, model-loading time,
  compile artifacts and CUDA-graph capture differences.

## 0.3.2 — 2026-11-02

- Extract the actual parallel configuration (`parallel.world_size` and the
  DP/PP/PCP/TP/ETP/EP/EPLB role assignment) and the KV cache layout
  (`Using LBNHC KV cache layout`, 0.30+). Found by dogfooding the parser against
  full vLLM 0.28/0.30 startup logs; the role-assignment line is the only place
  TP>1 and expert parallel appear in the log, so it makes TP4+EP configs visible
  to `servediff diff`.

## 0.3.1 — 2026-10-31

- `--markdown` now also works on `servediff env` snapshots, so a `capture`-based
  environment diff can be rendered as a paste-ready report.

## 0.3.0 — 2026-10-31

- `servediff capture -o run.log -- <server command>`: run a server, tee its
  stdout+stderr to a log file, and snapshot the environment to `run.log.env.json`
  before the server starts. Two captures become a one-command `servediff diff`.

## 0.2.0 — 2026-10-31

- `servediff diff --markdown`: GitHub-flavored report with a table and a collapsible
  **Evidence** section quoting the log line behind each fact. Ready to paste into a
  regression issue.
- PyPI badges in the README.

## 0.1.0 — 2026-10-29

First version.

- `servediff show <log>`: effective configuration facts from a vLLM or SGLang startup log,
  each with the line it came from.
- `servediff diff <a.log> <b.log>`: version, arguments, attention / MoE backend, CUDA graph
  mode and captures, model runner, UVA fallback, startup outcome and root-cause error.
  Warns when one side failed to start. Errors after the server is ready are hidden unless
  `--all`.
- `servediff env [--python PATH]`: snapshot package versions, GPU and engine env vars of a
  server environment, with secret-looking values redacted. `servediff diff` on two
  snapshots lists inference-relevant changes first.






