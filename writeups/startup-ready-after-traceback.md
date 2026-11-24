# "The server is fired up and ready to roll!" — four lines after the traceback that a diff read as a server which never started

A startup log can record two attempts in one file. The process dies, the supervisor restarts
it, the restart comes up. Read from the top, that log says `failed` first, and any tool that
stops reading there will describe the wrong run — including the tool in this repo, until
0.4.106.

Two real SGLang logs, both kept in `tests/fixtures/` of this repo, record exactly that:

| log | line 55 / 59 | `startup` fact |
|---|---|---|
| `sglang-2026-10/sglang-41654.log` | `TypeError: unsupported operand type(s) for //: 'NoneType' and 'int'` (55) — `The server is fired up and ready to roll!` (59) | `["failed", "ready"]` |
| `sglang-2026-10/sglang-43741.log` | `The server is fired up and ready to roll!` (5) — `Scheduler hit an exception` (6) | `["ready", "failed (scheduler)"]` |

Read the second row: the server started, served, and then died. Read the first: the server
died, and then started. Neither is "a server that never came up", and both were reported as
one or the other wrong thing.

## What 0.4.105 printed

```
$ servediff diff sglang-41654.log env-dump.json
warning: baseline server failed to start: TypeError: unsupported operand type(s) for //: 'NoneType'...; facts after the failure are absent because startup stopped, not because they were off
```

The second half of that sentence is a claim about the whole table, and it was false. The
server started four lines later, so the facts were not absent because startup stopped. On a
real pair from the same repo (`sglang-41654.log` against `sglang-42684.log`, 21 differences =
10 changed + 11 one-sided), the warning told the reader to read 11 rows as consequences of
a startup that never happened.

The other log got nothing at all:

```
$ servediff diff sglang-43741.log env-dump.json
note: baseline: no all-reduce facts and no rank line: ...
```

`after_ready.error` appeared in the table as a one-sided row, with no sentence saying an
error can arrive after startup and that it says nothing about the configuration.

## The rule that was wrong

The warning was keyed off a boolean over the values of one fact:

```python
if "failed" in (parsed["facts"].get("startup") or Fact()).values:
```

That answers "did a failure appear in this log", not "did this server come up". The two
diverge exactly when a retry succeeded, which is the case a startup log is most worth reading
carefully. It was never a parsing problem: the parser had already recorded both values,
each with its own evidence line — `[(55, "TypeError: ..."), (59, "The server is fired up and
ready to roll!")]`. The order was in the data; the warning ignored it.

## What 0.4.106 prints

The warning now follows the recorded line numbers, so each of the three outcomes gets the
tense its log supports:

```
$ servediff diff sglang-41654.log env-dump.json
warning: baseline failed to start and then came up on a retry: TypeError: unsupported
operand type(s) for //: 'NoneType'...; the facts below come from both attempts, so a row
that only one side has may belong to the attempt that died, not to the one that served

$ servediff diff sglang-43741.log env-dump.json
warning: baseline reported ready and then failed: AssertionError: Unsupported rhs dtype
fp8e4nv; an error after startup says nothing about how it was configured — that row is
compared as a fact, not read as a reason
```

A log that genuinely never came up keeps the original sentence, word for word. `--markdown`
carries the same sentence in the same tense, so a report pasted into an issue does not say
one thing while the terminal said another.

The three cases are worth stating as a rule, because they are the three ways a startup
log ends:

| `startup` values, in line order | what the log is | what a warning owes the reader |
|---|---|---|
| `failed`, never `ready` | the run never came up | later facts are absent because startup stopped — not because they were off |
| `failed` … `ready` | a retry succeeded | the facts come from both attempts; a one-sided row may belong to the attempt that died |
| `ready` … `failed` | it started and then died | an error after startup says nothing about how it was configured |

That third row is the one that was silent. A post-ready failure is not a configuration
difference, and putting its text in a table of settings invites exactly the wrong reading —
so it stays compared as a fact, with a sentence saying not to read it as a reason.

## Reproduce it

```console
$ servediff diff tests/fixtures/sglang-2026-10/sglang-41654.log tests/fixtures/sglang-0.5.20-env-dump.log
$ servediff diff tests/fixtures/sglang-2026-10/sglang-43741.log tests/fixtures/sglang-0.5.20-env-dump.log
```

Both exit 1, because both pairs differ. The first prints the retry sentence and no longer
prints "facts after the failure are absent"; the second prints the post-ready sentence where
0.4.105 printed nothing. The regression tests read the same two files, and they fail if the
warning is keyed off the string again.
