# 198 regexes per line, 1 456 284 times

A stranger's vLLM startup log is not 2 000 lines. With `VLLM_LOGGING_LEVEL=DEBUG`,
tensor-parallel workers, and CUDA graph capture, a 70 B model's dry run lands
around 100 MB. I generated one — 1 456 284 lines, 96 MB — and pointed the tool at it.

```
$ servediff show big.log
... 59.01s
$ servediff audit big.log
... 90.21s
```

A minute to read a file. That is the difference between "I'll try this" and closing
the tab. So I profiled it instead of guessing.

## Where the time was

`show` walks every line through `LINE_RULES`: 198 compiled patterns, each one
`re.search`-ed against the raw line. cProfile on a 10 MB slice: **4.95 million
`re.search` calls, 6.0 s**.

The important part is not the count, it is the distribution. A log is five shapes
repeated a million times:

```
[model_runner.py:1043] INFO 11-23 22:02:40 Capturing CUDA graphs (FULL):  0%| ...
[worker.py:291] INFO 11-23 22:02:41 It took 0.252 s to load weights
[core.py:71] INFO 11-23 22:02:41 Engine 0 started with 1163 blocks
```

Every one of those lines tests all 198 rules, and 196 of those tests are
guaranteed to fail — the same 196, on every repetition. That's the shape of a
problem that wants memoisation, not a faster regex.

## What I tried first, and why it failed

Each rule in the table is annotated with the log site it came from, like
`[model_runner.py:1043]`. Natural idea: bucket rules by that marker, so a
`worker.py` line only tests `worker.py` rules.

It doesn't work. Of the 198 rules, **16** are anchored on a literal `\[`, the rest
start with `\b` or match anywhere in the line. Grouping by a marker that 92% of
the
rules don't carry gives you no grouping at all.

Second attempt: lift the longest literal out of each pattern, join all 190 of them
into one alternation, and use that as a gate — one `re.search` to decide whether a
line is worth the full walk. This made it **slower**: 77 s against 59 s. The
alternation is a small automaton over 190 branches; on a line that matches nothing,
you pay to traverse all of it, every line, every time. A gate that always says
"no" is not cheaper than the thing it gates.

## What worked

Two changes, both boring.

**Ask the question per shape, not per line.** Strip the digits out of a line and
what's left is its shape — `It took N s to load weights`. Two lines with the same
shape accept exactly the same rules. So cache `shape -> tuple of rules that
matched`, and on a repeat only those few rules run. In practice that is **0 to 2
regexes per repeated line** instead of 198, and the first occurrence of a shape
still pays the full walk to fill the cache.

The literal extraction from the failed attempt survives, used differently: per
rule, take the longest run of literal text at paren depth zero (bail out at a
top-level `|`, don't descend into `(?` groups or character classes). That gives
each line a cheap pre-filter — 190 rules keep a literal, 8 don't and stay in the
always-test set.

**Stop running whole-line regexes that are going to be no-ops.** `_normalize()`
used to do three `sub` passes on every line (ANSI escapes, `[11-23 22:02:40]`,
fancy punctuation). All three are now a character pre-check — `"\x1b" in line`,
`line.startswith("[")`, six membership tests for the typographic characters —
and the `sub` only runs when the pre-check says there's something to remove.

Then the same trick in the other two commands: `audit`'s unread-line sampler was
walking the full flat rule list per line (that was the whole 90 s), and it now asks
the same shape cache whether the line is already known. `repeats` caches
`message()`+`shape()` per shape, and only runs the embedded-record lookahead split
when the line actually contains `pid=`.

## The numbers

Same machine, warmed page cache, 96 MB / 1 456 284 lines:

| command   | before | after  |
|-----------|--------|--------|
| `show`    | 59.01s | 11.05s |
| `audit`   | 90.21s | 14.76s |
| `repeats` | 37.90s | 22.80s |

`repeats` is the laggard because it still calls the clock parser on each line to
compute gaps; it's the next thing to look at, not something I fixed this round.

## The part that actually matters

A cache is a promise that two inputs are equivalent. If the promise is wrong,
`show` silently reports different facts on the same log, which is worse than
being slow. So every change was checked the same way: run the previous build and
the new one over all 87 corpus logs and compare output **byte for byte**.

- `parse` on 87 logs: `0 differing`
- `repeats.groups` on 87 logs: `0 differing`
- `pytest`: 178 passed

That check caught one real regression. The empty-body fast path cached "no
message" as an ordinary shape key, so progress-bar lines that strip to an empty
message started getting counted as a group of their own — 8 of 87 logs changed
output, all of them real vLLM logs. The fix is small (cache the absence explicitly
and skip before bucketing), but without the byte comparison it would have shipped
as "performance work, no behaviour change".

`tools/bench_parse.py` regenerates the synthetic log and prints the three timings,
so this number is checkable rather than remembered.
