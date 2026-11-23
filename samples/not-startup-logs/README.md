# Pastes that are not startup logs

Sixteen pastes collected from the same vLLM and SGLang issues as the corpus in
`tests/fixtures/`, kept here because they are evidence of what those threads actually contain,
and deliberately out of the corpus.

They are reproducer scripts, docstrings and `collect_env`-style notes pasted into issue bodies
next to the log. No rule reads a single fact out of any of them: `servediff show` on one exits
`2` with `no vLLM or SGLang configuration lines recognised`, which is the correct answer.

They are here because of how they used to be treated. While they sat in `tests/fixtures/`,
`servediff audit tests/fixtures --max-shapes 0` reported **87 log(s), 0 with shapes worth a
look**: every line of a Python script had been classified as noise — `try:`, `flush()`,
`time.sleep(N)` all matched the traceback-frame patterns — so the sixteen files that produced
no fact at all were the sixteen files the corpus report was quietest about. A corpus that
cannot read 18% of itself and calls itself clean is the failure mode this tool exists to
catch in someone else's log.

Since 0.4.96:

- `audit` names a file that yields no fact: `0 facts: not one configuration line in this file
  is recognised — an unread file is not a clean one`.
- Such a file always counts once toward `--max-shapes`, so no baseline can accept it away.
- The startup-log corpus is `tests/fixtures/` at 71 logs; this folder is the expected-failure
  set, and `test_a_file_that_reads_nothing_is_not_reported_clean` runs against it.

Issue numbers for each file are in the `PROVENANCE.md` of the collection it came from.
