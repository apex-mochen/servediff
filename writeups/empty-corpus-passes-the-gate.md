# A gate that measured zero logs and reported green

`servediff audit <dir> --max-shapes 0` is the shape a CI gate takes: any line the parser cannot
read is a shape worth a look, and zero is the only acceptable number. Point it at a directory
that holds no logs at all and, until 0.4.111, this is the whole exchange:

```
$ servediff audit /var/log/vllm-old --max-shapes 0
audit: 0 shape(s) worth a look, threshold 0
$ echo $?
0
```

No file was read. The number is real — it is the count of shapes over an empty set — and it is
the one answer a gate must never produce, because the same output comes from a typo in the
path, from a volume that failed to mount, and from a server that never wrote a log.

`repeats <dir>` did the same thing, and the directory mode of
`diff <dir> --against facts.json` said `no *.log inside` about a directory whose logs were
named `serve.log.1`.

## Why "0 shapes" is the worst possible output

The tool already treats a paste that yields no fact as unread, not as clean:

> a file that yields no fact is not a clean file: it is a file this tool cannot read

That rule exists because an empty answer and a clean answer look the same. A directory with no
logs is the same failure at the level above the file: the corpus is empty, so every claim made
about it is vacuously true.

## The fix, and the first thing it broke

All three commands name the directory and exit 2:

```
$ servediff audit /var/log/vllm-old
audit: /var/log/vllm-old: no log inside (looked for *.log, *.log.N and .gz/.bz2/.xz/.zstd of those)
$ echo $?
2
```

The first thing this broke was this repo's own CI. The regression gate loops over the fixture
corpora, and one of them — `tests/fixtures/sglang-2026-11` — holds only a `PROVENANCE.md` for
pastes that were found and kept elsewhere. It had been passing for months by measuring nothing.
Four jobs went red on the release commit, all at the same step.

The fix is not to weaken the check, and it is not to delete the note. It is to make the gate
list only directories that hold a log, and to make that a test rather than a comment: the test
parses the `for d in` list out of `.github/workflows/ci.yml` and asserts every listed directory
expands to at least one log. A directory added to the gate without a fixture now fails locally,
before it can fail in CI for a reason nobody understands.

## What this changes for a gate

```
servediff audit /var/log/vllm --max-shapes 0     # green means: logs were read, and none were unread
servediff diff /var/log/vllm --against facts.json  # green means: every log in the corpus matches
```

Green now means two things instead of one. If the path is wrong you get a sentence with the
path and the exit code a broken input deserves, which is what `2` was already for everywhere
else in this tool.
