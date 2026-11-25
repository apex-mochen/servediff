# One file out of four: `servediff audit <dir>` on a corpus rotated the way logrotate does it

`servediff audit <dir>` is the command that answers "is my parser behind this corpus of logs".
Its scope is therefore part of its answer. Until 0.4.109 the scope was two suffixes —
`*.log` and `*.log.gz` — and logrotate's default layout is neither of those:

```
serve.log
serve.log.1
serve.log.2
serve.log.3.gz
```

Four files, one matched. The other three were not in the list, not in the count, and not
mentioned:

```
$ servediff audit rotated/
serve.log                              32 unread    32 shape(s)  nothing worth a look
1 log(s): 0 with shapes worth a look
```

That `1 log(s)` is the sentence this whole post is about. It is not an error message, so
nothing looks wrong; it reads like a measurement of the directory. A gate built on it passes,
and passes on a promise nobody checked.

## Why the number is the problem, not the missing files

The tool was already explicit about this failure mode one level up. Its directory expansion is
recursive because a corpus lives in dated subdirectories, and the docstring says why:

> a one-level glob silently audits 21 of 86 logs while the summary still reports a count —
> the exact silent-scope failure this tool exists to catch

The same failure was still there one level down, in the file names. A rotated log is not a
different kind of file. `serve.log.2` contains the same kind of lines as `serve.log`; it is the
previous day of them.

## What the fix is

Expansion is by name pattern now, not by a list of suffixes:

```
serve.log        serve.log.1     serve.log.2     serve.log.3.gz
```

and the same directory reads as four:

```
$ servediff audit rotated/
serve.log                              32 unread    32 shape(s)  nothing worth a look
serve.log.1                            32 unread    32 shape(s)  nothing worth a look
serve.log.2                            32 unread    32 shape(s)  nothing worth a look
note: rotated/serve.log.3.gz was gzip-compressed; decompressed 1906 byte(s)
serve.log.3.gz                         32 unread    32 shape(s)  nothing worth a look
4 log(s): 0 with shapes worth a look
```

`--json` lists the same four entries, each carrying the facts and the shapes of the single
file. That is the assertion in the test, not the layout of the text output.

Listings also sort by rotation number. A plain string sort reads `serve.log`, `serve.log.1`,
`serve.log.10`, `serve.log.2` — an order nobody chose, in the one place where the order is
supposed to be the history.

## The check that notices this without being told

`tools/invariants.py` asks the same question of the corpus from several angles; the tenth one
builds the four-file layout out of every fixture and compares each entry against a single-file
audit of the same log:

```
$ python tools/invariants.py --repo . --only rotations
71 log(s) under tests/fixtures
ok    rotations are logs             71 log(s)
```

Point it at the commit before the fix and it says what the tool used to say, per log:

```
$ git worktree add /tmp/w a85db0f && python tools/invariants.py --repo /tmp/w --only rotations
FAIL  rotations are logs             3 log(s)
        ('…/vllm-0.32.0-cuda-collectenv.log', 'counted', 1, 'serve.log .1 .2 .3.gz')
```

## What to run on your own box

```
servediff audit /var/log/vllm/            # every rotation, or one file, in the count
servediff audit /var/log/vllm/ --json     # the same entries, for a gate
servediff diff /var/log/vllm/ --against facts.json   # did any row of the matrix change a fact
```

If a directory holds a rotation this tool does not read — bzip2, xz, zstd — 0.4.110 names the
container and exits 2 (`serve.log.1.bz2: bzip2-compressed, which is not read here; \`bunzip2' it
first`), which is what a single file already did. A copy somebody kept by hand
(`serve.log.5.bak`) is not a rotation and stays out. gzip is read directly, on disk and through
a pipe.
