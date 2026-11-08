# Contributing

The most useful contribution is a **log line servediff missed**.

1. Run `servediff show your-server.log` and find the setting that is `<absent>` but is
   printed in the log.
2. Open an issue with the engine version and the exact log line(s). Remove paths,
   hostnames and anything private first. `servediff audit run.log` prints exactly the
   lines no rule reads yet — paste that, it is the useful half of a report.
3. For a pull request: add the lines as a fixture under `tests/fixtures/` (say where they
   came from in `tests/fixtures/README.md`), add a test, then the rule in
   `src/servediff/cli.py`.

```bash
pip install -e ".[dev]"
pytest -q
```

Rules for new facts:

- A fact comes from a line the server printed. No defaults, no guessing.
- If the same setting is printed in several formats across versions, one rule should
  cover them, with a fixture for each format.

## Where a rule goes

`LINE_RULES` in `src/servediff/cli.py` is a list of `(compiled regex, handler)`, where the
handler returns `[(fact_key, value)]` pairs:

```python
(re.compile(r"Loading weights took ([\d.]+) seconds"),
 lambda m: [("weights.load_s", float(m[1]))]),
```

- One fact per independent thing. `Model loading took 5.2 GiB and 3.13 seconds` is a total,
  and a total cannot answer "which step got slower" — that is why the loading phase is three
  facts here.
- Copy the line from the log verbatim, **including trailing punctuation**. A rule written as
  `increase to ([\d.]+)` captures the sentence's final period and `float("0.4859.")` raises;
  match the period explicitly: `([\d.]+)\.`.
- Assert the value **and** the line number in `tests/test_cli.py`. `show` prints line
  numbers on purpose: a fact attributed to the wrong line is a wrong fact.

## Test rules

- Test the value, not the rendered text. An assertion on a substring of a formatted table
  passes while the number is wrong.
- **Tests must pass from the built sdist**, not only from the checkout: the publish workflow
  unpacks the sdist and runs `pytest` there. Any file a test opens — fixture, script, asset —
  must be listed in `MANIFEST.in`, or the release fails. This is how a missing `tools/` entry
  first showed up.
- For a rule about absence, the negative test matters more: `<absent>` must never be
  reported as "off" or "disabled".

## What a PR here should not do

Infer which difference caused a slowdown, and never fill a gap with a default. If the log
did not print it, the answer is `absent` — that distinction is the product.
