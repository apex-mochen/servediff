# Contributing

The most useful contribution is a **log line servediff missed**.

1. Run `servediff show your-server.log` and find the setting that is `<absent>` but is
   printed in the log.
2. Open an issue with the engine version and the exact log line(s). Remove paths,
   hostnames and anything private first.
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
