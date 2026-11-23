"""Run every servediff command over every corpus log and report anything that raises.

The corpus tests assert what the commands *print*; nothing asserts that they *return*
instead of raising on a log they have not seen. A stranger's log is usually the first
one of its kind, so this sweeps the matrix — commands x logs x the flags people
actually pass — plus a handful of hostile files (empty, binary, one 4 MB line,
UTF-16 from a PowerShell redirect, a .gz), and prints exit codes and tracebacks.

    python tools/cli_sweep.py --fixtures tests/fixtures
    python tools/cli_sweep.py --fixtures tests/fixtures --verbose

Exit status is 1 when anything raised, so it can gate a release.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import gzip
import io
import os
import sys
import tempfile
import traceback
from pathlib import Path

# `--src ""` runs the sweep against the *installed* package, which is how you check that the
# harness would have caught the bug a previous release shipped.
_SRC = os.environ.get("SERVEDIFF_SRC", str(Path(__file__).resolve().parent.parent / "src"))
if _SRC:
    sys.path.insert(0, _SRC)

from servediff.cli import main  # noqa: E402


def run(argv: list[str]) -> tuple[int, str]:
    """(exit code, stdout+stderr) for one invocation, never raising.

    A crash is reported as -1: the printed text cannot be inspected for "Traceback",
    because a log excerpt the command quotes legitimately contains that line.
    """
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = main(list(argv))
    except SystemExit as exc:  # argparse --help
        code = exc.code if isinstance(exc.code, int) else 2
    except Exception:
        return -1, traceback.format_exc(limit=3)
    return code, out.getvalue()


def hostile_files(tmp: Path) -> list[Path]:
    made = []
    empty = tmp / "empty.log"
    empty.write_bytes(b"")
    made.append(empty)

    binary = tmp / "binary.log"
    binary.write_bytes(bytes(range(256)) * 40)
    made.append(binary)

    oneline = tmp / "one-huge-line.log"
    oneline.write_text("[model_runner.py:1043] INFO 11-23 22:02:40 " + "x" * (4 << 20) + "\n",
                       encoding="utf-8")
    made.append(oneline)

    utf16 = tmp / "utf16.log"
    body = "[model_runner.py:1043] INFO 11-23 22:02:40 kv cache 4.91 GiB\n"
    utf16.write_bytes(body.encode("utf-16-le"))
    made.append(utf16)

    crlf = tmp / "crlf.log"
    crlf.write_bytes(b"[worker.py:291] INFO 11-23 22:02:41 It took 0.25 s to load\r\n" * 3)
    made.append(crlf)

    gz = tmp / "rotated.log.gz"
    with gzip.open(gz, "wb") as handle:
        handle.write(b"[core.py:71] INFO 11-23 22:02:41 Engine 0 started with 3 blocks\n" * 3)
    made.append(gz)

    dashdash = tmp / "dashes.log"
    dashdash.write_text("--- not a log ---\n", encoding="utf-8")
    made.append(dashdash)

    return made


def commands(log: Path, other: Path | None) -> list[tuple[str, list[str]]]:
    cases = [
        ("show", ["show", str(log)]),
        ("show --json", ["show", str(log), "--json"]),
        ("audit", ["audit", str(log)]),
        ("audit --everything", ["audit", str(log), "--everything"]),
        ("audit --json", ["audit", str(log), "--json"]),
        ("repeats", ["repeats", str(log)]),
        ("repeats --json", ["repeats", str(log), "--json"]),
        ("repeats --top 2", ["repeats", str(log), "--top", "2"]),
        ("repeats --min 4", ["repeats", str(log), "--min", "4"]),
    ]
    if other is not None:
        cases.append(("diff", ["diff", str(other), str(log)]))
        cases.append(("diff --json", ["diff", str(other), str(log), "--json"]))
    return cases


def extra_cases(fixtures: Path, logs: list[Path]) -> list[tuple[str, list[str]]]:
    """The shapes a stranger hits by typing the wrong thing, not by having a bad log."""
    cases = [
        ("no arguments", []),
        ("unknown command", ["nonsense"]),
        ("show nothing", ["show"]),
        ("audit nothing", ["audit"]),
        ("repeats nothing", ["repeats"]),
        ("diff one arg", ["diff", str(logs[0])]),
        ("series two logs", ["series", str(logs[0]), str(logs[1])]),
        ("series three logs", ["series", str(logs[0]), str(logs[1]), str(logs[2])]),
        ("show a directory", ["show", str(fixtures)]),
        ("audit a directory", ["audit", str(fixtures)]),
        ("diff across engines", ["diff", str(logs[0]), str(logs[-1])]),
        ("show missing file", ["show", str(fixtures / "does-not-exist.log")]),
        ("repeats missing file", ["repeats", str(fixtures / "does-not-exist.log")]),
        ("demo", ["demo"]),
        ("env --json", ["env", "--json"]),
        ("capture --help", ["capture", "--help"]),
        ("explain list", ["explain"]),
        ("explain a key", ["explain", "kv_cache.tokens"]),
        ("explain a typo", ["explain", "kv_cache.tokenz"]),
        ("explain --json", ["explain", "--json"]),
        ("diff --why", ["diff", str(logs[0]), str(logs[-1]), "--why"]),
        ("diff --why --json", ["diff", str(logs[0]), str(logs[-1]), "--why", "--json"]),
        ("diff --why --markdown", ["diff", str(logs[0]), str(logs[-1]), "--why", "--markdown"]),
        ("series --why", ["series", str(logs[0]), str(logs[1]), str(logs[-1]), "--why"]),
    ]
    return cases


def main_sweep(fixtures: Path, verbose: bool) -> int:
    logs = sorted(Path(p) for p in glob.glob(str(fixtures / "**" / "*.log"), recursive=True))
    print(f"{len(logs)} corpus log(s) under {fixtures}")

    failures: list[tuple[str, str]] = []
    counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        hostile = hostile_files(tmp)
        for log in logs:
            pair = logs[logs.index(log) - 1] if logs.index(log) > 0 else None
            for label, argv in commands(log, pair):
                code, text = run(argv)
                counts[label] = counts.get(label, 0) + 1
                if code < 0:
                    failures.append((f"{label} {log.name}", text.strip().splitlines()[-1]))
                if verbose:
                    print(f"  {code} {label} {log.name}")

        for label, argv in extra_cases(fixtures, logs):
            code, text = run(argv)
            counts[label] = counts.get(label, 0) + 1
            if code < 0:
                failures.append((label, text.strip().splitlines()[-1]))

        for log in hostile:
            for label, argv in commands(log, None):
                code, text = run(argv)
                counts[label] = counts.get(label, 0) + 1
                if code < 0:
                    failures.append((f"{label} {log.name}", text.strip().splitlines()[-1]))

    for label in sorted(counts):
        print(f"  {counts[label]:5d} runs  {label}")

    if failures:
        print(f"\n{len(failures)} case(s) raised:")
        for where, last in failures:
            print(f"  {where}\n      {last}")
        return 1
    print("\nno tracebacks in any case")
    return 0


def main_entry() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", default="tests/fixtures")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    return main_sweep(Path(args.fixtures), args.verbose)


if __name__ == "__main__":
    raise SystemExit(main_entry())
